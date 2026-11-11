"""Stronger L2 attacks for `attacks.run`: APGD (Croce & Hein, ICML 2020) and Square
(Andriushchenko, Croce, Flammarion & Hein, ECCV 2020), as in AutoAttack.

Same threat model as the PGD of `attacks.py`: perturbations delta of the 224x224
input with ||delta||_2 <= eps, no clamping to [0, 1] (the certificate is about the
unconstrained ball), one eps per row, every row seeded from its own instance seed
(results do not depend on how instances are batched). Like `pgd_deltas`, each
function only PROPOSES perturbations (phase A, on the network used for gradients,
possibly float16); `attacks.run` re-evaluates them exactly in float32 (phase B) and
only those numbers count.

APGD (`apgd_run`), L2 version of AutoAttack's `APGDAttack.attack_single_run`:
* start on the sphere of radius eps; first step size 2 eps (per row);
* step along the normalised gradient, projection onto the ball, then momentum
  0.75 between the new and the previous iterate, projection again;
* checkpoints after n_iter_2 = 0.22 N iterations, then every k iterations with k
  shrinking by 0.03 N down to 0.06 N; at a checkpoint the step of a row is halved
  and the row restarts from its best point when (i) the loss increased in at most
  75 % of the steps since the last checkpoint, or (ii) the step was not halved at
  the last checkpoint and the best loss has not improved since;
* the returned perturbation is the first iterate that misclassified the row, or
  else the iterate with the highest loss (not the last one, unlike PGD).
Losses: `apgd-ce` cross-entropy; `apgd-dlr` the DLR loss
-(z_y - max_{j!=y} z_j) / (z_pi1 - z_pi3), invariant to the scale of the logits;
`apgd-t` targeted, once per target class among the `target_classes` highest
non-true clean logits (0 = every other class): targeted DLR
-(z_y - z_t) / (z_pi1 - (z_pi3 + z_pi4) / 2), which needs >= 4 classes; with 3
classes (step-2) the targeted margin z_t - z_y instead. Of the targets of one
restart the perturbation with the largest (phase-A) untargeted margin is kept, so
every attack returns one perturbation per restart.

Square (`square_run`), L2 version of AutoAttack's `SquareAttack`: score-based
random search, no gradients (forward passes only):
* start: a grid of square "bumps" (`_eta`) of side h/5 with random signs per
  channel, scaled to ||delta|| = eps;
* each query: a random square window of side s gets a new bump with random signs
  per channel plus its old content, a second random window is emptied, the mass
  is redistributed so that ||delta|| = eps again; the proposal is kept when the
  margin max_{j!=y} z_j - z_y grows (or the image becomes misclassified);
* s follows AutoAttack's schedule from p_init (fraction of pixels in a window,
  default 0.8) down by halvings as the queries run out; a row stops querying as
  soon as it is misclassified. Window positions and signs are drawn per row from
  its own generator.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .core import _norms, margin_loss, multi_logits, unit_directions

ATTACKS = ("pgd", "apgd-ce", "apgd-dlr", "apgd-t", "square")
LABELS = {"pgd": "PGD", "apgd-ce": "APGD-CE", "apgd-dlr": "APGD-DLR", "apgd-t": "APGD-T",
          "square": "Square"}
DEFAULT_STEPS = {"pgd": 20, "apgd-ce": 100, "apgd-dlr": 100, "apgd-t": 100, "square": 0}


# ---------------------------------------------------------------------------
# losses (to be maximised)
# ---------------------------------------------------------------------------

def _gather(z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return z.gather(1, idx[:, None]).squeeze(1)


def dlr_loss(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """-(z_y - max_{j!=y} z_j) / (z_pi1 - z_pi3); needs >= 3 classes."""
    if z.shape[1] < 3:
        raise ValueError("the DLR loss needs at least 3 classes")
    zs, ind = z.sort(dim=1)
    other = torch.where(ind[:, -1] == y, zs[:, -2], zs[:, -1])
    return -(_gather(z, y) - other) / (zs[:, -1] - zs[:, -3] + 1e-12)


def dlr_targeted_loss(z: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """-(z_y - z_t) / (z_pi1 - (z_pi3 + z_pi4) / 2); needs >= 4 classes."""
    if z.shape[1] < 4:
        raise ValueError("the targeted DLR loss needs at least 4 classes")
    zs, _ = z.sort(dim=1)
    return -(_gather(z, y) - _gather(z, t)) / (zs[:, -1] - 0.5 * (zs[:, -3] + zs[:, -4]) + 1e-12)


def attack_loss(name: str, z: torch.Tensor, y: torch.Tensor,
                t: torch.Tensor | None = None) -> torch.Tensor:
    if name == "ce":
        return F.cross_entropy(z, y, reduction="none")
    if name == "dlr":
        return dlr_loss(z, y)
    if name == "dlr-t":
        return dlr_targeted_loss(z, y, t)
    if name == "margin-t":
        return _gather(z, t) - _gather(z, y)
    raise ValueError(f"unknown loss {name!r}")


# ---------------------------------------------------------------------------
# APGD
# ---------------------------------------------------------------------------

def _project(d: torch.Tensor, eps_v: torch.Tensor) -> torch.Tensor:
    """Onto the L2 ball of radius eps (per row)."""
    n = _norms(d).clamp_min(1e-12).view(eps_v.shape)
    return d * torch.clamp(eps_v / n, max=1.0)


def apgd_run(fwd, x, y, eps, *, n_iter: int, loss: str, seeds: list[int],
             target: torch.Tensor | None = None, rho: float = 0.75) -> torch.Tensor:
    """One APGD run per row; returns delta (first misclassifying iterate, else the
    iterate of highest loss). `fwd`: images -> logits, differentiable."""
    view = (-1,) + (1,) * (x.dim() - 1)
    eps_v = eps.view(view)
    n_iter_2 = max(int(0.22 * n_iter), 1)
    n_iter_min = max(int(0.06 * n_iter), 1)
    size_decr = max(int(0.03 * n_iter), 1)

    def loss_grad(d):
        d = d.detach().requires_grad_(True)
        z = fwd(x + d)
        lo = attack_loss(loss, z, y, target)
        (g,) = torch.autograd.grad(lo.sum(), d)
        return lo.detach(), g.detach(), z.detach()

    d = unit_directions(x.shape, seeds).to(x.device) * eps_v
    lo, g, z = loss_grad(d)
    success = margin_loss(z, y) > 0
    d_adv = d.clone()
    d_best, g_best, lo_best = d.clone(), g.clone(), lo.clone()
    step = 2.0 * eps.clone()
    d_old = d.clone()
    hist = [lo.clone()]                      # loss of every iterate, the start included
    k, counter = n_iter_2, 0
    lo_best_last = lo_best.clone()
    reduced_last = torch.ones_like(success)
    for i in range(n_iter):
        a = 0.75 if i > 0 else 1.0
        g_n = g / _norms(g).clamp_min(1e-12).view(view)
        d1 = _project(d + step.view(view) * g_n, eps_v)
        d_new = _project(d + a * (d1 - d) + (1 - a) * (d - d_old), eps_v)
        d_old, d = d, d_new
        lo, g, z = loss_grad(d)
        hit = (margin_loss(z, y) > 0) & ~success
        d_adv[hit] = d[hit]
        success |= hit
        better = lo > lo_best
        d_best[better], g_best[better], lo_best[better] = d[better], g[better], lo[better]
        hist.append(lo.clone())
        hist = hist[-(n_iter_2 + 1):]
        counter += 1
        if counter == k:
            ups = sum((hist[-1 - j] > hist[-2 - j]).float() for j in range(k))
            reduce = (ups <= k * rho) | (~reduced_last & (lo_best_last >= lo_best))
            reduced_last, lo_best_last = reduce.clone(), lo_best.clone()
            if reduce.any():
                step[reduce] /= 2.0
                d[reduce], g[reduce] = d_best[reduce], g_best[reduce]
            k = max(k - size_decr, n_iter_min)
            counter = 0
    return torch.where(success.view(view), d_adv, d_best).detach()


def target_order(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Non-true classes per row, highest clean logit first: (rows, classes - 1)."""
    masked = z.masked_fill(F.one_hot(y, z.shape[1]).bool(), float("-inf"))
    return masked.argsort(dim=1, descending=True)[:, :-1]


# ---------------------------------------------------------------------------
# Square
# ---------------------------------------------------------------------------

def p_selection(p_init: float, it: int, n_queries: int) -> float:
    """AutoAttack's schedule of the window size (fraction of the pixels)."""
    it = int(it / n_queries * 10000)
    for lim, div in ((10, 1), (50, 2), (200, 4), (500, 8), (1000, 16), (2000, 32),
                     (4000, 64), (6000, 128), (8000, 256)):
        if it <= lim:
            return p_init / div
    return p_init / 512


def _eta_rectangles(h: int, w: int) -> torch.Tensor:
    delta = torch.zeros(h, w)
    if h == 0 or w == 0:
        return delta
    xc, yc = h // 2 + 1, w // 2 + 1
    c0, c1 = xc - 1, yc - 1
    for counter in range(max(xc, yc)):
        delta[max(c0, 0):min(c0 + 2 * counter + 1, h),
              max(c1, 0):min(c1 + 2 * counter + 1, w)] += 1.0 / (counter + 1) ** 2
        c0, c1 = c0 - 1, c1 - 1
    return delta / delta.pow(2).sum().sqrt()


def _eta(s: int, gen: torch.Generator) -> torch.Tensor:
    """A square bump of side s: + on one half, - on the other, unit L2 norm,
    transposed with probability 1/2."""
    delta = torch.zeros(s, s)
    delta[:s // 2] = _eta_rectangles(s // 2, s)
    delta[s // 2:] = -_eta_rectangles(s - s // 2, s)
    delta = delta / delta.pow(2).sum().sqrt()
    if torch.rand(1, generator=gen).item() > 0.5:
        delta = delta.t()
    return delta


def _signs(c: int, gen: torch.Generator) -> torch.Tensor:
    return (torch.randint(0, 2, (c, 1, 1), generator=gen) * 2 - 1).float()


def _randint(lo: int, hi: int, gen: torch.Generator) -> int:
    """Uniform integer in [lo, hi]."""
    return int(torch.randint(lo, hi + 1, (1,), generator=gen).item())


def square_init(shape, eps: torch.Tensor, gens: list[torch.Generator]) -> torch.Tensor:
    _, c, h, w = shape
    s = max(h // 5, 1)
    nh, nw = h // s, w // s
    oh, ow = (h - nh * s) // 2, (w - nw * s) // 2
    out = torch.zeros(shape)
    for b, gen in enumerate(gens):
        for i in range(nh):
            for j in range(nw):
                out[b, :, oh + i * s:oh + (i + 1) * s, ow + j * s:ow + (j + 1) * s] += \
                    _eta(s, gen).view(1, s, s) * _signs(c, gen)
    return out * (eps.cpu() / _norms(out).clamp_min(1e-12)).view(-1, 1, 1, 1)


@torch.no_grad()
def square_run(fwd, x, y, eps, *, n_queries: int, p_init: float, seeds: list[int]
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Square-L2 per row; returns (delta, queries used). `fwd(images, rows)` -> logits,
    `rows` the indices (into x) of the images passed: only the rows still to fool
    are queried."""
    if x.dim() != 4:
        raise ValueError("Square works on images (rows, channels, height, width)")
    _, c, h, w = x.shape
    n_feat = c * h * w
    gens = [torch.Generator().manual_seed(int(s) % (2**63)) for s in seeds]
    eps_list = eps.tolist()
    delta = square_init(x.shape, eps, gens).to(x.device)
    margin = margin_loss(fwd(x + delta, torch.arange(len(x), device=x.device)), y)
    queries = torch.ones(len(x), dtype=torch.long)
    for it in range(n_queries - 1):
        active = (margin <= 0).nonzero(as_tuple=True)[0]
        if len(active) == 0:
            break
        s = max(int(round(math.sqrt(p_selection(p_init, it, n_queries) * n_feat / c))), 3)
        s = min(s + (s % 2 == 0), h, w)
        cand = delta[active].clone()
        for k, b in enumerate(active.tolist()):
            gen, e = gens[b], eps_list[b]
            dc = cand[k]
            vh, vw = _randint(0, h - s, gen), _randint(0, w - s, gen)
            vh2, vw2 = _randint(0, h - s, gen), _randint(0, w - s, gen)
            win1 = dc[:, vh:vh + s, vw:vw + s]
            norm_win1 = win1.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt()
            mask = torch.zeros_like(dc[:1])
            mask[:, vh:vh + s, vw:vw + s] = 1.0
            mask[:, vh2:vh2 + s, vw2:vw2 + s] = 1.0
            norm_img = dc.norm()
            norm_masked = (dc * mask).norm()
            new = (_eta(s, gen).view(1, s, s) * _signs(c, gen)).to(dc) + win1 / (1e-12 + norm_win1)
            budget = (torch.clamp(e ** 2 - norm_img ** 2, min=0.0) / c + norm_masked ** 2).sqrt()
            new = new / (1e-12 + new.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt()) * budget
            dc[:, vh2:vh2 + s, vw2:vw2 + s] = 0.0
            dc[:, vh:vh + s, vw:vw + s] = new
            cand[k] = e * dc / (1e-12 + dc.norm())
        m_new = margin_loss(fwd(x[active] + cand, active), y[active])
        queries[active.cpu()] += 1
        keep = (m_new > margin[active]) | (m_new > 0)
        rows = active[keep]
        delta[rows] = cand[keep]
        margin[rows] = m_new[keep]
    return delta, queries


# ---------------------------------------------------------------------------
# the phase-A entry point used by attacks.run
# ---------------------------------------------------------------------------

def deltas(cfg, grad_fn, classifiers, model_idx, x, y, eps, seeds: list[int]
           ) -> list[torch.Tensor]:
    """One perturbation per restart and row for cfg.attack (on the CPU)."""
    def fwd(xx):
        return multi_logits(classifiers, model_idx, grad_fn(xx))

    out = []
    for r in range(cfg.restarts):
        rs = [s * 1_000 + 100 + r for s in seeds]
        if cfg.attack in ("apgd-ce", "apgd-dlr"):
            d = apgd_run(fwd, x, y, eps, n_iter=cfg.steps, loss=cfg.attack[5:], seeds=rs)
        elif cfg.attack == "apgd-t":
            with torch.no_grad():
                z0 = fwd(x)
            order = target_order(z0, y)
            n_t = order.shape[1] if cfg.target_classes <= 0 else min(cfg.target_classes,
                                                                     order.shape[1])
            loss = "dlr-t" if z0.shape[1] >= 4 else "margin-t"
            d, best = None, None
            for ti in range(n_t):
                dt = apgd_run(fwd, x, y, eps, n_iter=cfg.steps, loss=loss,
                              seeds=[s + 10 * ti for s in rs], target=order[:, ti])
                with torch.no_grad():
                    m = margin_loss(fwd(x + dt), y)
                if d is None:
                    d, best = dt, m
                else:
                    take = m > best
                    d[take], best[take] = dt[take], m[take]
        elif cfg.attack == "square":
            d, _ = square_run(lambda xx, rows: multi_logits(classifiers, model_idx[rows],
                                                            grad_fn(xx)),
                              x, y, eps, n_queries=cfg.queries, p_init=cfg.p_init,
                              seeds=[s + 700 for s in rs])
        else:
            raise ValueError(f"unknown attack {cfg.attack!r}")
        out.append(d.detach().cpu())
    return out
