"""Accuracy against eps of the attack folders: random noise, the attack, the certificate."""

from __future__ import annotations

import numpy as np

from ..display import ORDER, style


def curves(rows: list[dict], lpre: str = "official", grid=None) -> dict:
    """Per run: eps points and random / PGD / certified accuracy (fractions)."""
    grid = np.linspace(0, 0.5, 201) if grid is None else grid
    out = {}
    for run in sorted({r["run"] for r in rows}, key=lambda n: ORDER.index(n)
                      if n in ORDER else 99):
        rs = [r for r in rows if r["run"] == run]
        eps = sorted({r["eps"] for r in rs})
        per_image = {}                                       # image -> (correct, radius)
        for r in rs:
            per_image[r["image"]] = (r["clean_correct"], r[f"radius_{lpre}"])
        correct = np.array([c for c, _ in per_image.values()])
        radius = np.array([rad for _, rad in per_image.values()])
        clean = float(correct.mean())
        pts = [0.0] + eps
        rnd = [clean] + [float(np.mean([r["random_survived"] for r in rs if r["eps"] == e]))
                         for e in eps]
        pgd = [clean] + [float(np.mean([r["pgd_robust"] for r in rs if r["eps"] == e]))
                         for e in eps]
        cert = [float(np.mean(correct & (radius > e))) if e > 0 else clean for e in grid]
        out[run] = {"eps": pts, "random": rnd, "pgd": pgd, "grid": list(grid),
                    "certified": cert, "n": len(per_image)}
    return out



def _panels(axes, cv: dict, attacked_eps: list[float], attack: str = "pgd"):
    import matplotlib.ticker as mt

    from .strong import LABELS

    for run, c in cv.items():
        st = style(run)
        kw = dict(color=st["color"], ls=st["ls"], lw=1.8)
        axes[0].plot(c["eps"], c["random"], marker="o", ms=4, label=st["label"], **kw)
        axes[1].plot(c["eps"], c["pgd"], marker="o", ms=4, label=st["label"], **kw)
        axes[2].plot(c["grid"], c["certified"], label=st["label"], **kw)
        at = [c["certified"][int(np.argmin(np.abs(np.array(c["grid"]) - e)))]
              for e in attacked_eps]
        axes[2].plot(attacked_eps, at, "o", ms=4, color=st["color"])
    titles = ("Random direction of norm $\\epsilon$",
              f"Worst direction of norm $\\epsilon$ — {LABELS[attack]}-L2",
              "Certified: every direction of norm $\\epsilon$")
    ticks = [0.0] + list(attacked_eps)
    for ax, t in zip(axes, titles):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel(r"perturbation norm $\|d\|_2 = \epsilon$  (224×224, pixels in [0,1])",
                      fontsize=8)
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0"] + [f"{round(e * 255)}/255" for e in attacked_eps], fontsize=8)
        ax.set_xlim(-0.01, max(ticks) * 1.08)
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0))
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(-0.03, 1.03)
