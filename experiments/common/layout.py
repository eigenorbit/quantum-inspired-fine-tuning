"""Names of the output folders and files of the experiments -- the one place.

Each package writes only below ``<DERMA_REPRO_OUT>/<package>/``::

    backbone/                                   caches of the frozen network
    features/step_<n>/                          feature cache (resnet: .../<dtype>/)
    tuning/step_<n><bs>/                        selection.json, tuning_step<n><bs>.md,
        val_split.json (step_<n>/ only: shared)   <mode>/lr<lr>/seed<s>/ (grid runs)
    tuning/step_<n><bs><loss tag>/              the same, tuned at another eps_train /
                                                with an eps warm-up (bronet_mlp.tune)
    runs/step_<n><bs>/<mode>[_<variant>]/seed<s>/
        checkpoints/{last,best,final}.pt, metrics[_<which>].json,
        predictions_<which>.csv, <arrays>_<which>.npz, attacks/<attack>_l2_<which>.csv
    results/                                    summary_step<n><bs>[_last].md, ...
        attacks/step_<n><bs>_<checkpoint>[_eps..][_<attack>]/
        attacks/step_<n><bs>_<checkpoint>_mineps[_<attack>]/
        lossy/step_<n>_<checkpoint>_<at>/        coverage of lossy handling by the
                                                certificate, eps-sweep tables (bronet_mlp.lossy)
        figures/
    lossy_features/step_<n>/<at>/<op>.pt        phi of the transformed test images
    conv_features/step_<n>/last<k>_<dtype>/      inputs of the last k BRO blocks (conv_adapt)
    conv_adapt/step_<n>/last<k>/<mlp mode>[_<variant>]/seed<s>/
                                                runs with rotations inside the last k BRO
                                                blocks; summaries in results/conv_adapt/
    conv_adapt/tuning/step_<n>/last<k>/<mlp mode>[_<variant>]/conv<lr>_mlp<lr>/seed0/
                                                their lr tuning; selection.json next to it
    eps_sweep/step_<n>/eps<k>[_wu..]/<mode>[_<variant>]/seed<s>/
                                                runs with eps_train = k/255 (k != 36, or
                                                any k with an eps warm-up)

Suffix rules (they keep the folders of every protocol apart and the original
folders where they were):
    <bs>        '' for the default training batch 256, else '_bs<B>'
    [_last]     '' for the best-validation checkpoint, '_last' for the final one
    [_<attack>] '' for PGD (the first attack, made before the others), else '_<attack>'
    [_eps..]    resnet_adapt only: '' for the BRONet epsilons, else '_eps<a>-<b>-...'
    <loss tag>  bronet_mlp tuning: '' for eps_train 36/255 without warm-up, else
                '[_eps<k>][_wu<fraction>[from<k0>]]' (see loss_tag)
    [_wu..]     '' without an eps warm-up, else '_wu<fraction>[from<k0>]' (warmup_tag)
Order inside a name: batch tag, checkpoint, eps tag, attack tag; the batch tag
comes before '_last'.

Standard library only; the output root is read when a path is asked for, so a
changed DERMA_REPRO_OUT (tests) takes effect at once.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_BATCH_SIZE = 256
DEFAULT_EPS_255 = 36             # eps_train of the protocol, in units of 1/255

SELECTION_FILE = "selection.json"
VAL_SPLIT_FILE = "val_split.json"
CHECKPOINTS_DIR = "checkpoints"
HEADLINE_FILE = "metrics.json"


# ---------------------------------------------------------------------------
# suffixes
# ---------------------------------------------------------------------------

def batch_tag(batch_size: int = DEFAULT_BATCH_SIZE) -> str:
    """'' for the default training batch (256), else '_bs<B>'."""
    return "" if batch_size == DEFAULT_BATCH_SIZE else f"_bs{batch_size}"


def attack_tag(attack: str) -> str:
    """'' for PGD (the original folders), else '_<attack>'."""
    return "" if attack == "pgd" else f"_{attack}"


def last_tag(which: str) -> str:
    """'' for the best-validation checkpoint, '_last' for the final one."""
    return "" if which == "best" else "_last"


def _k255(eps: float) -> str:
    """eps in units of 1/255, as in folder names: 195/255 -> '195'."""
    return f"{round(eps * 255, 6):g}"


def warmup_tag(eps_warmup: float = 0.0, eps_start: float = 0.0) -> str:
    """'' without warm-up, else '_wu<fraction of the epochs>' and, for a non-zero
    starting eps, 'from<k0>' (k0 in units of 1/255): '_wu0.33', '_wu0.33from9'."""
    if not eps_warmup:
        return ""
    return f"_wu{eps_warmup:g}" + (f"from{_k255(eps_start)}" if eps_start else "")


def loss_tag(eps_train: float, eps_warmup: float = 0.0, eps_start: float = 0.0) -> str:
    """'' for the protocol's eps_train (36/255, no warm-up); else '_eps<k>' when
    eps_train differs, followed by the warm-up tag: '_eps195_wu0.33'."""
    eps = "" if _k255(eps_train) == f"{DEFAULT_EPS_255:g}" else f"_eps{_k255(eps_train)}"
    return eps + warmup_tag(eps_warmup, eps_start)


def step_name(step: int, batch_size: int = DEFAULT_BATCH_SIZE) -> str:
    return f"step_{step}{batch_tag(batch_size)}"


def run_name(mode: str, variant: str = "best") -> str:
    """Folder name of a (mode, tuning variant): `lora` for best, `lora_stable`."""
    return mode if variant == "best" else f"{mode}_{variant}"


def seed_name(seed: int) -> str:
    return f"seed{seed}"


# ---------------------------------------------------------------------------
# files of one run
# ---------------------------------------------------------------------------

def checkpoint_file(which: str) -> str:
    """In <run>/checkpoints/: last.pt, best.pt, final.pt."""
    return f"{which}.pt"


def epoch_checkpoint_file(epoch: int) -> str:
    """--keep-every copies; `epoch` counts from 1."""
    return f"epoch_{epoch:03d}.pt"


def metrics_file(which: str) -> str:
    return f"metrics_{which}.json"


def predictions_file(which: str) -> str:
    return f"predictions_{which}.csv"


def arrays_file(kind: str, which: str) -> str:
    """Per-image arrays of one checkpoint: kind `certificates` (bronet_mlp) or
    `outputs` (resnet_adapt)."""
    return f"{kind}_{which}.npz"


def run_attack_csv(attack: str, which: str) -> str:
    """Per-run copy of the attack rows, in <run>/attacks/: pgd_l2_<which>.csv."""
    return f"{attack.replace('-', '_')}_l2_{which}.csv"


def run_mineps_csv(attack: str, which: str) -> str:
    """Per-run copy of the min-eps rows, in <run>/attacks/."""
    return f"min_eps_{attack.replace('-', '_')}_{which}.csv"


# ---------------------------------------------------------------------------
# result files
# ---------------------------------------------------------------------------

def tuning_summary_file(step: int, batch_size: int = DEFAULT_BATCH_SIZE) -> str:
    return f"tuning_step{step}{batch_tag(batch_size)}.md"


def summary_file(step: int, batch_size: int, which: str) -> str:
    return f"summary_step{step}{batch_tag(batch_size)}{last_tag(which)}.md"


def mcnemar_file(step: int, batch_size: int, which: str, certified: bool = False) -> str:
    kind = "mcnemar_cert" if certified else "mcnemar"
    return f"{kind}_step{step}{batch_tag(batch_size)}{last_tag(which)}.csv"


def adaptation_file(step: int, batch_size: int) -> str:
    """bronet_mlp: one row per run and checkpoint."""
    return f"adaptation_step{step}{batch_tag(batch_size)}.csv"


def runs_table_file(step: int, batch_size: int) -> str:
    """resnet_adapt: one row per run and checkpoint."""
    return f"runs_step{step}{batch_tag(batch_size)}.csv"


def lipschitz_file(step: int, batch_size: int, ext: str) -> str:
    """resnet_adapt spectrum: lipschitz_step<n><bs>.json / .md."""
    return f"lipschitz_step{step}{batch_tag(batch_size)}.{ext}"


def attack_summary_file(step: int, ext: str, kind: str = "attacks") -> str:
    """Inside an attack folder: attacks_step<n>.csv/.md, min_eps_step<n>.csv/.md (kind
    `min_eps`). No batch or attack tag: the folder carries both."""
    return f"{kind}_step{step}.{ext}"


def attack_dir_name(step: int, batch_size: int, checkpoint: str, attack: str = "pgd",
                    eps: str = "") -> str:
    """`checkpoint` is the requested one (headline | best | final); `eps` an eps tag."""
    return f"{step_name(step, batch_size)}_{checkpoint}{eps}{attack_tag(attack)}"


def mineps_dir_name(step: int, batch_size: int, checkpoint: str, attack: str = "pgd") -> str:
    return f"{step_name(step, batch_size)}_{checkpoint}_mineps{attack_tag(attack)}"


def attack_figure(step: int | None, batch_size: int, checkpoint: str, attack: str = "pgd",
                  eps: str = "") -> str:
    """attacks_step<n>... for one step, attacks_both_steps... for step None."""
    where = "both_steps" if step is None else f"step{step}"
    return f"attacks_{where}{batch_tag(batch_size)}_{checkpoint}{eps}{attack_tag(attack)}.png"


def certified_figure(step: int, batch_size: int, lpre: str, which: str) -> str:
    return (f"certified_accuracy_step{step}{batch_tag(batch_size)}_{lpre}_"
            f"{'best' if which == 'best' else 'last'}.png")


def certified_curves_figure(batch_size: int, which: str, lpre: str) -> str:
    return f"certified_curves_both_steps{batch_tag(batch_size)}_{which}_{lpre}.png"


def lipschitz_growth_figure(batch_size: int, which: str) -> str:
    return f"lipschitz_growth{batch_tag(batch_size)}_{which}.png"


# ---------------------------------------------------------------------------
# the folders of one package
# ---------------------------------------------------------------------------

from .locations import repro_out  # noqa: E402


class Tree:
    """The output folders of one package, <DERMA_REPRO_OUT>/<package>/..."""

    def __init__(self, package: str):
        self.package = package

    def root(self) -> Path:
        return repro_out() / self.package

    def backbone_dir(self) -> Path:
        return self.root() / "backbone"

    def features_dir(self, step: int) -> Path:
        return self.root() / "features" / f"step_{step}"

    def runs_dir(self, step: int, batch_size: int = DEFAULT_BATCH_SIZE) -> Path:
        return self.root() / "runs" / step_name(step, batch_size)

    def run_dir(self, step: int, mode: str, seed: int, variant: str = "best",
                batch_size: int = DEFAULT_BATCH_SIZE) -> Path:
        return self.runs_dir(step, batch_size) / run_name(mode, variant) / seed_name(seed)

    def tuning_dir(self, step: int, batch_size: int = DEFAULT_BATCH_SIZE, tag: str = "") -> Path:
        """`tag`: a loss_tag ('' = the protocol's tuning)."""
        return self.root() / "tuning" / f"{step_name(step, batch_size)}{tag}"

    def tune_run_dir(self, step: int, mode: str, lr: float, seed: int,
                     batch_size: int = DEFAULT_BATCH_SIZE, tag: str = "") -> Path:
        return self.tuning_dir(step, batch_size, tag) / mode / f"lr{lr:g}" / seed_name(seed)

    def results_dir(self) -> Path:
        return self.root() / "results"

    def figures_dir(self) -> Path:
        return self.results_dir() / "figures"

    def attack_dir(self, step: int, batch_size: int, checkpoint: str, attack: str = "pgd",
                   eps: str = "") -> Path:
        return (self.results_dir() / "attacks"
                / attack_dir_name(step, batch_size, checkpoint, attack, eps))

    def mineps_dir(self, step: int, batch_size: int, checkpoint: str,
                   attack: str = "pgd") -> Path:
        return self.results_dir() / "attacks" / mineps_dir_name(step, batch_size, checkpoint, attack)

    def lossy_dir(self, step: int, checkpoint: str, at: str) -> Path:
        """`at`: where the operations are applied (native | input)."""
        return self.results_dir() / "lossy" / f"{step_name(step)}_{checkpoint}_{at}"

    def lossy_features_file(self, step: int, at: str, op: str) -> Path:
        return self.root() / "lossy_features" / f"step_{step}" / at / f"{op}.pt"

    # --- bronet_mlp.conv_adapt: rotations inside the last BRO blocks ---

    def conv_features_dir(self, step: int, blocks: int, dtype: str) -> Path:
        """Cached inputs of the last `blocks` BRO blocks: train.npy, test.npy, index.json."""
        return self.root() / "conv_features" / f"step_{step}" / f"last{blocks}_{dtype}"

    def conv_run_dir(self, step: int, blocks: int, mlp_mode: str, seed: int,
                     variant: str = "best") -> Path:
        return (self.root() / "conv_adapt" / f"step_{step}" / f"last{blocks}"
                / run_name(mlp_mode, variant) / seed_name(seed))

    def conv_tuning_dir(self, step: int, blocks: int) -> Path:
        return self.root() / "conv_adapt" / "tuning" / f"step_{step}" / f"last{blocks}"

    def conv_tune_run_dir(self, step: int, blocks: int, mlp_mode: str, variant: str,
                          lr_conv: float, lr_mlp: float, seed: int = 0) -> Path:
        return (self.conv_tuning_dir(step, blocks) / run_name(mlp_mode, variant)
                / f"conv{lr_conv:g}_mlp{lr_mlp:g}" / seed_name(seed))

    def conv_attack_dir(self, step: int, blocks: int, checkpoint: str, attack: str = "pgd",
                        kind: str = "attacks") -> Path:
        """Attacks (kind `attacks`) or the eps* search (kind `mineps`) on the conv runs:
        one sub-folder per run (<run>/seed<s>/) and the combined tables."""
        return (self.results_dir() / "conv_adapt"
                / f"{kind}_step{step}_last{blocks}_{checkpoint}{attack_tag(attack)}")

    def conv_lossy_dir(self, step: int, blocks: int, checkpoint: str, place: str) -> Path:
        """Coverage of lossy handling (JPEG ...) by the certificate of the conv runs;
        `place` as bronet_mlp.lossy.place (native, native_renorm-<mode>, input)."""
        return (self.results_dir() / "conv_adapt"
                / f"lossy_step{step}_last{blocks}_{checkpoint}_{place}")

    def conv_export_file(self, step: int, blocks: int) -> Path:
        return self.root() / f"conv_adapt_results_step{step}_last{blocks}.zip"

    def conv_summary_file(self, step: int, blocks: int, which: str) -> Path:
        return (self.results_dir() / "conv_adapt"
                / f"summary_step{step}_last{blocks}{last_tag(which)}.md")

    def eps_sweep_run_dir(self, step: int, eps_255: float, mode: str, seed: int,
                          variant: str = "best", warmup: str = "") -> Path:
        """`warmup`: a warmup_tag ('' = eps at full strength from the first step)."""
        return (self.root() / "eps_sweep" / f"step_{step}" / f"eps{eps_255:g}{warmup}"
                / run_name(mode, variant) / seed_name(seed))


BRONET = Tree("bronet_mlp")
RESNET = Tree("resnet_adapt")
