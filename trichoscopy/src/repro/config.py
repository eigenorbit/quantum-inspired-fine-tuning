"""Training configuration, transcribed from the paper's online supplement.

Every value here comes from supplementary Table S2 ("Training model
parameters") or from the Methods text of the same supplement:

    Adam optimizer, step LR scheduler, cross-entropy loss, batch size 32,
    30 epochs, images resized to 224x224, NVIDIA RTX-3060 GPUs, torch.

Table S2, verbatim:

    Step   Model            Optim  Batch  LR     WeightDecay  StepSize  Gamma
    1      DenseNet169      Adam   32     0.001  0            2         0.7
    1      ResNet152        Adam   32     0.001  0            2         0.7
    1      EfficientNetB0   Adam   32     0.001  0            2         0.7
    2      DenseNet169      Adam   32     0.001  0.001        2         0.7
    2      ResNet152        Adam   32     0.001  0            2         0.7
    2      EfficientNetB0   Adam   32     0.001  0            2         0.7

The single non-zero weight decay (Step-2 / DenseNet169) is not a typo on our
side -- that is how the table reads.

The dropout rates of the 256-unit head are *not* in the supplement; they are
set in `DROPOUT` below.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

# Architectures, in the order the paper's tables list them.
ARCHITECTURES = ("densenet169", "efficientnet_b0", "resnet152")


def variant_name(unique_only: bool = False) -> str:
    """Name of a dataset variant, used for output folders and reports.

    "full" is the delivered data exactly as published; "unique" is the
    deduplication experiment layered on top of it.
    """
    return "unique" if unique_only else "full"

STEPS = (1, 2)

# Output classes: 5 diagnoses in Step-1, 3 activity levels in Step-2.
N_CLASSES = {1: 5, 2: 3}

# Head dropout (not stated in the paper).
DROPOUT = {
    "densenet169": 0.4,
    "efficientnet_b0": 0.2,
    "resnet152": 0.5,
}

# Table S2: weight decay is 0 everywhere except Step-2 / DenseNet169.
WEIGHT_DECAY = {
    (1, "densenet169"): 0.0,
    (1, "efficientnet_b0"): 0.0,
    (1, "resnet152"): 0.0,
    (2, "densenet169"): 0.001,
    (2, "efficientnet_b0"): 0.0,
    (2, "resnet152"): 0.0,
}

# Names used in figures and report tables.
DISPLAY_NAMES = {
    "densenet169": "DenseNet169",
    "efficientnet_b0": "EfficientNetB0",
    "resnet152": "ResNet152",
}

# Index = integer label = image sub-folder name; do not reorder. Note that the
# paper's tables print "FD" for class 2 and "HV" for class 4.
CLASS_NAMES = {
    1: ["AA", "FFA-LPP", "DF", "AGA", "CHP"],
    2: ["AA active", "AA inactive", "AA in regrowth"],
}

# Input size and the ImageNet statistics the backbones were pretrained with.
IMAGE_SIZE = (224, 224)
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    """One training run: a single (step, architecture) pair.

    The defaults reproduce the paper. `amp`, `num_workers`, `seed` and
    `deterministic` have no counterpart in the publication -- they are runtime
    knobs that do not change the optimisation problem (mixed precision changes
    arithmetic precision, not the recipe; see README for the measured effect).
    """

    step: int
    architecture: str

    # --- from Table S2 / the supplement -------------------------------------
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float | None = None  # None -> look up in WEIGHT_DECAY
    lr_step_size: int = 2
    lr_gamma: float = 0.7
    optimizer: str = "adam"
    loss: str = "cross_entropy"
    pretrained: bool = True

    # --- runtime ------------------------------------------------------------
    seed: int = 0
    num_workers: int = 4
    amp: bool = False
    # Pin cuDNN/cuBLAS algorithm choice so two runs of the same command give
    # identical weights. Costs roughly 5-10% throughput. See `repro.seeding`.
    deterministic: bool = True
    # Drop byte-identical training images, keeping one of each. Changes the
    # experiment, not the runtime -- see `repro.dedup`.
    unique_only: bool = False
    device: str = "auto"
    # Evaluate on the non-augmented test images after every epoch. Cheap (717
    # or 240 images) and it is what produces the `best.pth` checkpoint.
    eval_every_epoch: bool = True

    def __post_init__(self) -> None:
        if self.step not in STEPS:
            raise ValueError(f"step must be 1 or 2, got {self.step!r}")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(
                f"unknown architecture {self.architecture!r}; "
                f"expected one of {list(ARCHITECTURES)}"
            )
        if self.weight_decay is None:
            self.weight_decay = WEIGHT_DECAY[(self.step, self.architecture)]

    @property
    def num_classes(self) -> int:
        return N_CLASSES[self.step]

    @property
    def dropout(self) -> float:
        return DROPOUT[self.architecture]

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES[self.architecture]

    @property
    def variant(self) -> str:
        return variant_name(self.unique_only)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
