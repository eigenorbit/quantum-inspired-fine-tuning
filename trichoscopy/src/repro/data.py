"""Datasets, splits and transforms.

Two things about the delivered data drive this module.

**Augmentation is offline.** The folders under ``model_data_step<n>_anon`` are
already augmented and class-balanced (Step-1: 5 x ~7,875 training images;
Step-2: 3 x ~6,359). The flips, rotations and PCA colour jitter the supplement
describes were applied when those files were written, so training applies no
random transform of its own -- only the deterministic resize-to-224 and
ImageNet normalisation that the supplement's last augmentation sentence calls
for ("Finally, the images were resized to 224x224 pixels as input data").
Re-augmenting online would stack a second round on top of the first.

**Test metrics exclude augmented copies.** The paper states it explicitly:
"The performance measures were calculated on the test dataset excluding the
augmented images." Augmented files are named ``<original stem>-aug<n>.jpg``,
so dropping every name containing ``-aug`` yields the evaluation sets:

    Step-1   241 AA / 121 FFA-LPP / 60 DF / 14 AGA / 281 CHP   = 717
    Step-2   78 active / 48 inactive / 114 regrowth            = 240

Step-2 matches the published 240 exactly. Step-1 gives 717 against the
published 641: the two differ only in the CHP/HV class (281 here, 205 in the
paper), because the delivered split is stratified at 20% per class while the
published one held out only 14.6% of CHP. The other four classes agree image
for image. See README, "Known deviations".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

from . import paths
from .config import IMAGE_SIZE, NORM_MEAN, NORM_STD, CLASS_NAMES

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Augmented copies carry a "-aug<n>" suffix on the stem.
_AUG_RE = re.compile(r"-aug\d+$")

# P<patient>_<image>_<diagnosis>_<activity>[-aug<n>], e.g. `P0001_00001_aa_act`
# or `P0002_00002_aga_ina-aug3`.
#
# The image id is not always purely numeric: an id such as `00042v` carries a
# trailing letter. Requiring digits there would leave those files with no
# patient id: an extra "patient" (the empty string) in every count.
_NAME_RE = re.compile(
    r"^(?P<patient>P\d+)_(?P<image>[A-Za-z0-9]+)_(?P<diagnosis>[a-z]+)_(?P<activity>[a-z]+)"
    r"(?:-aug(?P<aug>\d+))?$"
)


def is_augmented(stem: str) -> bool:
    """True for a generated copy, False for one of the original photographs."""
    return _AUG_RE.search(stem) is not None


@dataclass(frozen=True)
class ImageRecord:
    """One file on disk, with everything its name encodes."""

    path: Path
    label: int
    patient: str
    source_stem: str  # the original image this file was derived from
    augmented: bool


def parse_name(stem: str) -> tuple[str, str, bool]:
    """``(patient id, source stem, is augmented)`` for one filename stem.

    Filenames that do not match the documented pattern still work -- they just
    report an empty patient id -- so an unexpected file never aborts a run.
    """
    match = _NAME_RE.match(stem)
    if match is None:
        return "", _AUG_RE.sub("", stem), is_augmented(stem)
    return match.group("patient"), _AUG_RE.sub("", stem), match.group("aug") is not None


def list_images(step: int, split: str, include_augmented: bool = True) -> list[ImageRecord]:
    """Every image of one split, sorted, one sub-folder per integer label.

    Sub-folder names are sorted and mapped to consecutive labels, so with the
    numeric folder names used here folder "0" is always label 0.
    """
    root = paths.split_dir(step, split)
    if not root.is_dir():
        raise FileNotFoundError(f"split directory not found: {root}")

    class_dirs = sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    expected = len(CLASS_NAMES[step])
    if len(class_dirs) != expected:
        raise RuntimeError(
            f"expected {expected} class folders under {root}, found "
            f"{[d.name for d in class_dirs]}"
        )

    records: list[ImageRecord] = []
    for label, class_dir in enumerate(class_dirs):
        for file in sorted(class_dir.iterdir()):
            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            patient, source_stem, augmented = parse_name(file.stem)
            if augmented and not include_augmented:
                continue
            records.append(
                ImageRecord(
                    path=file,
                    label=label,
                    patient=patient,
                    source_stem=source_stem,
                    augmented=augmented,
                )
            )
    if not records:
        raise RuntimeError(f"no images found under {root}")
    return records


class TrichoscopyDataset(Dataset):
    """Wraps a list of :class:`ImageRecord` with a torchvision transform."""

    def __init__(self, records: list[ImageRecord], transform=None):
        self.records = records
        self.transform = transform
        self.labels = [r.label for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = Image.open(record.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, record.label


def base_transform():
    """Resize to 224x224, to tensor, ImageNet normalisation.

    Used for both training and evaluation: the augmentation already happened
    offline, so there is nothing random left to do.
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def class_counts(records: list[ImageRecord], step: int) -> dict[str, int]:
    """``{class name: number of images}`` for a record list."""
    labels = np.asarray([r.label for r in records])
    return {
        name: int((labels == label).sum())
        for label, name in enumerate(CLASS_NAMES[step])
    }


class EpochSeededSampler(Sampler):
    """Shuffle with a permutation that depends only on `(seed, epoch)`.

    `DataLoader(shuffle=True)` is not usable here. It draws the permutation
    from a generator that the loader also dips into for the worker base seed --
    but only on the first iteration inside a given process. With
    `persistent_workers`, epoch 2 of a straight run and epoch 2 of a resumed
    run therefore read that generator at different offsets and shuffle
    differently, which is enough to make a resumed run diverge.

    Building a fresh generator per epoch removes the dependence on iterator
    lifetime, worker persistence and process boundaries alike: epoch `n` is the
    same permutation however the run arrived there.
    """

    def __init__(self, size: int, seed: int):
        self.size = size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        # Arbitrary large odd multiplier; it only has to keep neighbouring
        # seeds from producing overlapping streams.
        generator.manual_seed(self.seed * 1_000_003 + self.epoch)
        yield from torch.randperm(self.size, generator=generator).tolist()

    def __len__(self) -> int:
        return self.size


def _seed_worker(worker_id: int) -> None:  # pragma: no cover - runs in a subprocess
    """Give every dataloader worker a deterministic, distinct seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    import random

    random.seed(seed)


def build_loaders(config, *, limit_train: int | None = None):
    """``(train_loader, test_loader, train_records, test_records)``.

    `limit_train` truncates the training set; it exists for the smoke test and
    must stay ``None`` for a real run.
    """
    transform = base_transform()

    train_records = list_images(config.step, "train", include_augmented=True)
    # Evaluation is on the original photographs only, as the paper specifies.
    test_records = list_images(config.step, "test", include_augmented=False)

    if getattr(config, "unique_only", False):
        from .dedup import unique_records

        train_records, dedup_stats = unique_records(
            train_records, step=config.step, split="train",
            workers=max(4, config.num_workers), verbose=True,
        )
        build_loaders.last_dedup_stats = dedup_stats
    else:
        build_loaders.last_dedup_stats = None

    # Evaluation is always on original photographs only, and those carry no
    # byte-identical duplicates, so deduplication never touches the test side.

    if limit_train is not None:
        # Keep every class represented: take a stride through the sorted list.
        stride = max(1, len(train_records) // limit_train)
        train_records = train_records[::stride][:limit_train]

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    common = dict(
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )

    train_loader = DataLoader(
        TrichoscopyDataset(train_records, transform),
        batch_size=config.batch_size,
        sampler=EpochSeededSampler(len(train_records), config.seed),
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
        **common,
    )
    test_loader = DataLoader(
        TrichoscopyDataset(test_records, transform),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, test_loader, train_records, test_records
