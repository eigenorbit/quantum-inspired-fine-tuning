"""Records and transform: the `trichoscopy/` protocol, variant ``unique``.

Train = the delivered ``train/`` folder with byte-identical copies removed
(within a class, `repro.dedup.unique_records`, same cache key "train" as
`repro.data.build_loaders`); test = the original photographs in ``test/``.
No cross-validation. Expected sizes:

    step-1  train 31,665 (of 39,368)   test 717
    step-2  train 13,707 (of 19,076)   test 240 (78 / 48 / 114)

The transform differs from `repro.data.base_transform`: BRONet takes pixels in
[0, 1] and subtracts 0.5 itself, so there is no ImageNet Normalize (with it the
Lipschitz constant w.r.t. pixels would be ~36 instead of ~8).
"""

from __future__ import annotations

from pathlib import Path

from .io import sha256_json
from .locations import data_root

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from repro.config import CLASS_NAMES, N_CLASSES

# Bump whenever the pixels fed to BRONet change; part of the feature cache key.
TRANSFORM_VERSION = "pil-bilinear-resize-224x224+to_tensor[0,1]+no-normalize/v1"
IMAGE_SIZE = (224, 224)

EXPECTED_SIZES = {1: {"train": 31_665, "train_before": 39_368, "test": 717},
                  2: {"train": 13_707, "train_before": 19_076, "test": 240}}


def bronet_transform():
    return transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])


def relative_path(path: Path) -> str:
    """Key of a file in every cache and manifest: posix path below DATA_ROOT."""
    return Path(path).resolve().relative_to(data_root().resolve()).as_posix()


def records(step: int, logger=None):
    """``(train_records, test_records, dedup_stats)``, exactly as repro builds them."""
    # imported here, not at the top: `repro.data` needs the locations when imported
    from repro import data as repro_data
    from repro import dedup as repro_dedup

    train = repro_data.list_images(step, "train", include_augmented=True)
    test = repro_data.list_images(step, "test", include_augmented=False)
    train, stats = repro_dedup.unique_records(train, step=step, split="train",
                                              verbose=logger is not None)
    if logger:
        logger.info(f"step-{step}: train {stats['before']:,} -> {stats['after']:,} unique, "
                    f"test {len(test):,} originals, "
                    f"cross-class groups {stats['cross_class_groups']}")
    return train, test, stats


def manifest(step: int, train, test, stats) -> dict:
    """Everything that identifies the data of a run; its sha256 goes into checkpoints."""
    def rows(recs):
        return [[relative_path(r.path), int(r.label)] for r in recs]

    tr, te = rows(train), rows(test)
    return {"step": step, "class_names": CLASS_NAMES[step], "num_classes": N_CLASSES[step],
            "protocol": "trichoscopy/unique: delivered train (dedup within class) / "
                        "test originals",
            "n_train": len(tr), "n_test": len(te),
            "train_sha256": sha256_json(tr), "test_sha256": sha256_json(te),
            "dedup_stats": stats, "train": tr, "test": te}


def manifest_sha256(man: dict) -> str:
    return sha256_json({"train": man["train_sha256"], "test": man["test_sha256"]})


def photo_key(rel_path: str) -> str:
    """``P<patient>_<image id>`` of a file: identifies the photograph.

    Coarser than `source_stem` on purpose: a photograph relabelled into two
    classes (e.g. P0001_00001 as aa_act and aa_ina) has two source stems but is
    one image, and must not straddle the train/validation boundary.
    """
    stem = Path(rel_path).stem
    parts = stem.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else stem


def validation_split(rows: list[list], fraction: float = 0.1, seed: int = 0):
    """``(fit_idx, val_idx)`` over manifest train rows ``[rel_path, label]``.

    Grouped by photograph (all augmented copies of one photo on the same side)
    and stratified by class: within each class the photographs are shuffled by
    a seeded generator and taken in that order until the held-out files reach
    `fraction` of the class. A photograph filed under several classes is
    assigned to its majority class. The shuffle is torch's CPU `randperm`; the
    resulting split is saved with its sha256 and checked on reuse. Used only for choosing hyper-parameters, never the test set.
    """
    from collections import Counter, defaultdict

    by_group: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_group[photo_key(r[0])].append(i)
    group_class = {g: Counter(int(rows[i][1]) for i in idx).most_common(1)[0][0]
                   for g, idx in by_group.items()}
    val: list[int] = []
    for c in sorted(set(group_class.values())):
        groups = sorted(g for g, gc in group_class.items() if gc == c)
        g = torch.Generator().manual_seed(seed * 1_000_003 + c)
        order = [groups[k] for k in torch.randperm(len(groups), generator=g).tolist()]
        target = fraction * sum(len(by_group[x]) for x in groups)
        taken = 0
        for grp in order:
            if taken >= target:
                break
            val.extend(by_group[grp])
            taken += len(by_group[grp])
    val_set = set(val)
    return [i for i in range(len(rows)) if i not in val_set], sorted(val)


class ImageDataset(Dataset):
    """(tensor in [0,1], index) -- the index maps back to the record list."""

    def __init__(self, paths: list[Path]):
        self.paths = list(paths)
        self.transform = bronet_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as image:
            return self.transform(image.convert("RGB")), i


def load_images(paths, device=None) -> torch.Tensor:
    x = torch.stack([ImageDataset(paths)[i][0] for i in range(len(paths))])
    return x.to(device) if device is not None else x
