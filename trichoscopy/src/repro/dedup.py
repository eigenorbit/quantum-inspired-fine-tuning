"""Find and drop byte-identical images in the training splits.

    python -m repro.dedup              # report both steps, write manifests
    python -m repro.dedup --step 1

The augmentation space is finite -- eight orientations, and the PCA colour
jitter is applied only some of the time -- so at the multipliers needed to
balance the small classes the generator started emitting copies that are
identical byte for byte. Measured on the delivered data:

    step-1 train   39,368 files ->  31,665 unique   (19.6% duplicates)
    step-2 train   19,076 files ->  13,707 unique   (28.1% duplicates)

Per class it is very uneven. AGA, the smallest class, is more than half
duplicates (7,873 -> 3,572, up to 22 identical copies of one frame); CHP, the
largest, is 2.5%. Duplicates inflate a class's weight in the loss without
adding information, and they fall hardest on exactly the class that needed
help.

The *test* splits are unaffected: evaluation uses original photographs only,
and all 717 (step-1) and 240 (step-2) of those are unique. A deduplicated run
is therefore directly comparable with the baseline -- same test set, same
metrics, only the training data changes.

Two things this module deliberately does not do:

* It does not delete or copy anything. Selection happens in memory when the
  loaders are built, so there is no second copy of the dataset on disk and no
  way for the two to drift apart. `--write-manifest` records exactly which
  files were kept, so a run stays as inspectable as a separate folder tree.
* It dedups *within a class*, never across. One pair in step-2 train is
  byte-identical across two different classes -- the same photograph filed
  under "active" and under "inactive", contradictory labels. Dropping one of
  them silently would be a labelling decision, not a deduplication, so both
  are kept and the conflict is reported instead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import console, paths
from .config import CLASS_NAMES, STEPS

_CHUNK = 1 << 20


def content_hash(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(step: int, split: str) -> Path:
    return paths.OUTPUT_ROOT / "cache" / f"hashes_step{step}_{split}.json"


# Bumped when `_record_key` changes shape, so a cache written by an older
# version is discarded instead of half-matching.
_KEY_FORMAT = 2


def _record_key(record) -> str:
    """Identity of one file within its step: ``<split>/<class>/<filename>``.

    The split folder has to be part of this. The class folders inside
    ``train/`` and ``test/`` share names ("0".."4"), so in a record list that
    spans both, a key built from the class folder alone stops being unique the
    moment the same filename exists on both sides. `hash_records` would then
    hand one of the two files the other's digest, and `unique_records` would
    drop a file that is not a duplicate -- silently, with nothing in the output
    to show for it.

    Nothing in the delivered data collides today (verified over both steps).
    This makes it impossible rather than merely unlikely.
    """
    return (f"{record.path.parent.parent.name}/{record.path.parent.name}/"
            f"{record.path.name}")


def _validity(records) -> dict:
    """Cheap signature of a file list, to detect that the data changed."""
    return {
        "count": len(records),
        "bytes": sum(record.path.stat().st_size for record in records),
        "key_format": _KEY_FORMAT,
    }


def hash_records(records, *, step: int, split: str, workers: int = 8,
                 use_cache: bool = True, verbose: bool = False) -> dict[str, str]:
    """``{_record_key(record): sha256}`` for every record.

    Hashing 39k files takes about half a minute, so the result is cached under
    ``cache/`` and re-used as long as the file count and total size still
    match. Threads rather than processes: hashlib releases the GIL and the work
    is mostly file I/O, and this avoids spawning subprocesses inside a training
    start-up on Windows.
    """
    key = _record_key
    cache_file = _cache_path(step, split)
    validity = _validity(records)

    if use_cache and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if cached.get("validity") == validity:
                hashes = cached["hashes"]
                if all(key(record) in hashes for record in records):
                    if verbose:
                        print(f"  hashes read from {cache_file}", flush=True)
                    return hashes
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # a damaged cache is just a cache miss

    if verbose:
        print(f"  hashing {len(records):,} files (cached afterwards)...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        digests = list(pool.map(lambda record: content_hash(record.path), records))
    hashes = {key(record): digest for record, digest in zip(records, digests)}

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"validity": validity, "hashes": hashes}), encoding="utf-8"
    )
    return hashes


def label_conflicts(records, hashes) -> dict[str, list]:
    """``{sha256: [records]}`` for images that carry more than one label.

    The same bytes filed under two classes is a labelling defect, not a
    duplicate, and what it costs depends on which side of a split it lands on:

    * in **training** it is noise -- two contradictory gradients on one image;
    * in **evaluation** it is a guaranteed error, because whichever way the
      model answers, one of the two rows is scored wrong.

    Training only ever passes the training side to `unique_records`;
    ``python -m repro.dedup --split test`` runs the same check on the test side.
    """
    by_hash: dict[str, list] = defaultdict(list)
    for record in records:
        by_hash[hashes[_record_key(record)]].append(record)
    return {
        digest: group
        for digest, group in by_hash.items()
        if len({record.label for record in group}) > 1
    }


def unique_records(records, *, step: int, split: str, workers: int = 8,
                   use_cache: bool = True, verbose: bool = False):
    """``(kept records, stats)`` -- one representative per identical group.

    Grouping is by ``(label, content hash)``, so identical files filed under
    different labels are both kept. Within a group the representative is chosen
    deterministically: an original photograph beats an augmented copy, and ties
    break on filename. Selection therefore does not depend on scan order, on
    the filesystem, or on the number of workers.
    """
    hashes = hash_records(records, step=step, split=split, workers=workers,
                          use_cache=use_cache, verbose=verbose)

    groups: dict[tuple[int, str], list] = defaultdict(list)
    for record in records:
        groups[(record.label, hashes[_record_key(record)])].append(record)

    kept, dropped = [], []
    for group in groups.values():
        # Originals first, then lexicographic: a stable, explainable choice.
        group = sorted(group, key=lambda r: (r.augmented, r.path.name))
        kept.append(group[0])
        dropped.extend(group[1:])

    # Keep the caller's ordering so downstream behaviour is unchanged.
    order = {id(record): index for index, record in enumerate(records)}
    kept.sort(key=lambda record: order[id(record)])

    cross_class = label_conflicts(records, hashes)

    names = CLASS_NAMES[step]
    per_class = {}
    for label, name in enumerate(names):
        before = sum(1 for record in records if record.label == label)
        after = sum(1 for record in kept if record.label == label)
        per_class[name] = {"before": before, "after": after, "dropped": before - after}

    stats = {
        "step": step,
        "split": split,
        "before": len(records),
        "after": len(kept),
        "dropped": len(dropped),
        "per_class": per_class,
        "cross_class_groups": len(cross_class),
        "cross_class_examples": [
            sorted(_record_key(record) for record in group)
            for group in list(cross_class.values())[:5]
        ],
    }
    return kept, stats


def write_manifest(path: Path, kept, dropped_from, *, step: int, split: str,
                   workers: int = 8) -> None:
    """List every file of the split and whether the deduplicated run used it.

    This makes the deduplicated variant as auditable as a separate folder
    tree, without a second copy of the data.
    """
    hashes = hash_records(dropped_from, step=step, split=split, workers=workers)
    kept_ids = {id(record) for record in kept}
    names = CLASS_NAMES[step]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "class", "class_name", "augmented", "sha256", "kept"])
        for record in dropped_from:
            writer.writerow([
                _record_key(record),
                record.label,
                names[record.label],
                int(record.augmented),
                hashes[_record_key(record)],
                int(id(record) in kept_ids),
            ])


def print_stats(stats: dict) -> None:
    step, split = stats["step"], stats["split"]
    before, after = stats["before"], stats["after"]
    share = 100.0 * stats["dropped"] / before if before else 0.0
    print(f"step-{step} {split}: {before:,} -> {after:,} "
          f"({stats['dropped']:,} duplicates, {share:.1f}%)")
    for name, row in stats["per_class"].items():
        class_share = 100.0 * row["dropped"] / row["before"] if row["before"] else 0.0
        print(f"    {name:<16} {row['before']:6,} -> {row['after']:6,}"
              f"   dropped {row['dropped']:6,} ({class_share:5.1f}%)")
    if stats["cross_class_groups"]:
        print(f"    WARNING: {stats['cross_class_groups']} identical image(s) appear "
              "under more than one label; both copies are kept:")
        for example in stats["cross_class_examples"]:
            print(f"      {example}")


def main(argv=None) -> int:
    console.init()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", type=int, choices=STEPS, default=None,
                        help="default: both steps")
    parser.add_argument("--split", default="train", choices=("train", "test"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.add_argument("--write-manifest", action="store_true",
                        help="write results/dedup_step<n>_<split>.csv listing every "
                             "file and whether the deduplicated run keeps it")
    args = parser.parse_args(argv)

    from .data import list_images

    paths.check_data_root()
    for step in ([args.step] if args.step else list(STEPS)):
        records = list_images(step, args.split, include_augmented=True)
        kept, stats = unique_records(records, step=step, split=args.split,
                                     workers=args.workers, use_cache=args.use_cache,
                                     verbose=True)
        print_stats(stats)
        if args.write_manifest:
            manifest = paths.RESULTS_DIR / f"dedup_step{step}_{args.split}.csv"
            write_manifest(manifest, kept, records, step=step, split=args.split,
                           workers=args.workers)
            print(f"    manifest: {manifest}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
