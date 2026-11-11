# Reproduction

Training code for Caposiena Caro, Orlova, Di Meo, Zalaudek, *"Analysis of
trichoscopic images using deep neural networks for the diagnosis and activity
assessment of alopecia areata"*, JDDG 2026;24:44–55
([10.1111/ddg.15847](https://doi.org/10.1111/ddg.15847)).

This package trains all six models of the paper — three architectures × two
steps — on the study's preprocessed data, using the hyperparameters of
supplementary Table S2, and compares the result with the published Tables 5
and 6.

---

## Quick start

### Environment

With conda:

```powershell
conda env create -f environment.yml
conda activate derma-repro
```

Or into an existing interpreter:

```powershell
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Either way, confirm CUDA actually came through before going further:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

`False` means the CPU build got installed and training would take weeks. The
`+cu121` suffix is what prevents that, and the failure is silent: pip counts a
plain `torch==2.4.1` requirement as satisfied by `2.4.1+cpu`, so re-installing
without the suffix is a no-op that looks like success. `pip install --force-reinstall --no-deps`
with the `+cu121` versions is the way out.

### Run

```powershell
# Locations: copy ..\paths.example.toml to ..\paths.toml (repository root) and set
# DERMA_DATA_ROOT; outputs go to DERMA_REPRO_OUT, default <DERMA_DATA_ROOT>\repro_out.
# Environment variables of the same names win over the file.
$env:PYTHONPATH = "$PWD\src"
python -m repro.where

# Verify the environment and the dataset
python -m repro.check

# Two truncated models end to end, ~4 min, writes nothing permanent
.\scripts\smoke_test.ps1

# Optional: prove runs are bit-for-bit reproducible here, ~5 min
python -m repro.selftest

# The real thing: six models, then the comparison report
.\scripts\train_all.ps1
```

Single model:

```powershell
python -m repro.train --step 2 --arch densenet169
python -m repro.report --step 2
```

Every run is resumable — re-running `train_all.ps1` picks each model up from
its last completed epoch.

---

## Layout

```
trichoscopy/
├── src/repro/
│   ├── config.py      hyperparameters, transcribed from Table S2
│   ├── published.py   Tables 5 and 6, the numbers we are compared against
│   ├── settings.py    locations: environment, else paths.toml (CLI: where.py)
│   ├── paths.py       where the data is, where outputs go
│   ├── data.py        splits, the -aug filter, transforms
│   ├── dedup.py       CLI: find byte-identical images, drop them from training
│   ├── models.py      the three backbones + the paper's 256-unit head
│   ├── engine.py      train / evaluate loops
│   ├── metrics.py     accuracy, per-class P/R/F1, ROC and PR areas
│   ├── train.py       CLI: train one (step, architecture)
│   ├── seeding.py     determinism: seeds, algorithms, RNG across a resume
│   ├── report.py      CLI: reproduced vs published tables and figures
│   ├── check.py       CLI: environment and dataset validation
│   └── selftest.py    CLI: proves runs are reproducible on this machine
├── scripts/           train_all.{ps1,sh}, setup_vm.sh, smoke_test.ps1
├── environment.yml    conda environment
├── requirements.txt   the same pins, for pip
├── models/, results/  empty placeholders: outputs go to DERMA_REPRO_OUT
└── data/              no images; see data/README.md
```

---

## What follows the paper

Everything in supplementary Table S2 and the supplement's Methods text:

| Setting | Value | Source |
| --- | --- | --- |
| Optimizer | Adam | Table S2 |
| Learning rate | 0.001 | Table S2 |
| Weight decay | 0, except Step-2/DenseNet169 = 0.001 | Table S2 |
| LR schedule | StepLR, step size 2, γ = 0.7 | Table S2 |
| Batch size | 32 | Table S2 |
| Epochs | 30 | supplement |
| Loss | cross-entropy | supplement |
| Input | 224×224, ImageNet normalisation | supplement |
| Backbones | ResNet152, DenseNet169, EfficientNetB0, ImageNet-pretrained | supplement |
| Head | `Linear(→256) → ReLU → Dropout → Linear(→n_classes)` | supplement |
| Frozen layers | none — "all layers were unfrozen" | supplement |
| Evaluation set | test split **excluding augmented images** | Results, p. 48 |

Two settings are not stated in the paper; this code uses the head dropout
rates DenseNet169 0.4, EfficientNetB0 0.2, ResNet152 0.5, and ResNet152's
`IMAGENET1K_V2` starting weights.

---

## Known deviations

Read this before interpreting any delta.

**1. Step-1 is not directly comparable; Step-2 is.**

The Step-1 test split holds 717 original images, the published one 641. They
agree on AA (241), FFA-LPP (121), DF (60) and AGA (14) and differ only in
CHP/HV: 281 here against 205 in the paper, that is 20% of the class against
14.6%. Every other class is held out at 20% in both. A Step-1 delta therefore
mixes a training difference with a different denominator on the largest class.

Step-2's test split is the published one — 78 / 48 / 114 = 240 — so those
deltas compare like with like.

**2. Model selection has no validation split.**

The data is split into train and test only. Two checkpoints are saved:

* `final.pth` — weights after epoch 30. **This is the reported number.**
* `best.pth` — the epoch with the highest test accuracy. Selected on the test
  set, so optimistically biased.

`python -m repro.report --which best` reports the latter.

**3. Training applies no random augmentation.**

The training folders are already augmented and class-balanced. The flips,
rotations and PCA colour jitter the supplement describes were applied offline
when those files were written, so adding online augmentation would stack a
second round on top of the first. Only the deterministic resize and
normalisation are applied at load time. The augmented folders match the
paper's description and its image totals (49,219 for Step-1, 23,885 for
Step-2).

**4. Run-to-run variance is not measured.**

One seed per configuration. A single fine-tuning run of a 58M-parameter network
at lr = 1e-3 is not a point estimate. Use `--seed` to repeat a configuration if
a delta matters.

**5. AGA will fail, as it did in the paper.**

Images labelled `aga_ina` and `aga_hl` belong to CHP, so only 67 of 234 AGA
images reach the AGA class, of which 14 are in the test split. Half the
"balanced" augmented AGA training class consists of byte-identical copies. The
paper reports F1 of 23.5 / 12.5 / 0.0 for this class; expect it to reproduce as
noise around zero.

**6. The two steps do not share a split.**

388 of 1,196 shared source images land in different splits across Step-1 and
Step-2, so the framework cannot be evaluated end to end on these folders —
80.8% of Step-1's AA test images are in Step-2's training set. The paper does
not run the chain end to end, so its numbers are unaffected; this code does not
either.

---

## Experiment: training without duplicate images

The augmentation space is finite -- eight orientations, and the PCA colour
jitter fires only some of the time -- so at the multipliers needed to balance
the small classes the generator began emitting files that are identical byte
for byte:

| Split | Files | Unique | Duplicates |
| --- | --- | --- | --- |
| step-1 train | 39,368 | 31,665 | 7,703 (19.6%) |
| step-2 train | 19,076 | 13,707 | 5,369 (28.1%) |
| step-1 test (originals) | 717 | 717 | none |
| step-2 test (originals) | 240 | 240 | none |

It falls hardest where it hurts most. AGA, the smallest class, is 54.6%
duplicates (7,873 -> 3,572, up to 22 identical copies of a single frame);
CHP, the largest, is 2.5%. Duplicates multiply a class's weight in the loss
without adding information.

Because the test splits hold original photographs only, and every one of those
is unique, a deduplicated run is measured on exactly the same images as the
baseline. The comparison is clean.

### Running it

```powershell
python -m repro.dedup --write-manifest          # what would be dropped, and why
python -m repro.train --step 1 --arch efficientnet_b0 --unique
python -m repro.report --variant unique
.\scripts\train_all.ps1 -Unique                 # all six
```

`--unique` writes to `models/step_<n>/<arch>_unique/`, so baseline runs are
never overwritten, and `results/variants_vs_full.md` appears automatically once
both variants exist for a model -- overall and per-class, with deltas.

Selection happens in memory when the loaders are built; no second copy of the
dataset is written. `--write-manifest` produces
`results/dedup_step<n>_train.csv` listing every file, its hash and whether the
deduplicated run keeps it, so the experiment is as auditable as a separate
folder tree would have been. Hashes are cached under `<DERMA_REPRO_OUT>/cache/`, so only the
first run pays the ~30 s scan.

Deduplication does not change reproducibility: two `--unique` runs with the
same seed still produce byte-identical weights.

### Two things to expect

**The classes stop being balanced.** That balance was manufactured by
duplication, so removing it undoes it:

| | step-1 | | step-2 | |
| --- | --- | --- | --- | --- |
| | full | unique | full | unique |
| class sizes | ~7,875 each | 7,049 / 6,422 / 6,943 / **3,572** / 7,679 | ~6,359 each | 4,819 / 3,540 / 5,348 |

AGA ends up roughly half the size of every other step-1 class. If that turns
out to dominate the result, class weighting in the loss is the natural next
step -- it is not implemented here, because it would confound this experiment.

**One image carries contradictory labels.** In step-2 train, one photograph
is filed, byte for byte, under both "AA active" and "AA inactive". Both copies
are in train, where they are only noise — two contradictory gradients on one
image. Deduplication is done per class, so both survive: dropping one would be
a labelling decision, not a deduplication. `repro.dedup` reports it, and
`metrics.json` records it as `dedup.cross_class_groups`.

---

## Reproducibility

Runs are deterministic by default. The same command with the same seed produces
byte-identical weights and identical metrics, and so does a run that was
interrupted and resumed, or run with a different `--num-workers`.

Four things are pinned to get there, and each one mattered:

1. **Seeds** for `random`, numpy and torch (CPU and CUDA).
2. **Algorithm selection.** cuDNN benchmarking is off, deterministic kernels are
   requested, `torch.use_deterministic_algorithms(True)` makes any remaining
   nondeterministic op raise rather than vary silently, and
   `CUBLAS_WORKSPACE_CONFIG` is set in `repro/__init__.py` — before anything can
   import torch, which is when cuBLAS reads it. All three architectures run
   clean under this.
3. **RNG state across a resume.** Generator state travels in `last_full.pth`, so
   a run resumed at epoch 20 continues the dropout stream instead of restarting
   it.
4. **Shuffling order.** `DataLoader(shuffle=True)` is *not* used. It draws the
   permutation from a generator the loader also dips into for the worker base
   seed — but only on the first iteration inside a process. With
   `persistent_workers`, that makes epoch 2 of a straight run and epoch 2 of a
   resumed run shuffle differently, which was enough to make resumed runs
   diverge. `data.EpochSeededSampler` derives each permutation from
   `(seed, epoch)` alone instead.

Verify it on your own machine:

```powershell
python -m repro.selftest
```

It trains the same small configuration four ways — straight, straight again,
one-epoch-at-a-time with `--resume`, and with a different worker count — in
separate processes, and checks that the checkpoint bytes, the metrics and the
per-image predictions all match. About five minutes.

`metrics.json` records a short hash of the weights (`fingerprint`), so two runs
can be compared without diffing checkpoints; `results/summary.md` prints it.

What is *not* fixed: wall-clock timings in `history.json` differ between runs,
which is inherent. Determinism costs roughly 5–10% throughput; `--no-deterministic`
trades it back. Results are reproducible on the same machine and library
versions — a different GPU or a different torch build can still differ in the
last bits.

---

## Runtime

Measured on an RTX 4070 Laptop (8 GB), batch 32, full precision. The paper used
an RTX 3060 6 GB.

| Step | Model | img/s | 30 epochs |
| --- | --- | --- | --- |
| 2 | EfficientNetB0 | 335 | ~30 min |
| 2 | DenseNet169 | 140 | ~70 min |
| 2 | ResNet152 | 86 | ~110 min |
| 1 | EfficientNetB0 | 335 | ~60 min |
| 1 | DenseNet169 | 140 | ~140 min |
| 1 | ResNet152 | 86 | ~230 min |
| | | | **~10.5 h total** |

Data loading is not the bottleneck (740–1,180 img/s at 4–12 workers).

Peak GPU memory, batch 32: EfficientNetB0 2.7 GiB, DenseNet169 4.9 GiB,
ResNet152 6.0 GiB. ResNet152 fits in 8 GB but leaves little headroom — close
other GPU applications, or pass `--amp`, which halves both memory (3.5 GiB) and
runtime at the cost of float16 arithmetic. Mixed precision is off by default
because the paper does not mention it.

---

## Outputs

Everything below is written under `DERMA_REPRO_OUT` (default
`<DERMA_DATA_ROOT>/repro_out`), never into the repository. Per run, in
`models/step_<n>/<architecture>/`:

| File | Contents |
| --- | --- |
| `final.pth` | state_dict after the last epoch — the reported model |
| `best.pth` | state_dict of the best test-accuracy epoch |
| `last_full.pth` | optimiser, scheduler and RNG state, for `--resume` |
| `config.json` | the resolved configuration actually used |
| `history.json` | per epoch: loss, accuracy, timing, and the full metric table |
| `metrics.json` | run summary: both checkpoints, with weight fingerprints |
| `metrics_final.json`, `metrics_best.json` | the metrics of one checkpoint each |
| `predictions_final.csv`, `predictions_best.csv` | per-image true class, predicted class, and every class probability |

Every one of those is rewritten at the end of each epoch, not just when training
finishes. A run stopped at epoch 29 still leaves metrics that describe the
checkpoints on disk, and `repro.report` works against a run still in progress.

The per-image CSVs mean later analysis — per-patient aggregation, error review,
a different averaging scheme — needs no second inference pass.

From `python -m repro.report`, in `results/`:

| File | Contents |
| --- | --- |
| `summary.md` | accuracy and weighted F1 for all six runs, vs published |
| `step<n>_comparison.{md,csv}` | every cell of Table 5 / Table 6, with deltas |
| `figures/step<n>_<arch>_confusion.png` | confusion matrix |
| `figures/step<n>_<arch>_{auc,auprc}.png` | per-class ROC / PR areas |

---

## Sanity check on the numbers

A single epoch of Step-2 / EfficientNetB0 already lands at accuracy 80.83,
weighted F1 80.82, against the published 80.83 / 80.75 — with per-class
precision and recall trading places against the paper's (the paper's
EfficientNetB0 is precision-heavy on active AA, this run is recall-heavy) while
the F1 scores stay within about one point.

That is a useful calibration for what "reproduced" will mean here. The task is
easy enough, and the augmented training set large enough, that the headline
metrics are reached almost immediately and are insensitive to the remaining 29
epochs. Agreement on overall accuracy is therefore weak evidence; the per-class
breakdown in `step<n>_comparison.md` is where a real difference would show.
