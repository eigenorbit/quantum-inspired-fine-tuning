# ResNet152 control experiments for the BRONet adaptation study

## Experiment 1: `full` — only the last bottleneck and a new head train

In the paper (and in `trichoscopy/`) the whole ResNet152 is fine-tuned:
58.7 M trainable parameters, head `2048→256→ReLU→Dropout→K`. Here:

* ResNet152, torchvision `IMAGENET1K_V2` (ImageNet top-1 82.284 %), frozen up to
  and including `layer4.1`;
* trained: `layer4.2` (3 convolutions + the affine parameters of its 3 BN layers,
  4 462 592 parameters; BN running statistics stay ImageNet's, BN always in eval
  mode) and a new **LLN head as in BRONet** (`bronet_mlp.adapters.LLNHead`:
  one linear layer `2048→K` with unit-norm rows and a bias, 10 245 / 6 147
  parameters);
* ImageNet normalisation is inside the model; inputs are the same `[0, 1]`
  224×224 tensors BRONet sees (`experiments/common/data.py`, protocol `unique`).

Protocol = the tuned protocol of `experiments/bronet_mlp` (its code is imported,
not copied): validation split **identical to BRONet's** (`val_split.json`,
checked by sha256), lr tuned on it (grid 1e-5, 3e-5, 1e-4, 3e-4, 1e-3; epoch
ceiling 30), selections `best` and `stable`, then 3 seeds trained on the 90 %
fit part with early stopping on validation; both the best-validation and the
last epoch are evaluated on test. Adam, StepLR(2, 0.7), deterministic; batch 256
by default, the main protocol of the study uses **batch 32** (`--batch-size 32`,
folders `*_bs32`, see "Other batch sizes").
Loss CE(τ·z), τ = 4: BRONet's `cert_ce` without the certificate term (ResNet
has no useful Lipschitz bound).

The input of `layer4.2` (2048×7×7) is cached once per step as a memory-mapped
`.npy` (float16 by default: step-1 6.5 GB, step-2 2.8 GB; `--feature-dtype
float32` doubles that).

## Experiment 2: `butterfly`, `lora` and `head-only` on the same block

Same cache, protocol, head and loss; only the adaptation of `layer4.2` differs:

| mode | layer4.2 | trainable (step-1 / step-2) | spectrum |
|---|---|---|---|
| `head-only` | BN folded into the convolutions, frozen | 10 245 / 6 147 | = pretrained |
| `butterfly` | BN folded, W' = R(θ)·W per convolution, R = `core.ButterflyOrthogonal` (512, 512, 2048 channels), θ = 0 at start | 15 872 angles + head | = pretrained exactly (σ of every kernel matrix and the operator norm of every convolution) |
| `lora` | BN folded, W' = W + B·A per convolution, rank 4, B = 0 at start (A from the stream `seed + 7919`) | 40 960 + head | drifts |
| `full` | everything trains (experiment 1) | 4 472 837 / 4 468 739 | drifts |

R mixes output channels only: (RW)ᵀ(RW) = WᵀW. The folded biases stay frozen
(as in `bronet_mlp`). lr grids: head-only, butterfly and lora as in `bronet_mlp`
(3e-4…1e-2, 3e-4…3e-2, 1e-5…3e-3). `metrics_*.json → spectrum` has, per convolution,
σ_max, σ_min and the operator norm on the 7×7 input (1×1: σ_max; 3×3: the FFT
bound of `bronet_mlp.bronet.conv_fft_bound`) with their ratios to the pretrained
block, and `orthogonality_error_max` for butterfly.

```bat
python -m experiments.resnet_adapt.tune     --step 2 --batch-size 32          # every mode; finished grid points are skipped
python -m experiments.resnet_adapt.run_grid --step 2 --seeds 0 1 2 --from-tuning --batch-size 32
```

Butterfly costs ~55 ms per optimisation step on the RTX 4070 Laptop (building
three orthogonal matrices dominates), almost independent of the batch: ~550
img/s at batch 32, ~2 800 img/s at batch 256.

## Robustness: Lipschitz bound and attacks

```bat
python -m experiments.resnet_adapt.spectrum --step 2 --batch-size 32    # rigorous upper bound on L, seconds after the first run
python -m experiments.resnet_adapt.attacks  --step 2 --batch-size 32    # PGD-L2 + random noise at 36/72/108 /255, best epoch, seed 0
python -m experiments.resnet_adapt.attacks  --step 2 --batch-size 32 --eps 1 2 4 9 18   # smaller radii, own folder
python -m experiments.resnet_adapt.attacks  --step 2 --batch-size 32 --attack apgd-t   # stronger attacks: apgd-ce|apgd-dlr|apgd-t|square
python -m experiments.resnet_adapt.min_eps  --step 2 --batch-size 32    # eps*(x): smallest flipping perturbation per image
python -m experiments.resnet_adapt.attacks  --plots-only --batch-size 32
```

`spectrum.py` bounds L of the whole network from above (stem, every bottleneck
via the FFT bound of its convolutions, pooling factors; the frozen front is
cached in `backbone/lipschitz_front.json`) for the best checkpoint of every run:
`results/lipschitz_step<n>[_bs<B>].json/.md`.

`attacks.py` and `min_eps.py` run the shared attack engine
(`experiments.common.attacks`, the same as `bronet_mlp.attacks` / `min_eps`) with
the frozen `Front` (ImageNet normalisation inside the model) as the network in
front of the classifiers, so eps is in pixels in [0, 1] of the 224×224 input as
for BRONet, on the same test subset (300 / 240 images by default, seed 999).
Defaults: best checkpoint (`--checkpoint final`: the last epoch), seed 0, every
run, eps 36/72/108 /255 (`--eps` in units of 1/255: own folder `_eps…`),
`--attack-batch 16`. The attack flags (`--attack`, `--steps`, `--queries`, …)
and the min-eps flags are those of `bronet_mlp`. The certified radii use the
bound of `spectrum.py` (L ≤ 10^109): they are ~0, the sanity check is kept for
symmetry. `clean_agreement.json` compares the clean logits through the front with
those of the cached features.

Outputs: `results/attacks/step_<n>[_bs<B>]_<checkpoint>[_eps…][_<attack>]/`,
`results/attacks/step_<n>[_bs<B>]_<checkpoint>_mineps[_<attack>]/`, figures in
`results/figures/`, and per run `<run dir>/attacks/<attack>_l2_<checkpoint>.csv` /
`min_eps_<attack>_<checkpoint>.csv`. The per-run CSV has no eps tag: a run with
`--eps …` overwrites the per-run CSV of the default eps (the folders under
`results/attacks/` stay separate).

## Running (repository root, the venv's python; cmd or PowerShell)

Environment: `requirements.txt` in the repository root (see the top-level README).
Locations (data, outputs) come from the environment or `paths.toml` in the
repository root, as for `bronet_mlp` (see `paths.example.toml`).

Prerequisite: `tune` reuses BRONet's validation split
`<DERMA_REPRO_OUT>/bronet_mlp/tuning/step_<n>/val_split.json` and stops without it, so
`python -m experiments.bronet_mlp.tune --step <n>` must have run first. The McNemar
comparisons of `report` use the `repro` ResNet152 and the BRONet runs when they are
found under `DERMA_BASELINE_ROOTS`, and are left out otherwise. The ImageNet weights
are downloaded by torchvision into its cache (sha256 recorded in every run).

```bat
python -m experiments.resnet_adapt.features extract --step 2
python -m experiments.resnet_adapt.features verify  --step 2
python -m experiments.resnet_adapt.tune     --step 2 --batch-size 32
python -m experiments.resnet_adapt.run_grid --step 2 --seeds 0 1 2 --from-tuning --batch-size 32

python -m experiments.resnet_adapt.features extract --step 1
python -m experiments.resnet_adapt.features verify  --step 1
python -m experiments.resnet_adapt.tune     --step 1 --batch-size 32
python -m experiments.resnet_adapt.run_grid --step 1 --seeds 0 1 2 --from-tuning --batch-size 32

python -m pytest experiments/resnet_adapt/tests -m "not slow"
python -m pytest experiments/resnet_adapt/tests
```

Every stage is resumable (extraction row by row, training epoch by epoch);
finished runs are skipped, runs with other settings are refused (`--force` /
`--restart`). `report` is called by `run_grid`; alone:
`python -m experiments.resnet_adapt.report --step 2 --batch-size 32`. A single
run: `python -m experiments.resnet_adapt.train --step 2 --mode butterfly --seed 0
--from-tuning --batch-size 32 [--variant stable]`.

Measured on the RTX 4070 Laptop: front ~230 img/s (extraction ≈ 2.5 min step-1,
≈ 1 min step-2), training of `layer4.2` + head ~6 200 img/s at batch 256
(≈ 1 GB GPU memory), i.e. one 30-epoch run ≈ 3–4 min on step-1 and ≈ 1–2 min
on step-2 including the per-epoch validation.

## Other batch sizes

`--batch-size B` (≠ 256) gives a separate protocol with its own folders —
`tuning/step_<n>_bsB/`, `runs/step_<n>_bsB/`, `results/summary_step<n>_bsB.md` —
so it never touches the batch-256 results; the feature cache is shared. Pass the
same `--batch-size` to `tune`, `run_grid` and `report`. On `attacks`, `min_eps`
it is also called `--train-batch-size` (as in `bronet_mlp`); the attack batch
there is `--attack-batch`.

## Outputs, under `<DERMA_REPRO_OUT>/resnet_adapt/`

```
backbone/lipschitz_front.json   cached bound of the frozen front (by weights sha256)
features/step_<n>/<dtype>/      features.npy, index.json (meta, rows, file stats, done, passes), DONE, verify.json
features/step_<n>/<dtype>_<command>.log
tuning/step_<n>[_bs<B>]/        val_split.json (copy of BRONet's), <mode>/lr<lr>/seed0/, selection.json,
                                tuning_step<n>[_bs<B>].md, tune.log
runs/step_<n>[_bs<B>]/<mode>[_stable]/seed<s>/
    config.json, data_manifest.json, history.json (per epoch: loss, val/test acc, times)
    checkpoints/{last,best,final}.pt      trainable tensors only, weights_only=True
    metrics_{best,final}.json, metrics.json (= best), predictions_{best,final}.csv, outputs_{best,final}.npz
    attacks/                               per-run rows of the attack folders
results/
    summary_step<n>[_bs<B>][_last].md, mcnemar_step<n>[_bs<B>][_last].csv, runs_step<n>[_bs<B>].csv
    lipschitz_step<n>[_bs<B>].json/.md, lipschitz.log
    attacks/, figures/                     see "Robustness"
```

The names and suffix rules are shared with `bronet_mlp` (`experiments/common/layout.py`).

`metrics_*.json` also holds σ_max / σ_min of the three BN-folded convolutions of
`layer4.2` (as matrices `out × in·k²`) and their ratio to the pretrained block,
and the times: `seconds.train` (optimisation steps), `seconds.total` (plus
per-epoch validation/test and the final evaluations).

The summary compares, on the same test images (McNemar, Holm): these models,
the full fine-tuning of ResNet152 from the reproduction
(`models/step_<n>/resnet152_unique/predictions_final.csv`) and every BRONet run
(`bronet_mlp/runs/step_<n>/<mode>/seed<s>/predictions_best.csv`).

## Tests

36 tests (5 `slow`), on a random-weight ResNet152 (no download) and synthetic
data; only the validation-split test needs the dataset (skipped without it).

| test | what |
|---|---|
| front + classifier = torchvision forward | split point, pooling, fc logits |
| only layer4.2 + head train; exact parameter count | freezing |
| BN in eval, running statistics unchanged by training | BN handling |
| normalisation inside the model; gradient w.r.t. pixels | attack readiness |
| head depends only on the seed; block = pretrained at start | initialisation |
| spectrum ratio 1 at start, moves after a step | spectrum bookkeeping |
| BN-folded block = pretrained at start; trainable counts per mode | butterfly / lora / head-only |
| butterfly keeps σ and operator norms exactly; head-only frozen, full moves the spectrum; FFT bound ≥ true norm | spectra of the modes |
| fit deterministic, resumable, `load_run` round trip (butterfly, lora) | determinism of the new modes |
| same seed → same fingerprint; interrupted + resumed = straight | determinism |
| incompatible resume refused; `load_run` round trip | checkpoints |
| cache = end-to-end (float32, float16), resumable, bad meta / changed file refused | cache |
| validation split = BRONet's (sha256), sizes 13 707 / 240 | data (skips without data) |
| McNemar only on identical images, Holm ≥ raw p; batch sizes have separate folders | report, layout |
| conv bound = BRONet's for 3×3, an upper bound; maxpool factor 2 tight; stem folding; layer4.2 term kept by butterfly | spectrum.py |
| total bound ≥ the network's measured gain (`slow`) | spectrum.py |
| attack / min-eps folder names; PGD and strong attacks through the front (`slow`) | attacks |
