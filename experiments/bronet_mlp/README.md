# Certified adaptation of BRONet to trichoscopy through its dense MLP

Does §6 of `experiments/robustness/analysis.ipynb` hold on real medical data
and a real certified backbone? The claim: adapting a pretrained layer as
**R·W with R orthogonal** (a butterfly of Givens rotations) keeps every singular
value of W, so the Lipschitz bound — and with it the certificate — is unchanged,
while LoRA and full fine-tuning break it.

* Backbone: **BRONet** ImageNet (`imagenet-14x588_399.pth`, Table 1 of the BRONet
  paper), L₀ ≈ 8.10 (official-style) / 8.40 (rigorous). Everything up to the neck
  is frozen and its 2048-d output φ(x) is **cached once**.
* Adapted: only the 8 dense 2048×2048 layers of `model.linear` + a new LLN head,
  in four modes — `head-only`, `butterfly`, `lora` (rank 4), `full`.
* Data: protocol of `trichoscopy/`, variant **`unique`** (delivered
  train/test, byte-identical copies removed from train within each class; test =
  originals: step-1 717, step-2 240). No cross-validation.

Since BRONet's L is not 1, the claim is "L stays L₀" — never "L = 1".

Code shared with `experiments/resnet_adapt` lives in `experiments/common`: output
names (`layout`), locations, I/O, the dataset records, the certificate maths,
the losses, McNemar/Holm, and the attack engine (`common/attacks`: PGD, APGD,
Square, the eps* search). The modules `data`, `certify`, `losses`,
`strong_attacks`, ... of this package re-export those names.

## Setup (any machine)

```bash
# 1. torch (CUDA build for the machine), then BRONet + checkpoint + packages
experiments/bronet_mlp/scripts/setup_bronet.sh          # Linux
.\experiments\bronet_mlp\scripts\setup_bronet.ps1       # Windows
```

The repository is the only code to copy; plus the data, BRONet (by the setup
script) and the locations below. Each is an environment variable; one that is
not set is read from `paths.toml` in the repository root (copy
`paths.example.toml`; not in git), else it takes the default. The rules live in
`trichoscopy/src/repro/settings.py`; `python -m repro.where` prints the result.

| variable | meaning | default |
|---|---|---|
| `DERMA_DATA_ROOT` | `model_data_step{1,2}_anon/{train,test}/<label>/*.jpg` | none: required |
| `DERMA_REPRO_OUT` | root of all outputs; this package writes only to `bronet_mlp/` below it | `<DATA_ROOT>/repro_out` |
| `BRONET_ROOT` | clone of ntuaislab/BRONet, commit 9d59c9b | `<DATA_ROOT>/pretrained/BRONet` |
| `BRONET_CKPT` | folder with `imagenet-14x588_399.pth` + `config.yaml` | `<DATA_ROOT>/pretrained/bronet_imagenet` |
| `DERMA_BASELINE_ROOTS` | (report only) where `repro` runs live; a list in `paths.toml`, `os.pathsep`-separated in the environment | `<DERMA_REPRO_OUT>` |
| `DERMA_COLOR_REFERENCE` | (`lossy --renorm authors\|ref` only) the dataset's colour-normalisation reference photograph | none |

Nothing large is ever written into the repository tree.

## Running

All commands from the repository root, with the venv's python.

```bash
# everything for one step, resumable at every stage:
# L_pre -> feature cache -> check -> lr tuning -> runs with the tuned settings -> report
experiments/bronet_mlp/scripts/run_step.sh --step 2 --seeds "0 1 2" --cache-bro-weights
.\experiments\bronet_mlp\scripts\run_step.ps1 -Step 2 -Seeds 0,1,2 -CacheBroWeights   # -TrainBatchSize 32

# or stage by stage
python -m experiments.bronet_mlp.bronet   lpre     [--cache-bro-weights]   # once, ~15 min (CPU float64)
python -m experiments.bronet_mlp.features extract --step 2 [--batch-size 64] [--cache-bro-weights]
python -m experiments.bronet_mlp.features verify  --step 2                 # cache vs BRONet forward
python -m experiments.bronet_mlp.tune     --step 2                         # lr / epoch grid on validation
python -m experiments.bronet_mlp.run_grid --step 2 --seeds 0 1 2 --from-tuning   # skips finished runs; then report
python -m experiments.bronet_mlp.report   --step 2                         # also run by run_grid
python -m experiments.bronet_mlp.train    --step 2 --mode butterfly --seed 0 [--from-tuning [--variant stable]]
python -m experiments.bronet_mlp.evaluate --run-dir <DERMA_REPRO_OUT>/bronet_mlp/runs/step_2/butterfly/seed0

python -m pytest experiments/bronet_mlp/tests                 # all available tests
python -m pytest experiments/bronet_mlp/tests -m "not slow"   # skip full-model loads
```

Without `--from-tuning`, `train` and `run_grid` use the fixed recipe (lr 1e-3,
30 epochs on the whole train, no validation, headline = last epoch); the results
of the study come from the tuned protocol below.

### Tuning: learning rate and number of epochs (validation split of train, never test)

```bash
python -m experiments.bronet_mlp.tune     --step 2                                  # grid, then selection
python -m experiments.bronet_mlp.run_grid --step 2 --seeds 0 1 2 --from-tuning      # --force retrains
```

* Validation = 10 % of the deduplicated train, stratified by class, grouped by
  photograph (`P<patient>_<image id>`: all augmented copies — and a photo filed
  under two labels — stay on one side). Saved to `tuning/step_<n>/val_split.json`
  and re-checked on every call.
* Tuned **per mode**: the **lr** on its own grid — head-only {3e-4, 1e-3, 3e-3, 1e-2},
  butterfly {3e-4 … 3e-2}, lora {1e-5 … 3e-3}, full {1e-5, 3e-5, 1e-4,
  3e-4} — and the **number of epochs** (early stopping decided on validation):
  each grid point trains up to the ceiling `--epochs` (30) and every epoch's
  validation accuracy is a candidate. No extra training for the epoch choice.
* Shared and fixed: τ, ε_train, batch, schedule, loss, the epoch ceiling (they
  define the accuracy/radius trade-off itself).
* Criterion: highest **validation clean accuracy** over all (lr, epoch); ties:
  weighted F1, then the earlier epoch, then the smaller lr. Epochs at the
  majority-class rate (collapsed) only if everything collapsed.
  `tuning_step<n>.md`: best epoch per lr with its L_mlp and certified accuracy,
  plus the last-epoch accuracy of the same run; warnings when the choice sits at
  a grid edge or at the epoch ceiling.
* Two selections per mode. **best**: the criterion above. **stable**: the same
  criterion restricted to grid points whose run did not diverge (at its last
  epoch not collapsed and within 2 points of its own best validation accuracy).
  For head-only and butterfly they coincide; for lora and full both are trained,
  the stable one into `runs/step_<n>/<mode>_stable/`, and the report shows both
  columns (`lora`, `lora (stable)`). `run_grid --variants best stable` (default);
  `train --from-tuning --variant stable` for a single run.
* **Final protocol** (`--from-tuning`, alias `--lr-from-tuning`): the lr comes
  from the tuning; every seed trains on the fit part (90 % of train) up to the
  epoch ceiling (30), is scored on the validation part after every epoch and
  keeps its own best-validation epoch as `checkpoints/best.pt` (same tie rules;
  collapsed epochs only if all collapsed). Both `best.pt` and `final.pt` (last
  epoch) are evaluated on test: `metrics_best.json` / `metrics_final.json`,
  `predictions_{best,final}.csv`, `certificates_{best,final}.npz`;
  `metrics.json` is the headline = best. Test never influences the kept epoch.
  The report writes `summary_step<n>.md` (best-validation epoch) and
  `summary_step<n>_last.md` (last epoch). Note for the write-up: the final
  models see 90 % of train, the paper's baselines 100 %.
* A selection made with other fixed settings is refused; finished runs with
  other settings are never skipped silently (`--force` / `--restart`). Tuning
  a `--modes` subset keeps the other modes' selection.
* Cost on the RTX 4070 Laptop, step-2: ~55 min for the whole grid (butterfly
  ~6.5 min per point, the rest ≤ 1 min); finished points are reused.
* Caveat for the write-up: the validation optimum is taken over many (lr, epoch)
  pairs on 1.4k images, so the validation number is optimistic; the test set is
  untouched by the choice.

### Other training batch sizes

`--batch-size B` (≠ 256) on `tune`, `run_grid`, `train`, `report` and `plots`
(`--train-batch-size B` on `attacks` and `min_eps`, whose attack batch is
`--attack-batch`, alias `--batch-size`) gives
a separate protocol in its own folders — `tuning/step_<n>_bsB/`,
`runs/step_<n>_bsB/`, `results/summary_step<n>_bsB.md`,
`results/attacks/step_<n>_bsB_<checkpoint>/` — so the batch-256 results are never
touched. The feature cache, L_pre and the validation split (`tuning/step_<n>/val_split.json`)
are shared.

```bash
python -m experiments.bronet_mlp.tune     --step 2 --batch-size 32
python -m experiments.bronet_mlp.run_grid --step 2 --seeds 0 1 2 --from-tuning --batch-size 32
```

### Stage 2: end-to-end attacks (PGD-L2, stronger attacks, random noise)

```bash
python -m experiments.bronet_mlp.attacks --step 2 --n-images 240 --cache-bro-weights
python -m experiments.bronet_mlp.attacks --step 1 --n-images 300 --cache-bro-weights
python -m experiments.bronet_mlp.attacks --step 2 --n-images 240 --cache-bro-weights --attack apgd-t
python -m experiments.bronet_mlp.min_eps --step 2 --n-images 240 --cache-bro-weights   # eps*(x) per image
python -m experiments.bronet_mlp.plots   [--step 2] [--attack apgd-t]                 # figures of the attack folders
```

Every finished run of the tuned protocol (seed 0, headline checkpoint, 200 images
by default; `--runs`, `--seeds`, `--checkpoint headline|best|final`, `--n-images`
to change), eps 36/72/108 /255, the same random test subset (seed 999) for every
model; `--train-batch-size B` attacks the runs of another training batch. Phase A
computes the attack directions in batches of `--attack-batch` (alias
`--batch-size`; `--grad-precision fp16` = float16 BRO copies resident on the GPU,
the default on an 8 GB card; `fp32` = exact, the default when the matrices fit).
Phase B re-evaluates every attack result and every random perturbation EXACTLY
in float32, block-major (`--eval-chunk` images per pass). The mandatory sanity
check fails the command (exit code 2) if any successful attack lies inside the
certified radius (official or rigorous L). Resumable (`progress.jsonl`); other
settings in the same folder are refused, `--restart` discards.

* `--attack`: `pgd` (default; 20 steps, alpha = 2.5 eps / steps), or AutoAttack's
  `apgd-ce`, `apgd-dlr`, `apgd-t` (100 iterations; `--target-classes k`) and
  `square` (gradient-free, `--queries 5000`, `--p-init 0.8`), same threat model
  and the same two phases. The row fields `pgd_success` / `pgd_robust` hold the
  result of the folder's attack, whatever it is.
* `min_eps`: per clean-correct image the smallest perturbation the attack finds
  that flips it (check at 0.99 r(x), doubling from r(x), `--bisect-steps 8`;
  censored above `--eps-max 1020`/255): r(x) ≤ true distance ≤ eps*(x), and the
  tightness eps*/r. Same flags as `attacks` for the attack of every probe.

Outputs under `results/attacks/`: `step_<n>[_bs<B>]_<checkpoint>[_<attack>]/`
(`attack_config.json`, `progress.jsonl`, `sanity_check.json`,
`attacks_step<n>.md/.csv`) and `step_<n>[_bs<B>]_<checkpoint>_mineps[_<attack>]/`
(`min_eps_config.json`, …, `min_eps_step<n>.md/.csv`); per run
`<run dir>/attacks/<attack>_l2_<which>.csv` and `min_eps_<attack>_<which>.csv`
(`-` → `_`, e.g. `pgd_l2_best.csv`, `apgd_t_l2_best.csv`); figures
`results/figures/attacks_step<n>[_bs<B>]_<checkpoint>[_<attack>].png` and
`attacks_both_steps…png`.

### Stage 3: lossy image handling (JPEG, requantisation, resize, noise) and the eps_train sweep

```bash
python -m experiments.bronet_mlp.lossy measure --step 2 --cache-bro-weights --checkpoint final
python -m experiments.bronet_mlp.lossy sweep   --step 2 --eps 72 108 144 195 --cache-bro-weights
python -m experiments.bronet_mlp.lossy report  --step 2 --eps 36 72 108 144 195 --checkpoint final
```

* `measure`: every test image is changed by each operation (`--ops`, default all:
  `jpeg100 jpeg95 jpeg90 jpeg80 bits7 bits6 resize86 resize71 resize50 noise0.002
  noise0.005`), on the delivered 600×600 file (`--at native`, default) or on the
  224×224 input (`--at input`). φ of every changed image is cached once
  (`lossy_features/`); then, for every finished protocol run, an image counts as
  *covered* when it is correct and d(x) = ‖x_T − x‖₂ < r(x). Noise uses a fixed seed.
* `sweep`: retrains the protocol (`run_grid --from-tuning`, lr tuned at 36/255)
  with a larger `eps_train` (in units of 1/255; 36 = the protocol runs themselves)
  into `eps_sweep/step_<n>/eps<k>/`; `report` tabulates accuracy, median radius,
  certified accuracy and coverage per mode and eps.
* `--checkpoint best|final` (default `best`); the sweep is read at `final`.
* `--eps-warmup F --eps-start E` (on `train`, `tune`, `run_grid` and `lossy sweep`;
  units in `--help`) ramps the loss eps linearly from E to `eps_train` over the first fraction F of the
  epochs; only epochs after the ramp can be the best-validation epoch. `tune` with
  a loss other than the protocol's writes to `tuning/step_<n>_eps<k>_wu<F>/` and
  never over the protocol's selection; `--lrs` replaces the per-mode lr grid.
* `--renorm authors|ref|oracle` (native only) colour-normalises the changed file
  again before the network. `authors` and `ref` need the dataset's reference
  photograph, set by `DERMA_COLOR_REFERENCE` (no default); the normalisation is
  the dataset's own.

Outputs: `results/lossy/step_<n>_<checkpoint>_<at>[_renorm-<mode>]/` with `coverage.md/.csv`
and `coverage_images.csv` (`measure`) and `eps_sweep[_wu<F>].md/.csv` (`report`).

### Rotations inside the last BRO blocks (`conv_adapt`, `conv_attacks`, `conv_lossy`)

An extension that also adapts the last `--blocks` convolutional BRO blocks: their
588 output channels are rotated by an exactly orthogonal `PartitionedButterfly`
(588 = 512 + 64 + 8 + 4, `partitioned.py`), so L_pre is unchanged; the MLP is adapted
as in the protocol (`--mlp-modes`).

```bash
python -m experiments.bronet_mlp.conv_adapt   extract --step 2 --blocks 1 --cache-bro-weights
python -m experiments.bronet_mlp.conv_adapt   tune    --step 2 --blocks 1 --mlp-modes butterfly lora --cache-bro-weights
python -m experiments.bronet_mlp.conv_adapt   train   --step 2 --blocks 1 --mlp-modes butterfly lora --from-tuning --seeds 0 1 2 --cache-bro-weights
python -m experiments.bronet_mlp.conv_adapt   summary --step 2 --blocks 1
python -m experiments.bronet_mlp.conv_attacks attack  --step 2 --blocks 1 --cache-bro-weights
python -m experiments.bronet_mlp.conv_attacks mineps  --step 2 --blocks 1 --cache-bro-weights
python -m experiments.bronet_mlp.conv_lossy           --step 2 --blocks 1 --cache-bro-weights
```

`extract` caches the input of the first adapted block (float16, step-2 ~13–15 GB,
step-1 ~29–34 GB); `train` accumulates a batch of 256 over `--micro-batch` pieces
(64 fits 8 GB). Defaults of `--checkpoint`: `final` for `conv_attacks` and
`conv_lossy`; `summary` writes both. Outputs: `conv_features/`, `conv_adapt/`,
`results/conv_adapt/`.

### Hardware and memory

The code picks the device (`--device auto|cuda|cpu`) and where BRONet's ~7.7 GB
of BRO frequency matrices live (`--bro-weights auto|gpu|stream|cpu`, recorded in
the feature index and logs):

* `gpu` — ≥ 12 GB free on the GPU: all matrices on the GPU;
* `stream` — smaller GPU (e.g. 8 GB): matrices on the CPU, one block copied to
  the GPU while it runs;
* `cpu` — no GPU; slow, fine for smoke tests.

`--cache-bro-weights` computes the matrices once, one block at a time, into
`backbone/eval_weights/` (7.7 GB on disk) and memory-maps them afterwards: peak
RAM drops from ~9 GB to ~1.5 GB and later loads take seconds. Recommended on
machines with ≤ 16 GB RAM. The matrices are computed with the same CPU
arithmetic as `OrthoConvV2.train(False)`.

Precise singular values (for L) are always float64 **on the CPU**; the GPU is
used in float32 only. Every stage logs its wall time.

## Method, in one screen

```
f(x) = head( MLP_adapted( φ(x) ) ),  φ = neck ∘ conv ∘ conv_1 ∘ stem (x − 0.5)   [cached]
MLP_adapted: h ← MinMax(W_i' h + b_i'),  i = 1..8,   W_i = linear.eval_weight[i]
head: LLN (unit-norm rows) 2048 → K, new, always trained
```

| mode | W_i' | trained | L_mlp |
|---|---|---|---|
| `head-only` | W_i | head | = L_mlp⁰ |
| `butterfly` | R(θ_i)·W_i, θ = 0 at start (`core.ButterflyOrthogonal(2048)`) | θ (8 × 11 264), head | = L_mlp⁰ by construction |
| `lora` | W_i + B_i A_i, B = 0 at start, rank 4 | A, B, head | drifts |
| `full` | W_i, b_i free | W, b, head | drifts |

All four modes compute exactly the pretrained MLP at initialisation (tested).

**Loss** (same for all modes): certificate-aware cross-entropy in the style of
BRONet's `margin_loss`: `z_j += L_train·ε_train·‖w_y − w_j‖` for j ≠ y, then
`CE(τ·z, y)`, with L_train the current L_total (detached; recomputed each epoch
by power iteration for lora/full, the constant L₀ for head-only/butterfly).
Defaults ε_train = 36/255, τ = 4. BRONet's `logit_annealing` loss is available
with `--loss logit_annealing`.

**Training**: cached features, batch 256 (`--batch-size`), Adam, no weight
decay, StepLR(2, 0.7), epoch ceiling 30; lr and the kept epoch from the tuning
(`--from-tuning`; headline = best-validation epoch, the last epoch is reported
too), else lr 1e-3 and headline = last epoch. Deterministic (`repro.seeding`,
TF32 off), shuffling by `randperm` with generator `seed·1 000 003 + epoch`; the
LoRA/butterfly adapters are initialised from the stream `seed + 7919`.

**Certificate** (BRONet `get_predictions`): `r(x) = min_{j≠p} (z_p − z_j) /
(L·‖w_p − w_j‖)`, certified at ε iff correct and r > ε. Two constants:
**official-style** `L_pre(BRONet lipschitz(), 500 it) × L_mlp(exact SVD)` and
**rigorous** `L_pre(stem SVD × conv_1 FFT bound × ∏σ_max BRO × neck SVD) ×
L_mlp`. Radii are in the 224×224 input (pixels in [0,1]); ÷ 0.37458 for the
600×600 files.

## Outputs

Everything under `<DERMA_REPRO_OUT>/bronet_mlp/`:

```
backbone/
  L_pre.json            L_pre official + rigorous, per part; checkpoint sha256, code version
  bronet_audit.json     copy of reference/lipschitz.json (the full audit of the checkpoint)
  mlp_base.pt           W_i, b_i and the ImageNet head, derived from the checkpoint (tagged by sha256)
  L_mlp0.json           exact spectrum of the pretrained MLP
  ckpt_sha256.json      cached checkpoint hash (by size + mtime)
  eval_weights/         only with --cache-bro-weights: BRO matrices per block + meta.json (sha256)
features/step_<n>/
  shard_XXXX.pt         φ(x) in shards (resumable extraction)
  index.json            relative path -> (shard, row, size, mtime); metadata; extraction passes + timings
  DONE                  marker: the last extraction pass completed
  verify.json           result of `features verify`
tuning/step_<n>[_bs<B>]/
  val_split.json        the validation split (step_<n>/ only: shared by every batch size and by resnet_adapt)
  <mode>/lr<lr>/seed0/  one grid point (a run folder as below)
  selection.json        lr and epochs per mode, variants best / stable
  tuning_step<n>[_bs<B>].md, tune.log
runs/step_<n>[_bs<B>]/<mode>[_stable]/seed<s>/
  config.json           full config, provenance (hashes), env, device, versions, git branch/HEAD/dirty + hash
  data_manifest.json    train/test relative paths + labels, dedup stats, sha256 of both lists
  checkpoints/last.pt   resume point, rewritten atomically every epoch
  checkpoints/best.pt   best validation epoch (tuned protocol)
  checkpoints/final.pt  after the last epoch
  checkpoints/epoch_XXX.pt   only with --keep-every K
  history.json          per epoch: loss, running train acc, val/test acc/F1, L_train, L_mlp, cert. acc, times
  metrics_{best,final}.json  per checkpoint: quality (repro.metrics.full_report), L official + rigorous,
                        L/L₀, σ_max/σ_min per layer, radii, certified accuracy + curve, fingerprint, times
  metrics.json          the headline checkpoint (best with early stopping, else final) + the other one
  predictions_{best,final}.csv   as in repro + margin, radius, radius_rigorous, logits
  certificates_{best,final}.npz  per image: logits, pred, margin, radii (both L, and in 600 space), head
  attacks/              per-run rows of the attack folders (see Stage 2)
  train.log
results/
  summary_step<n>[_bs<B>][_last].md     best-validation epoch / last epoch
  mcnemar_step<n>[_bs<B>][_last].csv, mcnemar_cert_step<n>[_bs<B>][_last].csv
  adaptation_step<n>[_bs<B>].csv        one row per run and checkpoint
  figures/certified_accuracy_step<n>[_bs<B>]_<official|rigorous>_<best|last>.png, attack figures
  attacks/                              see Stage 2
```

The names and suffix rules are in `experiments/common/layout.py` (pinned by
`experiments/common/tests`). Checkpoints hold only the trainable tensors
(θ / A,B / W,b and the head) plus optimizer, scheduler, RNG state, config and the
sha256 of the BRONet checkpoint, the features and the data manifest; they load
with `weights_only=True`. A resume with any of those changed is refused
(`--restart` discards it); runtime flags (`--device`, `--keep-every`) may change.
`train.load_run(run_dir, which=...)` rebuilds the classifier (φ features in) from
one checkpoint.

## Tests

137 tests run without the `slow` marker (`-m "not slow"`, a few minutes); the
`slow` ones load the real BRONet. Everything not listed under "needs" runs on
synthetic data.

| file | what | needs |
|---|---|---|
| `test_bronet_mlp_math.py` | identity at init (all modes; also vs BRONet's `LiResMLP`, `minmax`); butterfly orthogonal, σ(RW) = σ(W); L_mlp kept by head-only/butterfly, changed by lora/full; power iteration ≈ exact; radius = BRONet `get_predictions`, a hand example, random perturbations below r never flip | BRONet source for 7 of them |
| `test_bronet_mlp_training.py` | same seed → same fingerprint, another seed → other weights; interrupted + resumed = straight; `load_run` reproduces the logits; incompatible `last.pt` refused; a crash while writing keeps the previous checkpoint | — |
| `test_bronet_mlp_earlystop.py` | best epoch = validation optimum, kept across a resume; report writes best and last | — |
| `test_bronet_mlp_tuning.py` | validation split grouped by photo, stratified, deterministic; selection rule; collapse marking; tuning end to end; a short run = prefix of a long one; finished run with other settings detected | — |
| `test_bronet_mlp_variants.py` | best / stable selection and its threshold; run names; the report shows both and skips runs that no longer match the selection | — |
| `test_bronet_mlp_cert_mcnemar.py` | certified McNemar counts, majority vote, Holm (known values, families, primary) | — |
| `test_bronet_mlp_attacks.py` | an attack never beats the certificate; an underestimated L is caught; PGD beats random noise; resume and refusal; plots | — |
| `test_bronet_mlp_strong_attacks.py` | DLR loss, targets, Square schedule; APGD / Square sound and strong, independent of batching; PGD folders and keys unchanged; CLI; through the pipeline; seeded resume | — |
| `test_bronet_mlp_min_eps.py` | eps* exact on a linear model; censoring and floor; an overestimated radius / underestimated L is caught; pipeline per attack; resume; quantiles with censored values | — |
| `test_bronet_mlp_warmup.py` | eps warm-up schedule; best epoch only after the ramp; tagged tuning folders; head-row logging | — |
| `test_bronet_mlp_lossy.py` | operations (JPEG, bits, resize, noise) and renormalisations; coverage arithmetic; sweep folders and report | — |
| `test_bronet_mlp_conv_adapt.py` | `PartitionedButterfly` orthogonal and identity at 0; conv_adapt starts at the frozen function, keeps rotations orthogonal, micro-batches = one batch; tuning selection; attacks, eps* and lossy coverage of a conv run; export | — |
| `test_bronet_mlp_real.py` | records == `repro.data.build_loaders(unique_only=True)`, sizes; transform has no Normalize; bad cache metadata refused | data (+ checkpoint and cache) |
| `test_bronet_mlp_real.py` (`slow`) | official L_pre × L_mlp = 8.1321 ± 1e-3 (float32, TF32 off; BRONet's 8.1305 is with TF32); cached φ == end-to-end forward; PGD through the real BRONet; fp16 gradient path ≈ fp32 | checkpoint, data, CUDA for fp16 |

The real-data tests use (and may fill) the caches under `DERMA_REPRO_OUT`.

## Known limitations

* The delivered split is by photograph, not by patient.
* Only high-level features after spatial pooling are adapted.
* Deduplicated training classes are imbalanced; no class weighting.
* The certificate covers L2 perturbations of the 224×224 input only — not
  illumination, dermatoscope or JPEG changes (JPEG and rounding are not
  Lipschitz).
* lr and epochs are tuned on a 10 % validation split of train (grouped by
  photo), never on test; the final runs train on the other 90 %, the paper's
  baselines on 100 %. The validation optimum is taken over many (lr, epoch)
  pairs on 1.4k images, so validation accuracy is optimistic.
* Attacks (stage 2, `attacks.py`) are on a fixed subset of test images, without
  clamping to [0,1] (as in the notebook). With `--grad-precision fp16` the PGD
  gradients use float16 BRO matrices; every reported success is re-evaluated in
  float32, so this can only weaken the attack, never fake a result.
