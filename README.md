# Quantum-inspired orthogonal adapters for certified medical image classification

Code of the paper "Quantum-inspired orthogonal adapters for certifiably robust
medical image classification". A pretrained network is adapted to a new task as
**R(θ)·W**, with R(θ) an orthogonal butterfly of Givens rotations (RBS gates).
R keeps every singular value of W, so the Lipschitz constant of a certified
network, and with it its certified radius, survives the adaptation; LoRA and full
fine-tuning do not keep it. The main example is trichoscopy, on the data of
Caposiena Caro et al., "Analysis of trichoscopic images using deep neural
networks for the diagnosis and activity assessment of alopecia areata",
JDDG 2026;24:44–55 (DOI 10.1111/ddg.15847).

## Repository layout

| folder | content |
|---|---|
| [`experiments/robustness/`](experiments/robustness/) | controlled experiments on MNIST (networks from scratch, depth, adaptation); `core.py` holds `ButterflyOrthogonal`, the paper's settings of eigenkit's `OrthogonalConv1x1` |
| [`experiments/bronet_mlp/`](experiments/bronet_mlp/) | certified BRONet on trichoscopy: its dense MLP adapted in four modes (head-only, butterfly, LoRA, full), certificates, attacks, eps*, JPEG coverage, eps_train sweep |
| [`experiments/resnet_adapt/`](experiments/resnet_adapt/) | control with a non-certified ResNet152: the same four modes on its last block `layer4.2`; Lipschitz upper bound; attacks |
| [`experiments/common/`](experiments/common/) | shared by the two trichoscopy experiments: output names (`layout`), locations, I/O, dataset records, certificate maths, losses, McNemar/Holm, the attack engine (`attacks/`) |
| [`trichoscopy/`](trichoscopy/) | package `repro`: data, deduplication, seeding, metrics; the reproduction of the CNNs of Caposiena Caro et al. (DenseNet169 / EfficientNetB0 / ResNet152) |
| [`experiments/parity/`](experiments/parity/) | `check_paper_numbers.py`: recomputes every number of Tables I–III from the result files and compares it with the paper |

Each folder has its own README with every command and output. Data and every
output live outside the repository.

## Installation

Python 3.12 and an NVIDIA GPU with a CUDA 12 driver. The results were produced on
an NVIDIA RTX 4070 Laptop GPU (8 GB) under Windows 11; some reproduction runs used
an AWS machine with a Tesla T4.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt          # Windows: .venv\Scripts\pip
```

[`requirements.txt`](requirements.txt) pins every package and its dependencies
(torch 2.5.1+cu121, torchvision 0.20.1+cu121, numpy 2.4.4, scipy 1.15.3,
pillow 10.4.0, ...). The orthogonal layers, the certificate and the losses come
from [eigenkit](https://github.com/eigenorbit/eigenkit) 1.2.2, installed from
PyPI. BRONet and its ImageNet checkpoint:

```bash
experiments/bronet_mlp/scripts/setup_bronet.sh            # Linux
.\experiments\bronet_mlp\scripts\setup_bronet.ps1         # Windows
```

(ntuaislab/BRONet at commit 9d59c9b; checkpoint `imagenet-14x588_399.pth`, its
sha256 is checked).

## Locations

Set by environment variables, or by the same keys in `paths.toml` in the
repository root (copy [`paths.example.toml`](paths.example.toml); not in git):

| variable | meaning | default |
|---|---|---|
| `DERMA_DATA_ROOT` | folder holding `model_data_step1_anon/` and `model_data_step2_anon/` | required |
| `DERMA_REPRO_OUT` | root of every output | `<DERMA_DATA_ROOT>/repro_out` |
| `BRONET_ROOT`, `BRONET_CKPT` | BRONet clone and checkpoint folder | `<DERMA_DATA_ROOT>/pretrained/...` |
| `DERMA_BASELINE_ROOTS` | where the `repro` CNN runs live (McNemar tests) | `<DERMA_REPRO_OUT>` |
| `DERMA_COLOR_REFERENCE` | the dataset's colour-normalisation reference photograph (only for `experiments.bronet_mlp.lossy --renorm authors\|ref`) | none |

`python -m repro.where` (with `PYTHONPATH=trichoscopy/src`) prints what is used.
All `python -m experiments...` commands run from the repository root.

## Reproducing the paper

The trichoscopy pipeline, per step (1 = diagnosis, 5 classes; 2 = alopecia areata
activity, 3 classes); every stage is resumable:

```bash
# BRONet: L_pre, feature cache, lr tuning on validation, 3 seeds per mode, report
experiments/bronet_mlp/scripts/run_step.sh --step 1 --seeds "0 1 2" --cache-bro-weights
experiments/bronet_mlp/scripts/run_step.sh --step 2 --seeds "0 1 2" --cache-bro-weights
# ResNet152 control (needs BRONet's tuning/step_<n>/val_split.json)
python -m experiments.resnet_adapt.features extract --step 1
python -m experiments.resnet_adapt.tune     --step 1 --batch-size 32
python -m experiments.resnet_adapt.run_grid --step 1 --seeds 0 1 2 --from-tuning --batch-size 32
python -m experiments.resnet_adapt.spectrum --step 1 --batch-size 32
```

| paper | what | command (step 1; step 2 likewise) | output under `<DERMA_REPRO_OUT>` |
|---|---|---|---|
| Fig. 2–5, 10, Table I | controlled experiments (MNIST, BloodMNIST → DermaMNIST, eps*, transforms) | see "Controlled experiments" below | — |
| Fig. 3b, Fig. 6, Table II | BRONet: accuracy, certified accuracy at 36/72/108 /255, L/L₀, radius, parameters, time; best-validation epoch, LoRA and full = `stable` | `run_step` above | `bronet_mlp/results/summary_step1.md`, `bronet_mlp/runs/step_1/<mode>/seed<s>/metrics_best.json` |
| statistics | McNemar with majority vote over seeds, Holm correction | written by the report | `bronet_mlp/results/mcnemar_step1.csv`, `mcnemar_cert_step1.csv` |
| Table III | ResNet152: accuracy, L₀ and L/L₀, certified accuracy, parameters | `resnet_adapt` commands above | `resnet_adapt/results/summary_step1_bs32.md`, `lipschitz_step1_bs32.md` |
| Fig. 7 | certified vs PGD vs APGD-T accuracy (seed 0, best epoch) | `python -m experiments.bronet_mlp.attacks --step 1 --n-images 300 --cache-bro-weights`, the same with `--attack apgd-t`; `python -m experiments.resnet_adapt.attacks --step 1 --batch-size 32` | `bronet_mlp/results/attacks/step_1_headline[_apgd-t]/`, `resnet_adapt/results/attacks/step_1_bs32_best/` |
| Fig. 8 | r(x) vs eps*(x) per image | `python -m experiments.bronet_mlp.min_eps --step 1 --n-images 300 --cache-bro-weights` | `bronet_mlp/results/attacks/step_1_headline_mineps/` |
| Fig. 9 | JPEG change vs radius; accuracy vs JPEG coverage over eps_train (last epoch) | `python -m experiments.bronet_mlp.lossy measure --step 1 --checkpoint final --cache-bro-weights`; `lossy sweep --step 1 --eps 72 108 144 195`; `lossy report --step 1 --eps 36 72 108 144 195 --checkpoint final` | `bronet_mlp/results/lossy/step_1_final_native/` |

Step 2 uses `--n-images 240` (the whole test set) for the attacks. The figures
and LaTeX tables of the paper are drawn from these files by the paper's plotting
scripts, which are not part of this repository.

### Controlled experiments

The MNIST, BloodMNIST → DermaMNIST, eps* (six seeds) and transform experiments of
the paper (Fig. 2–5, 10, Table I) are in
[`experiments/robustness`](experiments/robustness/), with its committed `results/`.
`experiments.py --experiment ...` writes most of the result files and the
scripts in [`reproduce/`](experiments/robustness/reproduce/README.md) the rest;
the folder's [README](experiments/robustness/README.md) gives every command and
the environment the results were made in. Its `core.py` also holds the butterfly
layer that the trichoscopy experiments import.

## Reproducibility

* **Seeds.** Every run is seeded (python, numpy, torch, CUDA). The data order,
  the validation split, the adapter initialisation, the test subsets of the
  attacks and every random perturbation draw from their own seeded generators, so
  they do not depend on batch composition, worker count or a resume (checkpoints
  store the RNG state).
* **Deterministic kernels.** Training sets `torch.backends.cudnn.deterministic`,
  disables cuDNN benchmarking, enables `torch.use_deterministic_algorithms` and
  sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` (when `repro` is imported); the
  trichoscopy experiments also disable TF32. `--no-deterministic` turns this off.
* **What is bit-identical.** Training, tuning and evaluation repeat bit for bit on
  the same GPU model with the same driver and the builds of `requirements.txt`:
  repeated short trainings of every mode give identical weights, logits, radii and
  predictions, and `python -m repro.selftest` checks the `repro` training
  (repeat, resume, worker count). Across GPU generations or library versions the
  numbers agree to about 1e-5, not bitwise (kernel choice; TF32 defaults of the
  `repro` CNNs).
* **Not bitwise.** Feature caches re-extracted with another batch size or GPU
  agree to about 1e-5. Attack gradients on the GPU are not forced to be
  deterministic, so success counts can differ in rare borderline cases; every
  reported success is re-evaluated exactly in float32 and checked against the
  certificate. JPEG results depend on the Pillow build (its version is recorded).
  The controlled experiments (`experiments/robustness`) are seeded but set no
  deterministic-kernel switches (their results were made on Apple MPS). Timings and
  absolute paths in `config.json` and provenance fields differ between machines.
* **Provenance.** Every run records the code hash, the git commit and dirty flag,
  the package versions, the device, and the sha256 of the checkpoint, the feature
  cache and the data lists.

## Tests

```bash
python -m pytest experiments/common/tests -q
python -m pytest experiments/bronet_mlp/tests -m "not slow" -q
python -m pytest experiments/resnet_adapt/tests -m "not slow" -q
python -m repro.selftest                            # PYTHONPATH=trichoscopy/src
cd experiments/robustness && python -m pytest tests -q
```

Run one suite per command: the bronet_mlp and resnet_adapt tests each import
their own `conftest` by name, so two suites in one pytest call collide. Tests
that need the data or BRONet skip themselves when those locations are not set
(see "Locations"), so the suites also pass on a fresh clone. The `slow` tests of
bronet_mlp need the data, the BRONet checkpoint and a GPU; those of resnet_adapt
take minutes on the CPU. The robustness tests download MNIST into `./data` of
the working folder (ignored by git).

## Dataset

Two classification tasks:

| step | task | classes (label: name) |
|---|---|---|
| 1 | scalp diagnosis | 0 AA (alopecia areata), 1 FFA-LPP (frontal fibrosing alopecia / lichen planopilaris), 2 DF (decalvans folliculitis), 3 AGA (androgenetic alopecia), 4 CHP (clinically healthy pattern) |
| 2 | alopecia areata activity | 0 AA active, 1 AA inactive, 2 AA in regrowth |

Layout under `DERMA_DATA_ROOT`:

```
model_data_step<1|2>_anon/{train,test}/<label>/*.jpg     600×600 images, one sub-folder per integer label
```

Labels are the sub-folder names (`0`, `1`, ...); the mapping to class names is
`CLASS_NAMES` in `trichoscopy/src/repro/config.py` and must not be reordered.

File names: `P<code>_<id>_<diagnosis>_<activity>[-aug<n>].jpg`, e.g.
`P0001_00001_aga_act.jpg`.

* `P<code>` — anonymised patient code (`P0001`, `P0002`, ...);
* `<id>` — acquisition id, normally 5 digits, occasionally with a trailing letter
  (e.g. `00002v`); `P<code>_<id>` identifies the photograph;
* `<diagnosis>_<activity>` — the clinical label suffix (e.g. `aga_act`), already
  folded into the folder label; some photos were relabelled after the shoot, and
  the file names carry the final labels;
* `-aug<n>` — offline-augmented copy (rotation, colour jitter); absent on originals.

| | step 1 | step 2 |
|---|---|---|
| train, delivered (originals + augmented copies) | 39 368 | 19 076 |
| train after removing byte-identical copies within a class (variant `unique`) | 31 665 | 13 707 |
| test, originals only | 717 (241 / 121 / 60 / 14 / 281) | 240 (78 / 48 / 114) |

The split is by photograph, not by patient. The paper's step-1 test set (641
images, CHP 205) is a different split from the delivered one; step-2 matches it
image for image.

## Reference numbers of Caposiena Caro et al.

The test results of the paper (Tables 5 and 6). ImageNet-pretrained networks,
head replaced by `Linear(→256) → ReLU → Dropout → Linear(→K)`, whole
network fine-tuned:

| model | step 1 accuracy | step 1 weighted F1 | step 2 accuracy | step 2 weighted F1 |
|---|---|---|---|---|
| EfficientNetB0 | **0.8892** | **0.8817** | 0.8083 | 0.8075 |
| DenseNet169 | 0.8736 | 0.8682 | **0.8333** | **0.8336** |
| ResNet152 | 0.8627 | 0.8540 | 0.7916 | 0.7913 |

## License

Apache-2.0, see [LICENSE](LICENSE).
