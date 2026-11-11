# Quantum orthogonal layers and the Lipschitz certificate

Does the **quantum orthogonal layer** — a butterfly circuit of RBS gates, which on unary
states is a product of Givens rotations and therefore an orthogonal matrix — buy anything
over ordinary spectral normalisation, at a matched Lipschitz certificate?

**Start here: [`analysis.ipynb`](analysis.ipynb)** — the full write-up: architecture,
experimental setup, results, figures, conclusions and next steps. Two experiments, each run
twice: on MNIST (§5–6) and on **DermaMNIST** (§7), a 7-class, 67%-imbalanced RGB
dermatoscopic dataset. It runs from the committed CSVs by default (`RUN = False`), so you
can read it without retraining anything.

**No Jupyter installed?** Open [`analysis.html`](analysis.html) in any browser — same content,
figures embedded, nothing to install.

Every accuracy is reported twice — **plain** and **balanced** (mean per-class recall) — and
every figure carries a dashed green line for an **ordinary unconstrained CNN** trained on the
same data with the same budget. It is not a fourth arm: it has no certificate and cannot have
one. It is there so that an accuracy can be read against something.

## The headline

**A certified backbone adapted to a new task by a quantum orthogonal circuit keeps its
certificate exactly, while LoRA and ordinary finetuning destroy it — at equal or better
clean accuracy and with fewer trainable parameters.**

Use it when you need a *proof*. Section 8.9 is the deployable form: an orthogonally adapted
model can **prove** that JPEG re-encoding, bit-depth reduction, moderate resizing and sensor
noise leave its diagnosis unchanged — on ~75% of clinical images, 100% on easier tasks —
where LoRA and full fine-tuning prove none of it. Section 8.7 shows where it stops: when
robustness has to be *learned* rather than inherited, an unconstrained adapter wins, because
learned invariance needs contraction and orthogonality forbids it.

`mnist35` → `mnist08`, all four adapters starting from the same certified backbone:

| | head-only | **butterfly** | LoRA | full finetune |
|---|---|---|---|---|
| clean accuracy | 0.9905 | **0.9936** | 0.9918 | 0.9913 |
| `L_analytic` | 1.000 | **1.000** | 1529 | 1825 |
| certified radius | 1.116 | **1.760** | 0.003 | 0.004 |
| robust accuracy @ ε=2.0 | 0.050 | **0.395** | 0.018 | 0.059 |
| trainable parameters | 2,050 | 8,194 | 18,434 | 135,170 |

Why it works: composing a layer with an orthogonal 1×1 convolution leaves its singular
values unchanged, so the certificate survives exactly — and a 1×1 orthogonal convolution
*is* a quantum orthogonal layer (an RBS circuit in the unary basis). This is the mechanism
behind orthogonal finetuning (OFT, BOFT, qGOFT, QuIC); none of those papers claim it as a
robustness property.

The same experiment on clinical data — `bloodmnist` → `dermamnist`, a real domain shift from
a balanced 8-class source to an imbalanced 7-class target — gives the same certificate result
and one new finding:

| | head-only | **butterfly** | LoRA | full finetune |
|---|---|---|---|---|
| balanced accuracy | 0.506 | 0.515 | **0.548** | 0.265 |
| `L_analytic` | 1.000 | **1.000** | 3.8e7 | 1.4e19 |
| certified radius | 0.181 | **0.214** | 0.000 | 0.000 |
| robust accuracy @ ε=1.0 | 0.053 | **0.112** | 0.002 | 0.000 |
| trainable parameters | 28,679 | 61,447 | 94,215 | 2,134,023 |

On MNIST the orthogonal adapter was also the most accurate, so nothing had to be traded. Here
**LoRA wins on accuracy by 3.3 balanced points and gives up the certificate to get it.** That
trade — about three points of balanced accuracy against a radius-0.21 guarantee and four times
the robust accuracy under attack — is the number a deployment would actually have to weigh.

## The short version

| | finding |
|---|---|
| Random noise | No advantage, and no vulnerability. Every model keeps 96–97% accuracy under a random perturbation of L2 norm 3.0. |
| Worst-case noise | Large advantage that **grows with depth**. At depth 128 the orthogonal model keeps 48% accuracy under PGD-L2 at ε=1.5; the unconstrained network keeps 0.1%. |
| Spectral normalisation | Becomes unreliable past depth 32 — one of two seeds collapses at depth 64, both at depth 128, where it has destroyed 93% of the signal. |
| Cost | 0.1 accuracy points at depth 128 (within seed noise), at 16× fewer mixing parameters. |
| On clinical data | The same failure four times sooner. On DermaMNIST, spectral normalisation collapses on **both** seeds at depth 16; the unconstrained model stays trainable but its bound explodes to 10³⁹. The orthogonal model is the only arm still learning at depth 64. |
| At shallow depth | No advantage. At depth 4 the two certified arms are indistinguishable (certified radius 0.176 vs 0.175) and spectral normalisation is in fact **twice as robust** under attack — heavy contraction is itself a defence, until it compounds. This is a statement about depth, not about spectral normalisation as such. |
| What the certificate costs | About **9 balanced-accuracy points** on DermaMNIST. Against an ordinary CNN at 0.621, our backbone with an unconstrained mixer gets 0.605 — so the constrained backbone is nearly free — and the same backbone at L = 1 gets 0.516. |
| Why the numbers look low | Mostly the metric. Trained the ordinary way our certified model reaches 0.706 *plain* accuracy against a published ResNet-18 at 0.735 — but it gets there by never predicting three of the seven classes. §7.6 reports both metrics side by side. |
| Harder to attack? Depends on the loss | Asked for the same **margin** — what ordinary fine-tuning does — butterfly is **2.3× harder to break** than LoRA (six seeds, p = 0.031). Asked for the same **certified radius**, the objective charges LoRA for its inflated L, LoRA compensates with larger margins, and the ImageNet-scale replication measures it 10–35% ahead instead. Both are real; they answer different questions. The certificate doesn't depend on the choice. |
| Is the certificate informative? | Butterfly's proof stops guaranteeing at 1.35 and the model breaks at 1.48 — it captures **92% of the real safe zone**. LoRA's captures 0.3%, full fine-tuning's 10⁻¹⁶%. Preserving L is not bookkeeping: the bound stays a description of the model. |
| What the certificate covers | Not lighting (5–20× too small), but **all ordinary lossy image handling**: JPEG down to q=80, 6-bit depth, resizing through 20×20, sensor noise. Butterfly proves 72–89% of DermaMNIST images unchanged and 100% of MNIST; LoRA and full prove 0%. And `full` — the arm that proves nothing — is the one that actually changes its answer most (10.5% after a resize). |
| The radius is trainable | Asking the loss for a bigger margin **more than doubles the certified radius at no cost in accuracy** (0.42 → 0.92, 3 seeds). And the adapter has headroom the frozen backbone does not: `head-only` saturates at 0.45 because it only moves the final layer, while `butterfly` keeps going because it can rotate the representation. A reason to prefer the orthogonal adapter that has nothing to do with preserving L. |
| Acquisition changes | Against a worst-case change of lighting, focus, tilt or white balance, full fine-tuning collapses on every seed — but the apparent advantage over LoRA does not survive three seeds (one reverses it significantly), and neither does it at ImageNet scale. **There is no reliable empirical advantage over LoRA here**, and the certificate does not reach this family anyway (5–20× outside it). |
| The limit of the idea | Orthogonality cannot *learn* invariance: retrain on the acquisition family and LoRA overtakes everything, because invariance needs to **contract** the nuisance directions and an orthogonal map cannot contract anything. The property that preserves the proof is the property that prevents the invariance. |
| Where the CNN wins | On `mnist08` (0 vs 8) the unconstrained CNN is **more** robust than every adapter — 0.713 at ε=2.0 against the orthogonal adapter's 0.395. On an easy, well-separated binary task an unconstrained model sits in a wide basin. §6.4 says so, and notes that 0.395 is a lower bound backed by a proof while 0.713 is an upper bound a stronger attack can only lower. On DermaMNIST the ordering reverses completely. |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-lock.txt   # exact versions
.venv/bin/python -m pytest tests/ -q          # 77 tests, ~20 s
.venv/bin/jupyter lab analysis.ipynb          # or just open analysis.html in a browser
```

`requirements-lock.txt` is the pinned environment of the repository (torch 2.5.1,
eigenkit 1.2.2) plus medmnist; `requirements.txt` lists the direct
dependencies only. On an M-series Mac use `.venv-freeze` (below) instead: with
torch 2.5.1, six tests stop on MPS; in `.venv-freeze` all 77 pass. MNIST and MedMNIST are downloaded into `./data` of the working
folder, so run the tests and the experiments from a folder outside the
repository.

The committed `results/` were computed on an M-series Mac (MPS) with torch 2.14.0
and Python 3.14, `requirements-freeze.txt`. To reproduce them exactly, use that
environment, which also pins eigenkit 1.2.2 and the four packages it needs:

```bash
python3.14 -m venv .venv-freeze
.venv-freeze/bin/pip install -r requirements-freeze.txt
```

In that environment the runs give the committed numbers bit for bit: re-running
10 adaptation runs (MNIST, every mode and seed; DermaMNIST, butterfly and LoRA)
reproduced every column of `adaptation.csv` and `derma_adaptation.csv` on an
Apple M5 Pro. Another Apple chip may differ in the last bits. The DermaMNIST
results use 300 attack images (`--n-images 300`), the MNIST ones the default 500.

The pinned environment of the repository (torch 2.5.1) cannot run the butterfly
on MPS: torch 2.5.1 has no MPS kernel for `index_copy`, which builds the butterfly
matrix, so the code stops with `NotImplementedError`. With
`PYTORCH_ENABLE_MPS_FALLBACK=1` torch runs that operation on the CPU, and all 77
tests pass (in about 80 s). On a CUDA GPU or the CPU it runs, with other bits
than the committed results.
`resolve_device("auto")`, which the command line uses, picks MPS whenever it is
present.

To regenerate every number from scratch (in `.venv-freeze` on an M-series Mac):

```bash
.venv-freeze/bin/python experiments.py --experiment depth        # section 5, ~45 min
.venv-freeze/bin/python experiments.py --experiment adaptation   # section 6, ~10 min
```

The DermaMNIST replication of section 7 is the same two commands with a different dataset.
It is slower — 256 channels instead of 64, and 30 epochs instead of 10 — at roughly 3 h and
1 h. `--class-weights` is not optional on this data: without it no arm learns anything.

```bash
.venv-freeze/bin/python experiments.py --experiment depth --dataset dermamnist \
    --depths 4 16 32 64 --epochs 30 --class-weights --out-prefix derma_
.venv-freeze/bin/python experiments.py --experiment adaptation --source bloodmnist \
    --target dermamnist --epochs 30 --class-weights --n-images 300 --out-prefix derma_
.venv-freeze/bin/python experiments.py --experiment baseline --dataset dermamnist  # §7.6, ~30 min
```

Section 8 measures, per image, the smallest perturbation that actually flips it, and compares
that with what the certificate proves:

```bash
.venv-freeze/bin/python experiments.py --experiment tightness --dataset mnist35 \
    --depth 128 --seeds 2 --epochs 10 --n-attack-images 200
.venv-freeze/bin/python experiments.py --experiment tightness-adapt --source mnist35 \
    --target mnist08 --seeds 2 --epochs 10 --depth 32 --n-attack-images 200
.venv-freeze/bin/python experiments.py --experiment attack-strength --dataset mnist35 \
    --arms dense dense-sn butterfly --depth 128 --epochs 10
```

Add `--dataset dermamnist --class-weights --out-prefix derma_` for the clinical half.

The green reference line in the MNIST figures comes from the same command, restricted to the
CNN and matched to section 5's budget:

```bash
.venv-freeze/bin/python experiments.py --experiment baseline --dataset mnist35 \
    --models simple-cnn --seeds 2 --epochs 10
.venv-freeze/bin/python experiments.py --experiment baseline --dataset mnist08 \
    --models simple-cnn --seeds 2 --epochs 10
```

The third command is the calibration grid of §7.6: an ordinary unconstrained CNN and three
of our models, each under four training recipes, so the cost of the Lipschitz constraint can
be separated from the cost of the dataset and the metric.

## Files

```
analysis.ipynb   the write-up: architecture, setup, results, conclusions, next steps
analysis.html    the same notebook as a standalone web page (no Jupyter needed)
core.py          the butterfly layer, the mixers, DepthNet, and the data loaders
experiments.py   training, measurement, both perturbations, and the two experiments
init_draws.py    the starting draws of the butterfly adapters
tests/           77 tests: the circuit is orthogonal, the bound holds under attack,
                 no image ever breaks inside its certified radius, spectral
                 normalisation contracts where orthogonality does not, the RGB
                 lift is an isometry, and the CNN reference is not certified
results/         depth.csv (36 models), perturbation.csv (324), adaptation.csv (72)
                 derma_*.csv — the same three for section 7 (24 / 216 / 72)
                 baseline.csv, baseline_perturbation.csv — the unconstrained CNN,
                 on every task and under four training recipes
                 tightness*.csv — the per-image breaking point and what the
                 certificate proves, for both studies on both datasets
                 attack_strength.csv — four attacks of increasing strength
                 offset_sweep.csv — trading loss margin for certified radius
                 pipeline.csv — what a PACS does to an image, against the radius
                 semantic*.csv, figure_data_*.npz — the acquisition-family
                 study, with and without training on that family, and the
                 arrays behind the section 8 figures
```

## Reproducing the result files

`experiments.py --experiment {depth,adaptation,baseline,tightness,tightness-adapt,attack-strength}`
covers most of `results/`. The remaining files -- `epsstar_seed*.npz`, `offset_sweep.csv`,
`pipeline.csv`, `figure_data_*.npz`, `acquisition_seed*.npz` -- are written by the scripts in
[`reproduce/`](reproduce/README.md), which also documents every setting used and the exact
environment (Python 3.14.6, torch 2.14.0, medmnist 3.0.2, MPS, 28x28 images).
