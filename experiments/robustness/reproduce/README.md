# Reproducing the result files

The notebook only *reads* `results/`. These scripts are what wrote the files that
are not produced by `experiments.py --experiment {depth,adaptation,baseline,tightness,...}`.
They were collected here after the runs; the settings below are the ones that
produced the committed files.

Run them from this directory's parent (`experiments/robustness`):

| file in `results/` | script | runs | used in |
|---|---|---|---|
| `epsstar_seed{0..5}.npz` | `python reproduce/epsstar.py SEED` | 6 (seed 0..5) | Fig. 5b |
| `offset_sweep.csv` | `python reproduce/offset_sweep.py SEED` | 3 (seed 0,1,2), then concatenate | Fig. 10 |
| `pipeline.csv` | `python reproduce/pipeline.py` | 1 | Fig. 10 |
| `figure_data_mnist.npz` | `python reproduce/figure_data.py mnist` | 1 | Fig. 4 |
| `figure_data_derma.npz` | `python reproduce/figure_data.py derma` | 1 | Fig. 4 |
| `acquisition_seed{0..5}.npz` | `python reproduce/acquisition.py SEED` | 6 | §8.4, not in the article |

`semantic*.csv` are superseded by `acquisition_seed*.npz` (the same measurement with
per-image outcomes and 6 seeds instead of 1). Their driver is not included; §8.4 of the
notebook reads the `.npz` files, so nothing in the article depends on the `.csv`.

`offset_sweep.py` writes one file per seed; concatenate afterwards:

```bash
python -c "import glob,pandas as pd; \
  pd.concat([pd.read_csv(f) for f in sorted(glob.glob('results/offset_sweep_seed*.csv'))]) \
    .to_csv('results/offset_sweep.csv', index=False)"
```

## Settings, in one place

Everything shared by all of these:

| | |
|---|---|
| backbone | `butterfly`, **depth 32**, trained from scratch by `build_and_train` |
| source → target | `mnist35` → `mnist08`, and `bloodmnist` → `dermamnist` |
| epochs | **10** for the MNIST transfer, **30** for `bloodmnist` → `dermamnist` (backbone *and* adapter) |
| class weights | off for MNIST, **on** for `dermamnist` (inverse frequency; without them no arm learns) |
| adapter rank | 4 (LoRA) |
| seeding | backbone and data split use `SEED`; the adapter uses `set_seed(SEED + 1)` |
| images for the expensive measurements | first **200** of `loaders["eval"]`, except `acquisition.py` (256) and `figure_data.py` (first **64**) |
| optimiser | Adam, lr `1e-3`, batch 128, loss `margin_loss(scale=8, offset=1)` unless the script overrides `offset` |

Per script:

* **`epsstar.py`** — modes `butterfly` and `lora` only, both deep-copied from **one** backbone per
  seed so the adapter is the only difference. `min_adversarial_distance` with
  `eps_start=0.25`, `eps_max=8.0`, `bisect=8`, attack `pgd_l2(steps=50, restarts=2, decay=True)`,
  attack generator seeded `SEED + 777`. Misclassified images record 0; images no budget up to
  `eps_max` breaks record `+inf` (**censored, not clipped** — a collapsed constant predictor is
  unbreakable for reasons that have nothing to do with robustness).
* **`offset_sweep.py`** — modes `butterfly` and `head-only`; offsets **0.5, 1.0, 2.0, 4.0, 8.0**.
  1.0 is the default used everywhere else in the notebook.
* **`pipeline.py`** — all four modes, both transfers, seed 0. Sensor-noise draws use a generator
  seeded **3**. The operations are defined by `pipeline_ops` in `experiments.py`
  (JPEG q = 100/95/90/80, 7- and 6-bit depth, resize through 24/20/14, noise σ = 0.002/0.005).
* **`figure_data.py`** — shared budget ε = **1.3** (mnist) / **0.35** (derma);
  `adversarial_delta(steps=100, restarts=5)`; semantic attack `steps=60, restarts=3`.
  The per-arm `eps_star` values baked into the script are the medians measured in §8.2 and are
  **inputs** to the figure, not results of it.
* **`acquisition.py`** — all four modes, severities 0.0/0.1/0.2/0.3/0.4/0.6, generator seeded 11,
  `semantic_attack(steps=60, restarts=3)`.

### How the example image in Fig. 4 is selected

From the first 64 evaluation images, keep those that

1. all four adapters classify **correctly when clean**,
2. the **butterfly** adapter still classifies correctly at the shared budget, and
3. **all three other adapters get wrong** at that same budget,

then take the lowest index. If no image satisfies (3) for all three, the rule relaxes to at
least two of three.

This is deliberately a favourable example, and the caption must say so. The `counts` array in
the `.npz` carries the honest survivor count out of 64 for each arm — quote it beside the
figure.

## Environment

The committed files were produced on:

| | |
|---|---|
| Python | **3.14.6** |
| platform | macOS 26.7.1, arm64 (Apple silicon) |
| device | **MPS** (`resolve_device()` picks `mps` → `cuda` → `cpu`) |
| torch | **2.14.0** |
| torchvision | **0.29.0** |
| numpy | **2.5.3** |
| medmnist | **3.0.2** |
| pandas | 3.0.6 |
| scipy | 1.18.1 |
| pillow | 12.3.0 |
| matplotlib | 3.11.2 |

Full freeze: [`../requirements-freeze.txt`](../requirements-freeze.txt).

**MedMNIST images are 28×28**, the default release size — `build_loaders` does not pass a
`size=` argument, so it loads `dermamnist.npz` / `bloodmnist.npz` at 28×28. Batches are
`(B, 3, 28, 28)` for DermaMNIST and `(B, 1, 28, 28)` for MNIST, pixels scaled to `[0, 1]` with
**no mean/std normalisation and no augmentation**, so a perturbation norm is directly
interpretable in input space and a certified radius means what it says.

## A caveat about exact reproduction

MPS is not bit-deterministic across torch versions, and `pgd_l2` sums over a batch, so a rerun
may differ in the last digits. The committed CSVs were checked to reproduce bit-for-bit on the
machine and version above. Treat medians and seed-level conclusions as the reproducible unit,
not individual per-image values.
