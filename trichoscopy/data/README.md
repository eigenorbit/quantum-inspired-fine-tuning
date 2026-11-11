# Data

No images are committed here — the dataset is roughly 4 GB. This folder exists
so the repository has the usual `data/` slot, and to document where the images
actually are and what shape they have.

## Where the code looks

`repro.paths` reads `DERMA_DATA_ROOT`: the environment variable, else the same
key in `paths.toml` in the repository root (copy `paths.example.toml`; see
`repro.settings`). There is no built-in default. That folder must contain:

```
model_data_step1_anon/
    train/{0,1,2,3,4}/*.jpg      39,368 images
    test/ {0,1,2,3,4}/*.jpg       9,851 images (717 of them original)
model_data_step2_anon/
    train/{0,1,2}/*.jpg          19,076 images
    test/ {0,1,2}/*.jpg           4,809 images (240 of them original)
```

Verify with:

```powershell
python -m repro.check
```

## Labels

Sub-folder names are sorted and mapped to consecutive integer labels, so the
folder name *is* the label.

**Step-1** — diagnosis. The paper's Table 5 prints `FD` for class 2 and `HV`
for class 4; the folders call them `DF` and `CHP`.

| Label | Code name | Paper | Meaning |
| --- | --- | --- | --- |
| 0 | AA | AA | alopecia areata |
| 1 | FFA-LPP | FFA-LPP | frontal fibrosing alopecia / lichen planopilaris |
| 2 | DF | FD | folliculitis decalvans |
| 3 | AGA | AGA | androgenetic alopecia |
| 4 | CHP | HV | clinically healthy pattern |

**Step-2** — alopecia areata activity.

| Label | Name |
| --- | --- |
| 0 | AA active |
| 1 | AA inactive |
| 2 | AA in regrowth |

## Filenames carry the ground truth

```
P<patient>_<image>_<diagnosis>_<activity>[-aug<n>].jpg
P0001_00001_aa_act.jpg           an original photograph
P0002_00002_aga_ina-aug3.jpg     an augmented copy of P0002_00002_aga_ina
P0003_00042v_aa_ingr.jpg         an image id need not be purely numeric
```

Two consequences the code relies on:

* **`-aug` marks generated copies.** Dropping them from the test split
  reproduces the evaluation sets the paper reports on — it says so explicitly:
  "The performance measures were calculated on the test dataset excluding the
  augmented images."
* **Labels come from the processed filenames, never from `data_raw_used_anon`.**
  84 of 3,583 images were relabelled after the shoot; only the processed names
  carry the final labels. Deriving labels from the raw folder gives AGA = 111
  and CHP = 1,361 instead of the correct AGA = 67 and CHP = 1,405.

## Provenance

The images are already preprocessed. The pipeline that produced them is:

1. crop 420 px from each side → 1080×1080
2. colour-normalise against a reference image of the dataset
3. resize to 600×600, bilinear
4. save as JPEG at PIL's default quality 75

Steps 1–2 follow the source study. The path of the reference
image is configured via the `DERMA_COLOR_REFERENCE` setting: the environment
variable, else the same key in `paths.toml` (see `repro.settings`). The
augmentation (flips, rotations, PCA colour jitter) was applied offline when
these folders were written, which is why training here applies no random
transform of its own.
