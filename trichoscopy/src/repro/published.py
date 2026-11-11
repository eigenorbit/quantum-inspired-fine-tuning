"""The numbers this reproduction is measured against.

Transcribed from Caposiena Caro, Orlova, Di Meo, Zalaudek, "Analysis of
trichoscopic images using deep neural networks for the diagnosis and activity
assessment of alopecia areata", JDDG 2026;24:44-55 (DOI 10.1111/ddg.15847):
Table 5 (Step-1, diagnostic classification) and Table 6 (Step-2, activity
level). All values are percentages, exactly as printed.

Two notes on the transcription:

* The paper's tables name class 2 "FD" and class 4 "HV"; the image folders
  call them "DF" and "CHP". The folder names are used here.
* Table 5, DenseNet169, HV recall prints 91.12. That value is not consistent
  with the rest of its own column: 187/205 = 91.22. `KNOWN_TYPOS` records it.
  The printed value is what `TABLE_5` stores, so the comparison table shows the
  paper as published.
"""

from __future__ import annotations

# (step, architecture) -> published percentages.
PUBLISHED: dict[tuple[int, str], dict] = {
    # --- Table 5: Step-1, alopecia areata diagnostic classification ----------
    (1, "densenet169"): {
        "accuracy": 87.36,
        "f1_weighted": 86.82,
        "per_class": {
            "AA":      {"precision": 89.87, "recall": 88.38, "f1": 89.12},
            "FFA-LPP": {"precision": 86.55, "recall": 85.12, "f1": 85.83},
            "DF":      {"precision": 91.66, "recall": 91.66, "f1": 91.66},
            "AGA":     {"precision": 66.67, "recall": 14.29, "f1": 23.53},
            "CHP":     {"precision": 84.23, "recall": 91.12, "f1": 87.59},
        },
    },
    (1, "efficientnet_b0"): {
        "accuracy": 88.92,
        "f1_weighted": 88.17,
        "per_class": {
            "AA":      {"precision": 89.88, "recall": 92.12, "f1": 90.98},
            "FFA-LPP": {"precision": 90.09, "recall": 82.64, "f1": 86.21},
            "DF":      {"precision": 94.83, "recall": 91.67, "f1": 93.22},
            "AGA":     {"precision": 50.00, "recall":  7.14, "f1": 12.50},
            "CHP":     {"precision": 86.10, "recall": 93.66, "f1": 89.71},
        },
    },
    (1, "resnet152"): {
        "accuracy": 86.27,
        "f1_weighted": 85.40,
        "per_class": {
            "AA":      {"precision": 87.92, "recall": 87.55, "f1": 87.73},
            "FFA-LPP": {"precision": 89.38, "recall": 83.47, "f1": 86.32},
            "DF":      {"precision": 91.53, "recall": 90.00, "f1": 90.76},
            "AGA":     {"precision":  0.00, "recall":  0.00, "f1":  0.00},
            "CHP":     {"precision": 82.02, "recall": 91.22, "f1": 86.37},
        },
    },
    # --- Table 6: Step-2, alopecia areata activity level ---------------------
    (2, "densenet169"): {
        "accuracy": 83.33,
        "f1_weighted": 83.36,
        "per_class": {
            "AA active":      {"precision": 79.27, "recall": 83.33, "f1": 81.25},
            "AA inactive":    {"precision": 90.48, "recall": 79.16, "f1": 84.44},
            "AA in regrowth": {"precision": 83.72, "recall": 85.09, "f1": 84.35},
        },
    },
    (2, "efficientnet_b0"): {
        "accuracy": 80.83,
        "f1_weighted": 80.75,
        "per_class": {
            "AA active":      {"precision": 90.47, "recall": 73.08, "f1": 80.85},
            "AA inactive":    {"precision": 83.33, "recall": 72.92, "f1": 77.78},
            "AA in regrowth": {"precision": 75.56, "recall": 89.47, "f1": 81.92},
        },
    },
    (2, "resnet152"): {
        "accuracy": 79.16,
        "f1_weighted": 79.13,
        "per_class": {
            "AA active":      {"precision": 76.00, "recall": 73.08, "f1": 74.51},
            "AA inactive":    {"precision": 84.44, "recall": 79.17, "f1": 81.72},
            "AA in regrowth": {"precision": 79.16, "recall": 83.33, "f1": 81.20},
        },
    },
}

# Test-set composition the published metrics were computed on (Results, p. 48
# and the recalls of Table 5 / Table 6, which pin every denominator).
PUBLISHED_SUPPORT = {
    1: {"AA": 241, "FFA-LPP": 121, "DF": 60, "AGA": 14, "CHP": 205},  # 641
    2: {"AA active": 78, "AA inactive": 48, "AA in regrowth": 114},   # 240
}

# Printed values that their own column contradicts.
KNOWN_TYPOS = [
    (
        "Table 5, DenseNet169, HV recall",
        "printed 91.12, consistent value 91.22 (= 187/205)",
    ),
]


def get(step: int, architecture: str) -> dict:
    """Published block for one (step, architecture); ``KeyError`` if absent."""
    return PUBLISHED[(step, architecture)]
