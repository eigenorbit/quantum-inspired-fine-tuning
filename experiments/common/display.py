"""How runs are named and drawn in every report and figure of the experiments."""

from __future__ import annotations

CHECKPOINT_TEXT = {
    "best": "weights of the epoch with the best VALIDATION accuracy (early stopping on the "
            "10 % validation split held out of train; training on the other 90 %)",
    "final": "weights of the last epoch",
}

# the colours of experiments/robustness/analysis.ipynb
CM = {"head-only": "#888888", "butterfly": "#d62728", "lora": "#1f77b4", "full": "#444444"}
LM = {"head-only": "head-only (no feature adaptation)", "butterfly": "butterfly (quantum orthogonal)",
      "lora": "LoRA", "full": "full finetuning"}
ORDER = ("head-only", "butterfly", "lora", "lora_stable", "full", "full_stable")
STEP_TITLE = {1: "step-1: diagnosis, 5 classes", 2: "step-2: AA activity, 3 classes"}


def label(name: str) -> str:
    """Run folder name -> table label: `lora_stable` -> `lora (stable)`."""
    return name.replace("_stable", " (stable)")


def base_mode(run_name: str) -> str:
    return run_name.replace("_stable", "")


def style(run_name: str) -> dict:
    mode = base_mode(run_name)
    stable = run_name.endswith("_stable")
    return {"color": CM[mode], "ls": "--" if stable else "-",
            "label": LM[mode] + (" — stable lr" if stable else "")}
