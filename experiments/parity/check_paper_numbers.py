"""Check every number of Tables I-III of the paper against the result files.

    python experiments/parity/check_paper_numbers.py [--table I II III]

The numbers, as printed, and where each comes from are in `paper_numbers.json`
(next to this file). For each one the value is computed from the results (mean
± sample SD over the seeds, or the mean, or the single value) and formatted with
the digits the paper prints; it passes when the two strings are equal.

- Table I reads experiments/robustness/results/*.csv, in this repository.
- Tables II and III read the runs under DERMA_REPRO_OUT (`metrics_best.json` of
  bronet_mlp/runs/step_<n>/<run>/seed<s> and resnet_adapt/runs/step_<n>_bs32/...);
  Table III's L/L0 and PGD columns read resnet_adapt/results under DERMA_REPRO_OUT
  (`lipschitz_step<n>_bs32.json`, `attacks/.../attacks_step<n>.csv`). Without these
  files those rows are MISSING, not failures.
- The number of seeds found must be the one of the table (2, 3, 3), or the
  entry's own `seeds` (1 for the seed-0 PGD columns).
- Entries marked `exact: false` (training time) are shown, never failures;
  entries without a source are listed as MANUAL. The training time is the sum of
  `seconds_train` over history.json: training steps only, without evaluation and
  checkpoint writes.
- Every file is read as UTF-8 (the printed numbers contain "±").

Exit status 1 if any number differs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = {"I": 2, "II": 3, "III": 3}          # as the captions of the tables say
REPO_ROOT = HERE.parents[1]


# ---------------------------------------------------------------------------
# formatting as printed
# ---------------------------------------------------------------------------

def _decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def render(values: list[float], printed: str) -> str:
    """`values` (one per seed) in the format of `printed`."""
    if "±" in printed:
        mean_txt, sd_txt = printed.split("±")
        sd = statistics.stdev(values) if len(values) > 1 else float("nan")
        return f"{statistics.fmean(values):.{_decimals(mean_txt)}f}±{sd:.{_decimals(sd_txt)}f}"
    value = statistics.fmean(values)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)e(\d+)", printed)
    if m:                                            # 3.8e7 stands for 3.8 x 10^7
        exponent = math.floor(math.log10(abs(value)))
        mantissa = value / 10 ** exponent
        if round(mantissa, _decimals(m.group(1))) >= 10:
            exponent, mantissa = exponent + 1, mantissa / 10
        return f"{mantissa:.{_decimals(m.group(1))}f}e{exponent}"
    if printed.endswith("k"):                        # 4,473k: thousands, with commas
        return f"{round(value / 1000):,}k"
    if "," in printed or printed.isdigit() and len(values) == 1 and value == int(value):
        return f"{round(value):,}"
    return f"{value:.{_decimals(printed)}f}"


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------

def _number(text: str) -> float:
    return float(text)


def from_csv(src: dict, root: Path = REPO_ROOT) -> list[float]:
    with (root / src["csv"]).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if all(
        (_number(r[k]) == v if isinstance(v, float) else r[k] == str(v))
        for k, v in src["where"].items())]
    expr = src["value"]
    if "/" in expr:
        num, den = expr.split("/")
        values = [_number(r[num]) / _number(r[den]) for r in rows]
    else:
        values = [_number(r[expr]) for r in rows]
    return [v * src["scale"] for v in values]


def _get(d: dict, dotted: str):
    """`a.b.36/255` -> d["a"]["b"]["36/255"]."""
    for part in dotted.split("."):
        d = d[part]
    return d


def _tree(package: str):
    sys.path.insert(0, str(REPO_ROOT))
    import experiments.bronet_mlp  # noqa: F401  (locations: DERMA_REPRO_OUT and its default)
    from experiments.common import layout

    return {"bronet_mlp": layout.BRONET, "resnet_adapt": layout.RESNET}[package]


def runs_root(package: str, step: int, batch_size: int) -> Path:
    return _tree(package).runs_dir(step, batch_size)


def from_runs(src: dict) -> list[float] | None:
    """`value` of metrics_<which>.json per seed, or, with `history_sum`, that field
    summed over the epochs of history.json (e.g. `seconds_train`, the training steps
    without evaluation and checkpoint writes)."""
    root = runs_root(src["runs"], src["step"], src.get("batch_size", 256)) / src["run"]
    if "history_sum" in src:
        files = sorted(root.glob("seed*/history.json"))
        return [sum(float(e[src["history_sum"]])
                    for e in json.loads(f.read_text(encoding="utf-8"))) * src["scale"]
                for f in files] or None
    files = sorted(root.glob(f"seed*/metrics_{src['which']}.json"))
    if not files:
        return None
    return [float(_get(json.loads(f.read_text(encoding="utf-8")), src["value"])) * src["scale"]
            for f in files]


def from_results(src: dict) -> list[float] | None:
    """A file in <package>/results: a CSV (rows chosen by `where`, as in Table I),
    or a JSON whose `list` holds one record per run and seed, chosen by `where`."""
    root = _tree(src["results"]).results_dir()
    if not (root / src["file"]).exists():
        return None
    if src["file"].endswith(".csv"):
        return from_csv({**src, "csv": src["file"]}, root)
    records = _get(json.loads((root / src["file"]).read_text(encoding="utf-8")), src["list"])
    return [float(_get(r, src["value"])) * src["scale"] for r in records
            if all(r.get(k) == v for k, v in src["where"].items())]


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--table", nargs="+", default=["I", "II", "III"])
    p.add_argument("--numbers", type=Path, default=HERE / "paper_numbers.json")
    a = p.parse_args(argv)

    entries = [e for e in json.loads(a.numbers.read_text(encoding="utf-8"))["numbers"]
               if e["table"] in a.table]
    counts = {"OK": 0, "DIFFERS": 0, "MISSING": 0, "MANUAL": 0, "INFO": 0}
    for e in entries:
        where = f"Table {e['table']} | {e['row']} | {e['column']}"
        src = e["source"]
        if src is None:
            counts["MANUAL"] += 1
            print(f"MANUAL   {where}: paper {e['printed']} ({e.get('note', '')})")
            continue
        try:
            values = (from_csv(src) if "csv" in src
                      else from_results(src) if "results" in src else from_runs(src))
        except Exception as err:  # noqa: BLE001  (an unreadable file is reported, not fatal)
            values, missing_why = None, f"{type(err).__name__}: {err}"
        else:
            missing_why = "no result files"
        if not values:
            counts["MISSING"] += 1
            print(f"MISSING  {where}: paper {e['printed']} ({missing_why})")
            continue
        got = render(values, e["printed"])
        seeds = e.get("seeds", SEEDS[e["table"]])
        n_ok = len(values) == seeds
        if e.get("exact", True) is False:
            status = "INFO"
        else:
            status = "OK" if got == e["printed"] and n_ok else "DIFFERS"
        counts[status] += 1
        if status != "OK":
            note = f"  -- {e['note']}" if "note" in e and status == "DIFFERS" else ""
            n = f"n = {len(values)}" + ("" if n_ok else f", the table has {seeds}")
            print(f"{status:8s} {where}: paper {e['printed']}, results {got} ({n}){note}")
    print(", ".join(f"{k} {v}" for k, v in counts.items()))
    return 1 if counts["DIFFERS"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
