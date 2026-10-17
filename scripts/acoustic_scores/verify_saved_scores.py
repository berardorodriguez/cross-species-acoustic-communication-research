"""Verify retained O1 A_adj scores from saved descriptors and reference-IQR tables.

This check does not load or process audio. It applies the archived scoring
function to the already-derived rows in data/derived/acoustic_scores.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCORE_IMPL = Path(__file__).with_name("score_saved_descriptors.py")
SCORE_TABLE = ROOT / "data" / "derived" / "acoustic_scores" / "scores_o1_parts.csv"
REFS = ROOT / "data" / "derived" / "acoustic_scores" / "reference_iqr"


def load_score_implementation():
    spec = importlib.util.spec_from_file_location("a_adj_score_impl", SCORE_IMPL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load archived scorer: {SCORE_IMPL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    import pandas as pd

    scorer = load_score_implementation()
    local = scorer.load_iqr_csv(str(REFS / "pastoreo_IQR.csv"))
    lf = scorer.load_iqr_csv(str(REFS / "bovinetalk_LF_reference_IQR.csv"))
    hf = scorer.load_iqr_csv(str(REFS / "bovinetalk_HF_reference_IQR.csv"))
    if not local or not lf or not hf:
        raise RuntimeError("One or more archived reference-IQR tables are empty or missing.")

    table = pd.read_csv(SCORE_TABLE)
    required = set(scorer.DIRECTIONS) | {
        "V_base", "A_base", "V_adj", "A_adj", "reward_lf", "penalty_hf"
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise RuntimeError(f"Retained score table is missing fields: {missing}")

    observed = {key: 0.0 for key in ("V_base", "A_base", "V_adj", "A_adj", "reward_lf", "penalty_hf")}
    for _, row in table.iterrows():
        metrics = {
            key: (float(row[key]) if pd.notna(row[key]) else None)
            for key in scorer.DIRECTIONS
        }
        v_base, a_base, v_adj, a_adj, _, reward_lf, penalty_hf = scorer.compute_scores(
            metrics, local, lf, hf
        )
        calculated = {
            "V_base": v_base,
            "A_base": a_base,
            "V_adj": v_adj,
            "A_adj": a_adj,
            "reward_lf": reward_lf,
            "penalty_hf": penalty_hf,
        }
        for key, value in calculated.items():
            observed[key] = max(observed[key], abs(float(row[key]) - float(value)))

    maximum = max(observed.values())
    print(f"Rows checked: {len(table)}")
    print(f"Reference metrics: local={len(local)}, LF={len(lf)}, HF={len(hf)}")
    print(f"Maximum absolute residual: {maximum:.3g}")
    print(f"Tolerance: {args.tolerance:.3g}")
    if maximum > args.tolerance:
        print(f"Residuals by score: {observed}")
        return 1
    print("PASS: saved O1 scores reproduce from the retained descriptors and reference tables; no audio was processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
