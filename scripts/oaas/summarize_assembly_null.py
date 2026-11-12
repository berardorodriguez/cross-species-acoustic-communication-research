from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OBSERVED = {"c1": 0.8364381324904796, "c2": 117 / 127}
PRIMARY_SEEDS = set(range(10000, 10500))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the released 500-iteration assembly Null results."
    )
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.results_csv)
    required = {"base_seed", "c1", "c2", "c2_gc_n"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data["base_seed"] = pd.to_numeric(data["base_seed"], errors="raise").astype(int)
    if len(data) != 500 or set(data["base_seed"]) != PRIMARY_SEEDS:
        raise ValueError("Expected exactly the 500 unique planned seeds 10000–10499.")
    if not np.isfinite(data[["c1", "c2", "c2_gc_n"]].to_numpy(dtype=float)).all():
        raise ValueError("Null result columns contain non-finite values.")

    rows = []
    for statistic in ("c1", "c2"):
        values = data[statistic].to_numpy(dtype=float)
        observed = OBSERVED[statistic]
        n_ge = int(np.count_nonzero(values >= observed))
        n_equal = int(np.count_nonzero(np.isclose(values, observed, rtol=0, atol=1e-12)))
        rows.append(
            {
                "statistic": statistic.upper(),
                "observed": observed,
                "null_n": len(values),
                "null_mean": float(np.mean(values)),
                "null_sd_sample": float(np.std(values, ddof=1)),
                "null_median": float(np.median(values)),
                "null_q025": float(np.quantile(values, 0.025)),
                "null_q750": float(np.quantile(values, 0.75)),
                "null_q950": float(np.quantile(values, 0.95)),
                "null_q975": float(np.quantile(values, 0.975)),
                "null_max": float(np.max(values)),
                "n_null_ge_observed": n_ge,
                "n_exact_ties": n_equal,
                "plus_one_upper_tail": (n_ge + 1) / (len(values) + 1),
            }
        )

    summary = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False, float_format="%.17g")
    print(f"Wrote {len(summary)} summary rows to {args.output_csv}")


if __name__ == "__main__":
    main()
