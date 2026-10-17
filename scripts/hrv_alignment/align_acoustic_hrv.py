#!/usr/bin/env python
"""
 Acoustic-to-Kubios Time-Varying alignment geometry

Purpose
-------
Define a conservative, auditable correspondence between:
1) the corrected O1 acoustic timeline, and
2) the O1 Kubios Time-Varying physiological records,

WITHOUT using HRV magnitudes, correlations, lags, or p-values.

Operational phase anchor
------------------------
The recorded O1 block boundary is treated as the operational acoustic
t=0 for the submitted O1 analysis:
    acoustic O1 t=0 <-> center of O1 physiological record 0

This is an operational provenance rule supported by the original workbook
boundary (VC Experimento / recovered O1 / formatting transition). It is NOT
presented as an independently instrument-triggered acoustic timestamp.

Kubios geometry
---------------
Window width = 300 s
Grid interval = 60 s
Thus physiological record k has support:
    [k*60 - 150, k*60 + 150] s
relative to the operational O1 onset.

Primary conservative subset
---------------------------
A physiological record is CORE_SUPPORT_MATCHED only if its entire 300-s Kubios
support lies within the corrected O1 acoustic master duration.

For every record, the script computes:
- physiological support start/end;
- exact seconds and fraction of support covered by O1 audio;
- boundary class (PRE_PARTIAL / CORE_SUPPORT_MATCHED / POST_PARTIAL);
- time-weighted acoustic predictors over the overlapping acoustic support.

NO HRV values are used.
NO physiological inference.
NO coupling.
NO lags.
NO p-values.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def native(x):
    if isinstance(x, dict):
        return {str(k): native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [native(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def find_col(df, candidates, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for x in candidates:
        key = str(x).strip().lower()
        if key in lookup:
            return lookup[key]
    if required:
        raise KeyError(f"Could not find {candidates}; columns={list(df.columns)}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acoustic-windows", required=True,
                    help="Corrected O1 acoustic windows CSV (include the terminal partial window if available).")
    ap.add_argument("--o1-physiology", required=True,
                    help="O1 individual-level Kubios Time-Varying CSV.")
    ap.add_argument("--output-dir", default=".")
    ap.add_argument("--window-width-s", type=float, default=300.0)
    ap.add_argument("--grid-s", type=float, default=60.0)
    args = ap.parse_args()

    p_ac = Path(args.acoustic_windows).resolve()
    p_o1 = Path(args.o1_physiology).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print(" ACOUSTIC-to-KUBIOS TIME-VARYING ALIGNMENT GEOMETRY")
    print("Acoustic windows:", p_ac)
    print("O1 physiology   :", p_o1)
    print("NO HRV MAGNITUDE / NO COUPLING / NO LAGS / NO P-VALUES / NO PHYSIOLOGICAL INFERENCE")
    print("=" * 92)

    ac = pd.read_csv(p_ac)
    o1 = pd.read_csv(p_o1)

    c_ws = find_col(ac, ["window_start_s"])
    c_we = find_col(ac, ["window_end_s"])
    c_dur = find_col(ac, ["actual_duration_s"], required=False)

    cow_col = find_col(o1, ["cow", "vaca", "animal"])
    rec_col = find_col(o1, ["o1_record_index_0based", "record_index", "o1_relative_record_min"])

    # Acoustic master duration from exact terminal edge.
    acoustic_duration_s = float(pd.to_numeric(ac[c_we], errors="coerce").max())

    # Expected O1 geometry.
    cows = sorted(pd.to_numeric(o1[cow_col], errors="coerce").dropna().astype(int).unique().tolist())
    counts = o1.groupby(cow_col).size()
    if len(cows) != 7 or not (counts == 23).all():
        raise RuntimeError(f"Unexpected O1 record geometry: cows={cows}, counts={counts.to_dict()}")

    # Candidate acoustic descriptor columns. Exclude geometry/IDs/flags.
    exclude_tokens = {
        "window_index_0based", "window_start_s", "window_end_s", "actual_duration_s",
        "is_full_60s_window", "n_sections_contributing", "dominant_section",
        "dominant_section_fraction"
    }
    descriptors = []
    for c in ac.columns:
        if c in exclude_tokens:
            continue
        x = pd.to_numeric(ac[c], errors="coerce")
        # Only keep numeric columns with at least one nonmissing value and non-ID-ish naming.
        if x.notna().any():
            descriptors.append(c)

    # Build record geometry once because all cows share the same operational O1 index.
    half = args.window_width_s / 2.0
    record_rows = []

    for k in range(23):
        center = k * args.grid_s
        ps = center - half
        pe = center + half
        covered = overlap(ps, pe, 0.0, acoustic_duration_s)
        frac = covered / args.window_width_s

        if ps >= 0.0 and pe <= acoustic_duration_s:
            cls = "CORE_SUPPORT_MATCHED"
        elif ps < 0.0 and pe > 0.0:
            cls = "PRE_PARTIAL"
        elif ps < acoustic_duration_s and pe > acoustic_duration_s:
            cls = "POST_PARTIAL"
        elif pe <= 0.0:
            cls = "PRE_OUTSIDE"
        elif ps >= acoustic_duration_s:
            cls = "POST_OUTSIDE"
        else:
            cls = "PARTIAL_OTHER"

        rec = {
            "o1_record_index_0based": k,
            "phys_center_rel_s": center,
            "phys_support_start_rel_s": ps,
            "phys_support_end_rel_s": pe,
            "acoustic_overlap_s": covered,
            "acoustic_overlap_fraction_of_300s": frac,
            "support_class": cls,
        }

        # Time-weighted acoustic descriptors over overlap with O1.
        for d in descriptors:
            vals = pd.to_numeric(ac[d], errors="coerce")
            num = 0.0
            den = 0.0
            for i, row in ac.iterrows():
                a0 = float(row[c_ws])
                a1 = float(row[c_we])
                ov = overlap(ps, pe, a0, a1)
                if ov <= 0:
                    continue
                v = vals.iloc[i]
                if pd.notna(v):
                    num += float(v) * ov
                    den += ov
            rec[f"{d}__support_weighted"] = (num / den) if den > 0 else np.nan

        record_rows.append(rec)

    geom = pd.DataFrame(record_rows)

    # Expand by cow without touching HRV values.
    long = (
        pd.MultiIndex.from_product([cows, geom["o1_record_index_0based"].tolist()],
                                   names=["cow", "o1_record_index_0based"])
        .to_frame(index=False)
        .merge(geom, on="o1_record_index_0based", how="left", validate="many_to_one")
    )

    # Cross-check that each cow has exactly the same record indices in the physiology table.
    source_keys = (
        o1[[cow_col, rec_col]].copy()
        .rename(columns={cow_col: "cow", rec_col: "o1_record_index_0based"})
    )
    source_keys["cow"] = pd.to_numeric(source_keys["cow"], errors="coerce").astype(int)
    source_keys["o1_record_index_0based"] = pd.to_numeric(
        source_keys["o1_record_index_0based"], errors="coerce"
    ).astype(int)

    chk = long.merge(source_keys, on=["cow", "o1_record_index_0based"], how="outer", indicator=True)
    if not (chk["_merge"] == "both").all():
        raise RuntimeError("O1 physiology record keys do not match the expected 7 x 23 alignment map.")

    counts_class = (
        geom["support_class"].value_counts()
        .rename_axis("support_class")
        .reset_index(name="n_records_per_cow")
    )

    core = geom[geom["support_class"] == "CORE_SUPPORT_MATCHED"].copy()
    if core.empty:
        raise RuntimeError("No full 300-s support windows fall completely within O1.")

    verdict = {
        "analysis": "acoustic_hrv_alignment_geometry",
        "status": "alignment_geometry_defined",
        "operational_anchor_rule": "corrected acoustic O1 t=0 aligned to the center of the first O1 physiological record / recorded experimental-block start",
        "anchor_is_instrument_triggered_timestamp": False,
        "phase_boundary_support": "VC Experimento start, recorded O1 start, and workbook formatting transition",
        "n_cows": len(cows),
        "records_per_cow": 23,
        "window_width_s": args.window_width_s,
        "grid_s": args.grid_s,
        "window_overlap_s": args.window_width_s - args.grid_s,
        "window_overlap_fraction": (args.window_width_s - args.grid_s) / args.window_width_s,
        "corrected_o1_acoustic_duration_s": acoustic_duration_s,
        "core_support_matched_records_per_cow": int(len(core)),
        "core_first_record_index": int(core["o1_record_index_0based"].min()),
        "core_last_record_index": int(core["o1_record_index_0based"].max()),
        "boundary_records_excluded_from_core": int(23 - len(core)),
        "hrv_values_used": False,
        "field_time_reconstruction_used": False,
        "acoustic_hrv_coupling_computed": False,
        "lags_tested": False,
        "p_values_computed": False,
        "physiological_inference_performed": False,
        "acoustic_windows_sha256": sha256_file(p_ac),
        "o1_physiology_sha256": sha256_file(p_o1),
    }

    stem = "acoustic_hrv_alignment_geometry"
    geom.to_csv(out / f"{stem}_record_geometry.csv", index=False)
    long.to_csv(out / f"{stem}_cow_record_map.csv", index=False)
    counts_class.to_csv(out / f"{stem}_class_counts.csv", index=False)

    with pd.ExcelWriter(out / f"{stem}.xlsx", engine="openpyxl") as w:
        pd.DataFrame([verdict]).to_excel(w, sheet_name="VERDICT", index=False)
        counts_class.to_excel(w, sheet_name="CLASS_COUNTS", index=False)
        geom.to_excel(w, sheet_name="RECORD_GEOMETRY", index=False)
        long.to_excel(w, sheet_name="COW_RECORD_MAP", index=False)

    with open(out / f"{stem}_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(native(verdict), fh, indent=2, ensure_ascii=False)

    print("\nSUPPORT CLASS COUNTS PER COW")
    print(counts_class.to_string(index=False))

    print("\nRECORD GEOMETRY")
    show_cols = [
        "o1_record_index_0based", "phys_center_rel_s",
        "phys_support_start_rel_s", "phys_support_end_rel_s",
        "acoustic_overlap_s", "acoustic_overlap_fraction_of_300s",
        "support_class"
    ]
    print(geom[show_cols].to_string(index=False))

    print("\nALIGNMENT VERDICT")
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    print("\nINTERPRETIVE BOUNDARY")
    print("  The operational t=0 rule preserves the recorded O1 phase boundary.")
    print("  It is not claimed to be an independently instrument-triggered acoustic timestamp.")
    print("  CORE_SUPPORT_MATCHED excludes boundary HRV estimates whose 300-s support extends")
    print("  before O1 onset or after the corrected O1 acoustic master ends.")
    print("  No HRV values or physiological outcomes were used to define this subset.")

    print("\nCreated outputs with stem:", stem)


if __name__ == "__main__":
    main()
