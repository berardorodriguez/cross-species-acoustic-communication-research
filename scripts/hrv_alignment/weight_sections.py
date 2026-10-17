#!/usr/bin/env python
"""
 O1 SECTION-AWARE CORE TABLE

Purpose
-------
Create the conservative, section-aware analytical table for the submitted O1
physiology using only the operationally aligned, fully supported records.

Inputs
------
1) Corrected O1 section table.
2) Acoustic-to-Kubios alignment table.
3) Individual O1 physiology table.

Core rule
---------
Retain only records classified as CORE_SUPPORT_MATCHED in the alignment table.
Expected geometry:
    7 cows x 18 records = 126 repeated observations
    record indices 3..20

Section-aware representation
----------------------------
For every retained 300-s Kubios support interval, compute exact temporal
overlap with each of the 13 unequal-duration O1 sections:

    section_01_fraction ... section_13_fraction

For core windows:
    sum(section fractions) == 1.0

Also report:
- seconds contributed by each section
- number of sections contributing
- dominant section
- dominant section fraction
- exact weighted acoustic predictors derived from section-level values when
  available in the section source table

IMPORTANT
---------
Section durations are NOT assumed equal.
Section identity is NOT treated as independent biological replication.
No physiological outcome is used to define the core or section composition.

NO statistical inference.
NO HRV-based predictor selection.
NO acoustic-HRV correlation.
NO lag testing.
NO p-values.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower().replace("_", " ").replace("-", " "))


def find_col(df, candidates, required=True):
    lookup = {norm(c): c for c in df.columns}
    for c in candidates:
        if norm(c) in lookup:
            return lookup[norm(c)]
    if required:
        raise KeyError(f"Could not find any of {candidates}. Available: {list(df.columns)}")
    return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", required=True)
    ap.add_argument("--alignment", required=True)
    ap.add_argument("--o1-physiology", required=True)
    ap.add_argument("--output-dir", default=".")
    args = ap.parse_args()

    p_sections = Path(args.sections).resolve()
    p_alignment = Path(args.alignment).resolve()
    p_o1 = Path(args.o1_physiology).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(" O1 SECTION-AWARE CORE TABLE")
    print("Sections :", p_sections)
    print("Alignment:", p_alignment)
    print("O1 physiology :", p_o1)
    print("NO INFERENCE / NO HRV-BASED SELECTION / NO COUPLING / NO LAGS / NO P-VALUES")
    print("=" * 96)

    sec = pd.read_csv(p_sections)
    aln = pd.read_csv(p_alignment)
    hrv = pd.read_csv(p_o1)

    # Section geometry.
    s_id = find_col(sec, ["section", "section_id", "section_number", "section_numeric"])
    s_start = find_col(sec, ["section_start_s", "start_s", "cumulative_start_s"])
    s_end = find_col(sec, ["section_end_s", "end_s", "cumulative_end_s"])

    sec["_section_id"] = pd.to_numeric(sec[s_id], errors="coerce").astype(int)
    sec["_start_s"] = pd.to_numeric(sec[s_start], errors="coerce")
    sec["_end_s"] = pd.to_numeric(sec[s_end], errors="coerce")
    sec["_duration_s"] = sec["_end_s"] - sec["_start_s"]

    if sorted(sec["_section_id"].tolist()) != list(range(1, 14)):
        raise RuntimeError(f"Expected sections 1..13; got {sorted(sec['_section_id'].tolist())}")

    if (sec["_duration_s"] <= 0).any():
        raise RuntimeError("One or more section durations are non-positive.")

    # Confirm continuity/order.
    sec = sec.sort_values("_section_id").reset_index(drop=True)
    gap_errors = []
    for i in range(len(sec)-1):
        gap = float(sec.loc[i+1, "_start_s"] - sec.loc[i, "_end_s"])
        if abs(gap) > 1e-6:
            gap_errors.append((i+1, i+2, gap))
    if gap_errors:
        raise RuntimeError(f"Section timeline contains gaps/overlaps: {gap_errors[:10]}")

    # Alignment geometry.
    a_cow = find_col(aln, ["cow"])
    a_idx = find_col(aln, ["o1_record_index_0based"])
    a_ps = find_col(aln, ["phys_support_start_rel_s"])
    a_pe = find_col(aln, ["phys_support_end_rel_s"])
    a_class = find_col(aln, ["support_class"])

    core = aln[aln[a_class].astype(str) == "CORE_SUPPORT_MATCHED"].copy()
    core["cow"] = pd.to_numeric(core[a_cow], errors="coerce").astype(int)
    core["o1_record_index_0based"] = pd.to_numeric(core[a_idx], errors="coerce").astype(int)
    core["phys_support_start_rel_s"] = pd.to_numeric(core[a_ps], errors="coerce")
    core["phys_support_end_rel_s"] = pd.to_numeric(core[a_pe], errors="coerce")

    counts = core.groupby("cow").size()
    if len(counts) != 7 or not (counts == 18).all():
        raise RuntimeError(f"Expected 18 core records for 7 cows. Got {counts.to_dict()}")

    if sorted(core["o1_record_index_0based"].unique().tolist()) != list(range(3, 21)):
        raise RuntimeError(
            f"Expected core indices 3..20; got {sorted(core['o1_record_index_0based'].unique())}"
        )

    # HRV keys + all physiological fields retained verbatim.
    h_cow = find_col(hrv, ["cow"])
    h_idx = find_col(hrv, ["o1_record_index_0based", "record_index", "o1_relative_record_min"])

    H = hrv.copy()
    H["cow"] = pd.to_numeric(H[h_cow], errors="coerce").astype(int)
    H["o1_record_index_0based"] = pd.to_numeric(H[h_idx], errors="coerce").astype(int)

    # Unique key check.
    if H.duplicated(["cow", "o1_record_index_0based"]).any():
        raise RuntimeError("Frozen O1 physiology contains duplicate cow x record keys.")

    analytical = core.merge(
        H,
        on=["cow", "o1_record_index_0based"],
        how="left",
        validate="one_to_one",
        suffixes=("_align", "_hrv"),
        indicator=True,
    )
    if not (analytical["_merge"] == "both").all():
        raise RuntimeError("Some core records failed to match frozen O1 physiology.")
    analytical = analytical.drop(columns=["_merge"])

    # Identify section-level acoustic descriptors that can be time-weighted.
    geometry_exclude = {
        s_id, s_start, s_end, "_section_id", "_start_s", "_end_s", "_duration_s",
        "file", "filename", "section_file", "section_name", "wav", "path"
    }
    descriptor_cols = []
    for c in sec.columns:
        if c in geometry_exclude:
            continue
        vals = pd.to_numeric(sec[c], errors="coerce")
        if vals.notna().all() and vals.nunique(dropna=True) > 1:
            descriptor_cols.append(c)

    # Build section composition per unique record geometry (same across cows).
    composition_rows = []
    unique_geom = (
        core[["o1_record_index_0based", "phys_support_start_rel_s", "phys_support_end_rel_s"]]
        .drop_duplicates()
        .sort_values("o1_record_index_0based")
    )

    for _, r in unique_geom.iterrows():
        k = int(r["o1_record_index_0based"])
        ws = float(r["phys_support_start_rel_s"])
        we = float(r["phys_support_end_rel_s"])
        duration = we - ws
        if abs(duration - 300.0) > 1e-9:
            raise RuntimeError(f"Core record {k} support is not 300 s: {duration}")

        rec = {
            "o1_record_index_0based": k,
            "phys_support_start_rel_s": ws,
            "phys_support_end_rel_s": we,
        }

        sec_seconds = []
        for _, sr in sec.iterrows():
            sid = int(sr["_section_id"])
            ov = overlap(ws, we, float(sr["_start_s"]), float(sr["_end_s"]))
            sec_seconds.append((sid, ov))
            rec[f"section_{sid:02d}_seconds"] = ov
            rec[f"section_{sid:02d}_fraction"] = ov / 300.0

        total = sum(v for _, v in sec_seconds)
        rec["section_overlap_total_s"] = total
        rec["section_fraction_sum"] = total / 300.0
        rec["n_sections_contributing"] = sum(v > 0 for _, v in sec_seconds)

        dom_sid, dom_sec = max(sec_seconds, key=lambda x: x[1])
        rec["dominant_section"] = dom_sid
        rec["dominant_section_seconds"] = dom_sec
        rec["dominant_section_fraction"] = dom_sec / 300.0

        # Exact time-weighted section-level descriptors.
        for d in descriptor_cols:
            num = 0.0
            den = 0.0
            for _, sr in sec.iterrows():
                sid = int(sr["_section_id"])
                ov = dict(sec_seconds)[sid]
                if ov <= 0:
                    continue
                v = pd.to_numeric(pd.Series([sr[d]]), errors="coerce").iloc[0]
                if pd.notna(v):
                    num += float(v) * ov
                    den += ov
            rec[f"{d}__section_timeweighted"] = num / den if den > 0 else np.nan

        composition_rows.append(rec)

    comp = pd.DataFrame(composition_rows)

    # Strong conservation requirement for core windows.
    max_abs_fraction_error = float(np.max(np.abs(comp["section_fraction_sum"] - 1.0)))
    max_abs_seconds_error = float(np.max(np.abs(comp["section_overlap_total_s"] - 300.0)))
    if max_abs_fraction_error > 1e-9 or max_abs_seconds_error > 1e-6:
        raise RuntimeError(
            f"Section overlap conservation failed. frac={max_abs_fraction_error}, sec={max_abs_seconds_error}"
        )

    # Add composition to analytical table.
    analytical = analytical.merge(
        comp.drop(columns=["phys_support_start_rel_s", "phys_support_end_rel_s"]),
        on="o1_record_index_0based",
        how="left",
        validate="many_to_one",
    )

    # Sort the retained analytical table.
    analytical = analytical.sort_values(["cow", "o1_record_index_0based"]).reset_index(drop=True)

    # Section duration summary.
    section_summary = sec[["_section_id", "_start_s", "_end_s", "_duration_s"]].copy()
    section_summary.columns = ["section", "start_s", "end_s", "duration_s"]
    section_summary["duration_fraction_of_o1"] = (
        section_summary["duration_s"] / section_summary["duration_s"].sum()
    )

    verdict = {
        "analysis": "o1_section_aware_core",
        "status": "section_aware_core_table_created",
        "n_cows": int(analytical["cow"].nunique()),
        "biological_n": int(analytical["cow"].nunique()),
        "records_per_cow": int(analytical.groupby("cow").size().iloc[0]),
        "repeated_observations_total": int(len(analytical)),
        "core_record_indices": sorted(analytical["o1_record_index_0based"].unique().tolist()),
        "n_o1_sections": int(len(sec)),
        "section_durations_equal": bool(np.allclose(sec["_duration_s"], sec["_duration_s"].iloc[0])),
        "section_duration_min_s": float(sec["_duration_s"].min()),
        "section_duration_max_s": float(sec["_duration_s"].max()),
        "section_duration_sum_s": float(sec["_duration_s"].sum()),
        "max_abs_section_overlap_conservation_error_s": max_abs_seconds_error,
        "max_abs_section_fraction_sum_error": max_abs_fraction_error,
        "section_weighting_rule": "exact seconds of section overlap within each 300-s Kubios support window",
        "section_identity_used_as_independent_replication": False,
        "hrv_values_used_to_define_core": False,
        "physiological_inference_performed": False,
        "acoustic_hrv_coupling_computed": False,
        "lags_tested": False,
        "p_values_computed": False,
        "sections_sha256": sha256_file(p_sections),
        "alignment_sha256": sha256_file(p_alignment),
        "o1_physiology_sha256": sha256_file(p_o1),
    }

    stem = "o1_section_weighted_core"

    analytical.to_csv(out / f"{stem}.csv", index=False)
    comp.to_csv(out / f"{stem}_section_composition_by_record.csv", index=False)
    section_summary.to_csv(out / f"{stem}_section_duration_summary.csv", index=False)

    with pd.ExcelWriter(out / f"{stem}.xlsx", engine="openpyxl") as w:
        pd.DataFrame([verdict]).to_excel(w, sheet_name="VERDICT", index=False)
        section_summary.to_excel(w, sheet_name="SECTION_DURATIONS", index=False)
        comp.to_excel(w, sheet_name="RECORD_COMPOSITION", index=False)
        analytical.to_excel(w, sheet_name="CORE_ANALYTICAL", index=False)

    with open(out / f"{stem}_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(native(verdict), fh, indent=2, ensure_ascii=False)

    print("\nSECTION DURATION SUMMARY")
    print(section_summary.to_string(index=False))

    print("\nCORE RECORD SECTION COMPOSITION")
    show = comp[[
        "o1_record_index_0based",
        "phys_support_start_rel_s",
        "phys_support_end_rel_s",
        "n_sections_contributing",
        "dominant_section",
        "dominant_section_fraction",
        "section_fraction_sum",
    ]]
    print(show.to_string(index=False))

    print("\nFREEZE VERDICT")
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    print("\nINTERPRETIVE BOUNDARY")
    print("  O1 sections are unequal in duration and are represented by exact overlap time.")
    print("  Section identity is a compositional/time-structure variable, not biological replication.")
    print("  Biological N is 7 cows; 126 rows are repeated observations nested within cows.")
    print("  No physiological outcome was used to select windows, sections, or acoustic descriptors.")

    print("\nCreated outputs with stem:", stem)


if __name__ == "__main__":
    main()
