"""Recover technical file metadata for the records in Dataset D1.

The source-audio corpus is provided as an external root. Audio is read only;
the script writes a path-free Dataset D2 table plus a separate local audit.
It does not claim to measure effective acoustic bandwidth or SNR.
"""

from __future__ import annotations

import csv
import argparse
import wave
from collections import Counter, defaultdict
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_D1 = PACKAGE_ROOT / "data/metadata/Supplementary_Dataset_D1_Corpus_Metadata.csv"

BITRATES_MPEG1 = {
    1: [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
    2: [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
    3: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
}
BITRATES_MPEG2 = {
    1: [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
    2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    3: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
}


def read_mp3(path: Path) -> dict[str, object]:
    """Read MPEG audio frame headers to recover codec, rate and duration."""
    with path.open("rb") as f:
        head = f.read(10)
        if head[:3] == b"ID3" and len(head) == 10:
            tag_size = ((head[6] & 0x7F) << 21) | ((head[7] & 0x7F) << 14) | ((head[8] & 0x7F) << 7) | (head[9] & 0x7F)
            f.seek(10 + tag_size)
        else:
            f.seek(0)

        def decode(raw: bytes):
            if len(raw) != 4:
                return None
            h = int.from_bytes(raw, "big")
            if (h >> 21) != 0x7FF:
                return None
            version = (h >> 19) & 3
            layer_bits = (h >> 17) & 3
            bitrate_index = (h >> 12) & 15
            rate_index = (h >> 10) & 3
            padding = (h >> 9) & 1
            channel_mode = (h >> 6) & 3
            if version == 1 or layer_bits == 0 or bitrate_index in (0, 15) or rate_index == 3:
                return None
            layer = 4 - layer_bits
            base_rate = [44100, 48000, 32000][rate_index]
            sample_rate = base_rate if version == 3 else (base_rate // 2 if version == 2 else base_rate // 4)
            bitrate_table = BITRATES_MPEG1 if version == 3 else BITRATES_MPEG2
            bitrate = bitrate_table[layer][bitrate_index]
            if layer == 1:
                frame_length = (int(12 * bitrate * 1000 / sample_rate) + padding) * 4
                samples_per_frame = 384
            else:
                factor = 144 if version == 3 or layer == 2 else 72
                frame_length = int(factor * bitrate * 1000 / sample_rate) + padding
                samples_per_frame = 1152 if version == 3 or layer == 2 else 576
            return version, layer, sample_rate, bitrate, frame_length, samples_per_frame, (1 if channel_mode == 3 else 2), h

        file_size = path.stat().st_size
        scan_start = f.tell()
        first_pos = None
        first = None
        scan_limit = min(file_size, scan_start + 1_000_000)
        for pos in range(scan_start, scan_limit - 4):
            f.seek(pos)
            candidate = decode(f.read(4))
            if candidate is None:
                continue
            next_pos = pos + candidate[4]
            f.seek(next_pos)
            following = decode(f.read(4))
            if following and candidate[:3] == following[:3]:
                first_pos, first = pos, candidate
                break

        if first is None or first_pos is None:
            raise ValueError("No consecutive MPEG frames found")

        version, layer, sample_rate, _, frame_length, samples_per_frame, channels, header = first
        crc_bytes = 0 if ((header >> 16) & 1) else 2
        if layer == 3:
            side_info = (17 if channels == 1 else 32) if version == 3 else (9 if channels == 1 else 17)
            tag_pos = first_pos + 4 + crc_bytes + side_info
            f.seek(tag_pos)
            tag = f.read(4)
            if tag in (b"Xing", b"Info"):
                flags_raw = f.read(4)
                flags = int.from_bytes(flags_raw, "big")
                frame_count = int.from_bytes(f.read(4), "big") if flags & 1 else 0
                if frame_count:
                    duration = frame_count * samples_per_frame / sample_rate
                    return {
                        "codec": f"MPEG Layer {layer}",
                        "channels": channels,
                        "sample_rate_hz": sample_rate,
                        "bit_depth": "n/a (lossy codec)",
                        "duration_seconds": round(duration, 3),
                        "average_bitrate_kbps": "",
                        "mpeg_frames_parsed": frame_count,
                    }

        # CBR/VBR files without a Xing/Info frame: count only a contiguous
        # sequence of headers with matching MPEG version, layer, and rate.
        frames = 0
        pos = first_pos
        while pos + 4 <= file_size:
            f.seek(pos)
            current = decode(f.read(4))
            if current is None or current[:3] != first[:3]:
                break
            frames += 1
            pos += current[4]
        if not frames:
            raise ValueError("No MPEG frames counted")
        duration = frames * samples_per_frame / sample_rate

    if not sample_rate:
        raise ValueError("No MPEG frames parsed")
    return {
        "codec": f"MPEG Layer {layer}",
        "channels": channels,
        "sample_rate_hz": sample_rate,
        "bit_depth": "n/a (lossy codec)",
        "duration_seconds": round(duration, 3),
        "average_bitrate_kbps": "",
        "mpeg_frames_parsed": frames,
    }


def technical_info(path: Path) -> dict[str, object]:
    info: dict[str, object] = {
        "source_extension": path.suffix.lower(),
        "source_size_bytes": path.stat().st_size,
    }
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            info.update(
                codec="PCM" if audio.getcomptype() == "NONE" else audio.getcomptype(),
                channels=audio.getnchannels(),
                sample_rate_hz=rate,
                bit_depth=audio.getsampwidth() * 8,
                duration_seconds=round(audio.getnframes() / rate, 3),
                average_bitrate_kbps="",
                mpeg_frames_parsed="",
            )
    elif path.suffix.lower() == ".mp3":
        info.update(read_mp3(path))
    else:
        raise ValueError(f"Unsupported source extension: {path.suffix}")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate path-free technical audio metadata for Dataset D2."
    )
    parser.add_argument("--corpus-root", type=Path, required=True,
                        help="External audio-corpus root containing D1 relative paths.")
    parser.add_argument("--d1", type=Path, default=DEFAULT_D1,
                        help="Dataset D1 metadata CSV (defaults to the packaged copy).")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Directory for the D2 table and separate local audit outputs.")
    args = parser.parse_args()
    corpus_root = args.corpus_root.resolve()
    d1_path = args.d1.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "D1_technical_metadata_audit.csv"
    out_md = outdir / "D1_technical_metadata_audit.md"
    out_d2 = outdir / "Supplementary_Dataset_D2_Technical_Audio_Metadata.csv"

    with d1_path.open(encoding="utf-8-sig", newline="") as f:
        d1_rows = list(csv.DictReader(f))
    required_path_fields = {"id", "domain", "subdomain", "relative_path_original", "relative_path_processed"}
    if d1_rows and not required_path_fields.issubset(d1_rows[0]):
        raise ValueError(f"Dataset D1 is missing path fields: {sorted(required_path_fields - set(d1_rows[0]))}")

    output = []
    missing = []
    errors = []
    for row in d1_rows:
        source = corpus_root / row["relative_path_original"]
        analysis = corpus_root / row["relative_path_processed"]
        selected_path = source if source.exists() else analysis
        if not selected_path.exists():
            missing.append((row["id"], row["relative_path_original"]))
            continue
        try:
            info = technical_info(selected_path)
            output.append(
                {
                    "id": row["id"],
                    "domain": row["domain"],
                    "subdomain": row["subdomain"],
                    "metadata_path_used": str(selected_path.relative_to(corpus_root)).replace("\\", "/"),
                    "metadata_basis": "original source" if selected_path == source else "analysis-ready fallback",
                    **info,
                    "nominal_nyquist_hz": round(float(info["sample_rate_hz"]) / 2, 1),
                }
            )
        except Exception as exc:  # retain the failure explicitly in the audit
            errors.append((row["id"], str(selected_path), type(exc).__name__, str(exc)))

    fields = [
        "id", "domain", "subdomain", "metadata_path_used", "metadata_basis",
        "source_extension", "codec", "channels", "sample_rate_hz", "nominal_nyquist_hz",
        "bit_depth", "duration_seconds", "average_bitrate_kbps", "mpeg_frames_parsed",
        "source_size_bytes",
    ]
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)

    release_fields = [
        "id", "domain", "subdomain", "source_extension", "codec", "channels",
        "sample_rate_hz", "nominal_nyquist_hz", "bit_depth", "duration_seconds",
        "source_size_bytes",
    ]
    with out_d2.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=release_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in release_fields} for row in output)

    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in output:
        by_domain[str(row["domain"])].append(row)
    ext_counts = Counter(str(r["source_extension"]) for r in output)
    rates = Counter(str(r["sample_rate_hz"]) for r in output)
    depths = Counter(str(r["bit_depth"]) for r in output)
    codecs = Counter(str(r["codec"]) for r in output)
    channel_counts = Counter(str(r["channels"]) for r in output)
    basis_counts = Counter(str(r["metadata_basis"]) for r in output)
    durations = sorted(float(r["duration_seconds"]) for r in output)
    median_duration = durations[len(durations) // 2] if durations else float("nan")

    lines = [
        "# Dataset D1 technical audio metadata audit",
        "",
        "**Date:** 2026-10-06  ",
        f"**Coverage:** {len(output)} of {len(d1_rows)} Dataset D1 records parsed; {len(missing)} missing; {len(errors)} parsing failures.",
        "",
        "## Recoverable fields",
        "",
        f"File extensions: `{dict(ext_counts)}`. Codecs: `{dict(codecs)}`. Channel counts: `{dict(channel_counts)}`. Sample rates (Hz): `{dict(rates)}`. Bit depths: `{dict(depths)}`.",
        f"Metadata was parsed from original-source files for `{basis_counts.get('original source', 0)}` records; an analysis-ready fallback was needed for `{basis_counts.get('analysis-ready fallback', 0)}` records.",
        f"Duration range: {min(durations):.3f}–{max(durations):.3f} s; median: {median_duration:.3f} s.",
        "The row-level audit is in `D1_technical_metadata_audit.csv`; the release-ready, path-free table keyed to Dataset D1 is `Supplementary_Dataset_D2_Technical_Audio_Metadata.csv`.",
        "",
        "## Interpretation limits",
        "",
        "The nominal Nyquist value (half the file sample rate) is a digital upper limit, not a measured recording or biological bandwidth. File codec, sample rate, channel count, bit depth, duration, and size are recoverable from file headers. These do not establish the original recorder, microphone, compression history before the retained file, signal-to-noise ratio, or performer identity.",
        "",
        "No SNR value was inferred. The author's qualitative call-screening rule is the complete acoustic envelope with no audible contaminating background noise; calculating a quantitative SNR would require a defensible signal/noise segmentation and cannot be substituted for that recorded decision. Performer/recordist identity can only be added where an independent source record explicitly identifies it.",
        "",
        "The source corpus is external and may include third-party audio. This metadata audit does not authorize redistribution of the audio; rights status remains a separate release check.",
    ]
    if missing:
        lines.extend(["", "## Missing records", "", *[f"- `{a}`: {b}" for a, b in missing[:50]]])
    if errors:
        lines.extend(["", "## Parsing failures", "", *[f"- `{a}` (`{c}`): {d}" for a, _, c, d in errors[:50]]])
    lines.extend(["", "## Counts by domain", ""])
    for domain, values in sorted(by_domain.items()):
        lines.append(f"- `{domain}`: {len(values)} records")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Parsed {len(output)}/{len(d1_rows)} Dataset D1 files; missing={len(missing)}; errors={len(errors)}")
    print(f"Extensions={dict(ext_counts)}; sample rates={dict(rates)}; bit depths={dict(depths)}")
    print(f"Duration min/median/max={min(durations):.3f}/{median_duration:.3f}/{max(durations):.3f} s")
    print(f"Wrote {out_csv}, {out_md}, and {out_d2}")


if __name__ == "__main__":
    main()
