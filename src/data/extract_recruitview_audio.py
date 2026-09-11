"""Extract deterministic vocal-delivery columns from RecruitView MP4 files.

The extractor decodes audio only long enough to calculate aggregate numerical
features.  It does not write waveforms and the resulting columns are the only
audio representation permitted in Model 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


SAMPLE_RATE = 16_000
FRAME_LENGTH = 400  # 25 ms
HOP_LENGTH = 160  # 10 ms
FEATURE_COLUMNS = (
    "audio_duration_seconds",
    "audio_rms_mean",
    "audio_rms_std",
    "audio_zero_crossing_rate",
    "audio_spectral_centroid_mean_hz",
    "audio_spectral_centroid_std_hz",
    "audio_pitch_mean_hz",
    "audio_pitch_std_hz",
    "audio_pause_count",
    "audio_pause_fraction",
    "audio_speaking_time_ratio",
    "audio_words_per_second",
)


class AudioExtractionError(ValueError):
    """Raised when a RecruitView media file cannot be decoded safely."""


def _decode(path: Path) -> np.ndarray:
    command = (
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    )
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = error.stderr.decode("utf-8", errors="replace") if isinstance(error, subprocess.CalledProcessError) else str(error)
        raise AudioExtractionError(f"Could not decode {path}: {detail}") from error
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size < FRAME_LENGTH:
        raise AudioExtractionError(f"Audio stream is too short: {path}")
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def _frames(audio: np.ndarray) -> np.ndarray:
    count = 1 + max(0, (len(audio) - FRAME_LENGTH) // HOP_LENGTH)
    indices = np.arange(FRAME_LENGTH)[None, :] + HOP_LENGTH * np.arange(count)[:, None]
    return audio[np.minimum(indices, len(audio) - 1)]


def _pitches(frames: np.ndarray) -> np.ndarray:
    """Estimate frame pitches with vectorised FFT autocorrelation."""
    centered = frames - np.mean(frames, axis=1, keepdims=True)
    size = 2 * centered.shape[1]
    spectrum = np.fft.rfft(centered, n=size, axis=1)
    correlation = np.fft.irfft(np.abs(spectrum) ** 2, n=size, axis=1)[:, : centered.shape[1]]
    minimum_lag = int(SAMPLE_RATE / 400)
    maximum_lag = min(int(SAMPLE_RATE / 60), correlation.shape[1] - 1)
    lags = np.arange(minimum_lag, maximum_lag + 1)
    best = lags[np.argmax(correlation[:, minimum_lag : maximum_lag + 1], axis=1)]
    confidence = correlation[np.arange(len(frames)), best] / np.maximum(correlation[:, 0], 1e-8)
    energies = np.sum(centered * centered, axis=1)
    result = SAMPLE_RATE / best.astype(float)
    result[(confidence < 0.25) | (energies <= 1e-8)] = 0.0
    return result


def extract_features(path: Path, transcript: str) -> dict[str, float]:
    audio = _decode(path)
    frame_matrix = _frames(audio)
    windowed = frame_matrix * np.hanning(FRAME_LENGTH)[None, :]
    rms = np.sqrt(np.mean(frame_matrix * frame_matrix, axis=1) + 1e-12)
    threshold = max(float(np.percentile(rms, 20)) * 1.5, 1e-4)
    voiced = rms >= threshold
    pauses = np.diff(np.r_[False, voiced, False].astype(np.int8))
    pause_count = int(np.sum(pauses == -1))
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))
    frequencies = np.fft.rfftfreq(FRAME_LENGTH, 1 / SAMPLE_RATE)
    spectral_total = np.maximum(spectrum.sum(axis=1), 1e-8)
    centroid = (spectrum * frequencies[None, :]).sum(axis=1) / spectral_total
    pitches = _pitches(frame_matrix)
    valid_pitch = pitches[pitches > 0]
    duration = len(audio) / SAMPLE_RATE
    words = len(transcript.split())
    return {
        "audio_duration_seconds": float(duration),
        "audio_rms_mean": float(rms.mean()),
        "audio_rms_std": float(rms.std()),
        "audio_zero_crossing_rate": float(np.mean(np.abs(np.diff(np.signbit(frame_matrix), axis=1)))),
        "audio_spectral_centroid_mean_hz": float(centroid.mean()),
        "audio_spectral_centroid_std_hz": float(centroid.std()),
        "audio_pitch_mean_hz": float(valid_pitch.mean()) if valid_pitch.size else 0.0,
        "audio_pitch_std_hz": float(valid_pitch.std()) if valid_pitch.size else 0.0,
        "audio_pause_count": float(pause_count),
        "audio_pause_fraction": float(np.mean(~voiced)),
        "audio_speaking_time_ratio": float(np.mean(voiced)),
        "audio_words_per_second": float(words / duration) if duration else 0.0,
    }


def extract(prepared_path: Path, media_root: Path, output_path: Path, summary_path: Path) -> dict[str, object]:
    with prepared_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"record_id", "media_path", "text_model"}
    if not rows or not required.issubset(rows[0]):
        raise AudioExtractionError(f"Prepared table must contain {sorted(required)}")
    features: list[dict[str, object]] = []
    missing: list[str] = []
    for row in rows:
        path = media_root / row["media_path"]
        if not path.is_file():
            missing.append(row["record_id"])
            continue
        values = extract_features(path, row["text_model"])
        features.append({"record_id": row["record_id"], **values})
    if missing:
        raise AudioExtractionError(
            f"Missing {len(missing)} media files; download the repository media before extraction."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["record_id", *FEATURE_COLUMNS])
        writer.writeheader()
        writer.writerows(features)
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "prepared_path": str(prepared_path),
        "media_root": str(media_root),
        "rows": len(features),
        "missing_media": missing,
        "feature_columns": list(FEATURE_COLUMNS),
        "audio_input_to_model": False,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/processed/recruitview/recruitview_prepared.csv"))
    parser.add_argument("--media-root", type=Path, default=Path("data/raw/recruitview"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/recruitview/recruitview_audio_features.csv"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/recruitview/recruitview_audio_features_summary.json"))
    args = parser.parse_args()
    print(json.dumps(extract(args.prepared, args.media_root, args.output, args.summary), indent=2))


if __name__ == "__main__":
    main()
