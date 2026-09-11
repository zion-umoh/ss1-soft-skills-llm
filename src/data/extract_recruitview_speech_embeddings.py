"""Extract resumable pretrained speech embeddings from RecruitView audio.

The waveform is decoded only long enough to create a fixed-size embedding with
the pretrained WavLM encoder.  Raw audio is never written by this script.  A
record is represented by the mean of non-overlapping five-second chunk
embeddings, which keeps memory bounded for the long interview answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

try:
    from src.data.extract_recruitview_audio import SAMPLE_RATE
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution
    from extract_recruitview_audio import SAMPLE_RATE


DEFAULT_MODEL = "microsoft/wavlm-base-plus"
CHUNK_SECONDS = 5
warnings.filterwarnings("ignore", message="Support for mismatched key_padding_mask and attn_mask is deprecated")


class SpeechEmbeddingError(ValueError):
    """Raised when speech embedding extraction cannot satisfy its contract."""


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
        raise SpeechEmbeddingError(f"Could not decode {path}: {detail}") from error
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size < SAMPLE_RATE // 4:
        raise SpeechEmbeddingError(f"Audio stream is too short: {path}")
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _load_checkpoint(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    completed: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                record_id = str(row["record_id"])
                vector = [float(value) for value in row["embedding"]]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise SpeechEmbeddingError(f"Invalid checkpoint row {line_number} in {path}") from error
            completed[record_id] = vector
    return completed


def _load_model(model_name: str, device: str):
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
    except ImportError as error:
        raise SpeechEmbeddingError("Speech embeddings require torch and transformers.") from error
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return extractor, model, torch


def _embed_audio(audio: np.ndarray, extractor, model, torch, device: str, batch_size: int) -> np.ndarray:
    chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
    # Ignore sub-second tail fragments; they are not a meaningful speech
    # window and can be shorter than WavLM's convolutional receptive field.
    chunks = [audio[start : start + chunk_samples] for start in range(0, len(audio), chunk_samples) if len(audio[start : start + chunk_samples]) >= SAMPLE_RATE]
    if not chunks:
        chunks = [audio]
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            inputs = extractor(batch, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            hidden = model(**inputs).last_hidden_state
            mask = inputs.get("attention_mask")
            if mask is None:
                pooled = hidden.mean(dim=1)
            else:
                # WavLM's hidden sequence is downsampled relative to samples.
                hidden_mask = torch.nn.functional.interpolate(
                    mask[:, None, :].float(), size=hidden.shape[1], mode="nearest"
                ).squeeze(1)
                pooled = (hidden * hidden_mask[:, :, None]).sum(dim=1) / hidden_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            vectors.extend(vector.detach().cpu().numpy() for vector in pooled)
    result = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    norm = float(np.linalg.norm(result))
    return result / max(norm, 1e-8)


def extract(
    prepared_path: Path,
    media_root: Path,
    output_path: Path,
    checkpoint_path: Path,
    summary_path: Path,
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    batch_size: int = 4,
    limit: int | None = None,
) -> dict[str, object]:
    with prepared_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"record_id", "media_path"}
    if not rows or not required.issubset(rows[0]):
        raise SpeechEmbeddingError(f"Prepared table must contain {sorted(required)}")
    if limit is not None:
        rows = rows[:limit]
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor, model, torch = _load_model(model_name, device)
    completed = _load_checkpoint(checkpoint_path)
    failures: list[dict[str, str]] = []
    embedding_dim: int | None = len(next(iter(completed.values()))) if completed else None
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_stream = checkpoint_path.open("a", encoding="utf-8")
    try:
        for index, row in enumerate(rows, start=1):
            record_id = row["record_id"]
            if record_id in completed:
                continue
            path = media_root / row["media_path"]
            try:
                vector = _embed_audio(_decode(path), extractor, model, torch, device, batch_size)
                if embedding_dim is None:
                    embedding_dim = int(vector.shape[0])
                if vector.shape != (embedding_dim,):
                    raise SpeechEmbeddingError(f"Unexpected embedding shape {vector.shape}; expected {(embedding_dim,)}")
                payload = {"record_id": record_id, "embedding": [float(value) for value in vector]}
                checkpoint_stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
                checkpoint_stream.flush()
                os.fsync(checkpoint_stream.fileno())
                completed[record_id] = payload["embedding"]
            except (OSError, SpeechEmbeddingError, subprocess.CalledProcessError, RuntimeError) as error:
                failures.append({"record_id": record_id, "error": str(error)})
            if index % 25 == 0 or index == len(rows):
                print(json.dumps({"processed": index, "completed": len(completed), "failures": len(failures)}), flush=True)
    finally:
        checkpoint_stream.close()
    if not completed:
        raise SpeechEmbeddingError("No speech embeddings were extracted.")
    if embedding_dim is None:
        raise SpeechEmbeddingError("Could not determine speech embedding dimension.")
    fields = ["record_id", *[f"speech_embedding_{index:03d}" for index in range(embedding_dim)]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            vector = completed.get(row["record_id"])
            if vector is None:
                continue
            writer.writerow({"record_id": row["record_id"], **{fields[index + 1]: value for index, value in enumerate(vector)}})
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "prepared_path": str(prepared_path),
        "media_root": str(media_root),
        "model_name": model_name,
        "device": device,
        "chunk_seconds": CHUNK_SECONDS,
        "rows_requested": len(rows),
        "rows_completed": sum(row["record_id"] in completed for row in rows),
        "failures_this_run": failures,
        "embedding_dimension": embedding_dim,
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "raw_audio_input_to_predictor": False,
    }
    _atomic_write(summary_path, json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/processed/recruitview/recruitview_prepared.csv"))
    parser.add_argument("--media-root", type=Path, default=Path("data/raw/recruitview"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/recruitview/recruitview_speech_embeddings.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/processed/recruitview/recruitview_speech_embeddings.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("data/processed/recruitview/recruitview_speech_embeddings_summary.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(extract(args.prepared, args.media_root, args.output, args.checkpoint, args.summary, args.model, args.device, args.batch_size, args.limit), indent=2))


if __name__ == "__main__":
    main()
