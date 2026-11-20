from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


EXPECTED_SPLIT_HASH = "5d0856918ba5ff10c5d46478f326aef7745b76433dd68cb30e538354c38580b0"
EXPECTED_CELLS = {
    ("no_gender", "English"): 1008,
    ("no_gender", "Bangla"): 1406,
    ("gender", "English"): 982,
    ("gender", "Bangla"): 1468,
}


@dataclass(frozen=True)
class CacheData:
    root: Path
    metadata: pd.DataFrame
    faces: np.ndarray
    voices: np.ndarray
    face_norm: np.ndarray
    face_aligned: np.ndarray
    voice_consistency: np.ndarray
    voice_chunks: np.ndarray
    voice_duration: np.ndarray


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_split(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("split_hash") != EXPECTED_SPLIT_HASH:
        raise ValueError("Track 2 refuses an unknown speaker split hash")
    if len(payload.get("groups", [])) != 5:
        raise ValueError("speaker split must contain five groups")
    return payload


def development_speakers(split: dict[str, Any]) -> list[str]:
    return sorted(s for group in split["groups"][1:] for s in group["speakers"])


def fold_speakers(split: dict[str, Any], fold: int) -> tuple[list[str], list[str]]:
    if fold not in range(4):
        raise ValueError("fold must be in [0, 3]")
    groups = split["groups"][1:]
    validation = sorted(groups[fold]["speakers"])
    training = sorted(
        speaker for index, group in enumerate(groups) if index != fold for speaker in group["speakers"]
    )
    if set(training) & set(validation):
        raise ValueError("speaker leakage between train and validation")
    return training, validation


def load_cache(
    root: str | Path,
    mmap_mode: str | None = "r",
    voice_dim: int = 256,
    face_dim: int = 512,
) -> CacheData:
    cache_root = Path(root).expanduser().resolve()
    metadata = pd.read_csv(cache_root / "metadata.csv")
    required = {
        "row_id", "partition", "protocol", "language", "pair_id", "identity", "manifest_order"
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata columns missing: {sorted(missing)}")
    if not np.array_equal(metadata["row_id"].to_numpy(), np.arange(len(metadata))):
        raise ValueError("metadata row_id is not canonical")
    if metadata["pair_id"].duplicated().any():
        raise ValueError("pair IDs must be globally unique")

    def array(name: str, *aliases: str) -> np.ndarray:
        candidates = (name, *aliases)
        path = next((cache_root / item for item in candidates if (cache_root / item).is_file()), None)
        if path is None:
            raise FileNotFoundError(
                f"none of the cache arrays exist: {', '.join(candidates)}"
            )
        if not path.is_file():
            raise FileNotFoundError(path)
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)

    result = CacheData(
        root=cache_root,
        metadata=metadata,
        faces=array("face_embeddings.float32.npy"),
        voices=array("voice_embeddings.float32.npy"),
        face_norm=array("face_feature_norm.float32.npy"),
        face_aligned=array("face_alignment_status.npy"),
        voice_consistency=array(
            "voice_chunk_consistency.float32.npy",
            "voice_window_consistency.float32.npy",
        ),
        voice_chunks=array("voice_chunk_count.int16.npy", "voice_window_count.int16.npy"),
        voice_duration=array("voice_duration_seconds.float32.npy"),
    )
    rows = len(metadata)
    shapes = {
        "faces": (rows, face_dim),
        "voices": (rows, voice_dim),
        "face_norm": (rows,),
        "face_aligned": (rows,),
        "voice_consistency": (rows,),
        "voice_chunks": (rows,),
        "voice_duration": (rows,),
    }
    for name, shape in shapes.items():
        values = getattr(result, name)
        if values.shape != shape:
            raise ValueError(f"{name} shape {values.shape}, expected {shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values")
    for name in ("faces", "voices"):
        values = getattr(result, name)
        error = float(np.max(np.abs(np.linalg.norm(values, axis=1) - 1.0)))
        if error > 2e-4:
            raise ValueError(f"{name} is not L2-normalized: {error}")
    return result


def canonical_train(cache: CacheData) -> pd.DataFrame:
    frame = cache.metadata[cache.metadata["partition"].eq("train")].sort_values("manifest_order")
    if len(frame) != 6485:
        raise ValueError(f"expected 6485 train rows, found {len(frame)}")
    if frame["manifest_order"].tolist() != list(range(6485)):
        raise ValueError("training manifest_order is not contiguous")
    if frame["identity"].nunique() != 70:
        raise ValueError("expected 70 training identities")
    return frame.reset_index(drop=True)


def development_rows(cache: CacheData, split: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    train = canonical_train(cache)
    mask = train["identity"].astype(str).isin(development_speakers(split)).to_numpy()
    return train[mask].reset_index(drop=True), train.loc[mask, "row_id"].to_numpy(dtype=np.int64)


def equal_error_rate(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("EER requires matching binary labels and scores")
    fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
    fnr = 1.0 - tpr
    diff = fpr - fnr
    exact = np.flatnonzero(np.isclose(diff, 0.0, atol=1e-12))
    if exact.size:
        index = int(exact[0])
        return float((fpr[index] + fnr[index]) / 2.0)
    crossing = np.flatnonzero(diff[:-1] * diff[1:] < 0)
    if crossing.size:
        index = int(crossing[0])
        fraction = -diff[index] / (diff[index + 1] - diff[index])
        return float(fpr[index] + fraction * (fpr[index + 1] - fpr[index]))
    index = int(np.argmin(np.abs(diff)))
    return float((fpr[index] + fnr[index]) / 2.0)


def speaker_weights(speakers: np.ndarray) -> np.ndarray:
    values = np.asarray(speakers, dtype=str)
    unique, counts = np.unique(values, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray([1.0 / count_map[value] for value in values], dtype=np.float64)
    return weights / weights.sum()


def verification_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    speakers = frame["voice_identity"].astype(str).to_numpy()
    score_values = np.asarray(scores, dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels, score_values)

    def tar(far: float) -> float:
        eligible = np.flatnonzero(fpr <= far)
        return float(tpr[eligible[-1]]) if eligible.size else 0.0

    result = {
        "eer": equal_error_rate(labels, score_values),
        "speaker_balanced_eer": equal_error_rate(labels, score_values, speaker_weights(speakers)),
        "roc_auc": float(roc_auc_score(labels, score_values)),
        "tar_at_far_0.01": tar(0.01),
        "tar_at_far_0.10": tar(0.10),
    }
    for gender, key in (("m", "male_eer"), ("f", "female_eer")):
        selected = frame["source_gender"].astype(str).to_numpy() == gender
        result[key] = equal_error_rate(labels[selected], score_values[selected])
    result["gender_gap"] = abs(result["male_eer"] - result["female_eer"])
    return result
