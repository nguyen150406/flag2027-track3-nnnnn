#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from track2.core import (
    canonical_train,
    fold_speakers,
    load_cache,
    read_split,
    sha256_file,
    stable_hash,
    verification_metrics,
    write_json,
)
from track2.model import (
    age_matched_same_gender_hard_negative_loss,
    batch_identity_prototype_alignment_loss,
    fixed_simplex_center_alignment_loss,
    DualLinearProjector,
    GenderAdversary,
    SharedAAMClassifier,
    cross_modal_supervised_contrastive_loss,
    covariance_decorrelation_loss,
    curriculum_cross_modal_negative_loss,
    gender_weighted_cross_modal_contrastive_loss,
    intra_modal_supervised_contrastive_loss,
    paired_alignment_loss,
    regular_simplex,
    same_gender_cross_modal_contrastive_loss,
    same_gender_hard_negative_loss,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_only_gender_direction(
    features: np.ndarray,
    speakers: np.ndarray,
    training_speakers: list[str],
    gender_map: dict[str, str],
) -> np.ndarray:
    """Speaker-balanced male-minus-female direction from fold training only."""
    means = {
        speaker: features[speakers == speaker].mean(axis=0)
        for speaker in training_speakers
    }
    male = np.stack([means[s] for s in training_speakers if gender_map[s] == "m"]).mean(axis=0)
    female = np.stack([means[s] for s in training_speakers if gender_map[s] == "f"]).mean(axis=0)
    direction = np.asarray(male - female, dtype=np.float32)
    return direction / np.linalg.norm(direction).clip(min=1e-8)


def attenuate_direction(features: np.ndarray, direction: np.ndarray, scale: float) -> np.ndarray:
    projected = features - np.float32(scale) * (features @ direction)[:, None] * direction[None, :]
    return np.ascontiguousarray(
        projected / np.linalg.norm(projected, axis=1, keepdims=True).clip(min=1e-8),
        dtype=np.float32,
    )


class PKSampler:
    def __init__(
        self,
        speakers: np.ndarray,
        gender_map: dict[str, str],
        male: int,
        female: int,
        samples: int,
        steps: int,
        seed: int,
    ) -> None:
        self.indices = {
            speaker: np.flatnonzero(speakers == speaker) for speaker in np.unique(speakers)
        }
        self.male = sorted([speaker for speaker in self.indices if gender_map[speaker] == "m"])
        self.female = sorted([speaker for speaker in self.indices if gender_map[speaker] == "f"])
        if len(self.male) < male or len(self.female) < female:
            raise ValueError("training fold cannot satisfy balanced P32K2 quota")
        self.male_count = male
        self.female_count = female
        self.samples = samples
        self.steps = steps
        self.seed = seed

    def epoch(self, number: int):
        rng = np.random.default_rng(self.seed + 1009 * number)
        for _ in range(self.steps):
            identities = np.concatenate(
                [
                    rng.choice(self.male, size=self.male_count, replace=False),
                    rng.choice(self.female, size=self.female_count, replace=False),
                ]
            )
            rng.shuffle(identities)
            batch = []
            for speaker in identities.tolist():
                available = self.indices[str(speaker)]
                batch.extend(
                    rng.choice(available, size=self.samples, replace=len(available) < self.samples).tolist()
                )
            yield np.asarray(batch, dtype=np.int64)


def sample_other_face_view(
    batch: np.ndarray,
    speakers: np.ndarray,
    identity_rows: dict[str, np.ndarray],
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Swap only the face view with another fold-training row of the same identity."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("face identity view swap probability must be in [0, 1]")
    face_rows = batch.copy()
    for index, row in enumerate(batch):
        if rng.random() >= probability:
            continue
        candidates = identity_rows[str(speakers[row])]
        if len(candidates) < 2:
            continue
        alternative = int(rng.choice(candidates))
        if alternative == row:
            position = int(np.flatnonzero(candidates == row)[0])
            alternative = int(candidates[(position + 1) % len(candidates)])
        face_rows[index] = alternative
    if not np.array_equal(speakers[face_rows], speakers[batch]):
        raise RuntimeError("cross-identity face view swap is forbidden")
    return face_rows


def sample_other_voice_view(
    batch: np.ndarray,
    speakers: np.ndarray,
    identity_rows: dict[str, np.ndarray],
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Choose another fold-training utterance of the same identity."""
    # The identity-safe row selector is modality-agnostic. Keep separate names
    # so callers cannot accidentally apply the face and voice probabilities
    # to the same input view.
    return sample_other_face_view(batch, speakers, identity_rows, probability, rng)


def replace_mav_face_encoder(source: dict, version: str, encoder: str, face_dim: int) -> dict:
    """Explicit source encoder, never inferred from vector dimension (AdaFace/FaRL=512)."""
    if version not in ("v1", "v3") or encoder not in ("adaface", "clip"):
        raise ValueError("unsupported MAV version/face encoder override")
    expected_dim = 512 if encoder == "adaface" else 768
    if face_dim != expected_dim:
        raise ValueError("requested face dimension differs from explicit encoder")
    # Validate the original downloaded archive as well as the live imported
    # array, not merely shape-compatible files from a different extractor.
    from import_t2_mav_complete_faces import validate_archive
    anchor = (ROOT.parent / "t2_mav_v1_english_farl_wavlm_l4/face_anchors.csv"
              if version == "v1" else ROOT / "outputs/t2_v3_face_voice_benchmark/face_anchors.csv")
    split = ROOT.parent / f"external_mavceleb/splits/{version}/English_train.txt"
    archive = ROOT / f"kaggle_mav_complete_faces/download/t2_mav_{version}_english_adaface_clip.zip"
    _, provenance = validate_archive(archive, version, anchor, split)
    cache = ROOT / f"outputs/t2_mav_complete_faces/{version}"
    if (cache / "face_anchors.csv").read_bytes() != anchor.read_bytes():
        raise ValueError("new encoder anchor order differs from original FaRL source")
    filename = f"{encoder}_faces.float32.npy"
    path = cache / filename
    digest = sha256_file(path)
    if digest != provenance["output_sha256"][filename]:
        raise ValueError("imported MAV face feature checksum changed")
    face = np.load(path, allow_pickle=False)
    if (face.shape != (len(source["face"]), face_dim) or face.dtype != np.float32
            or not np.isfinite(face).all()
            or not np.allclose(np.linalg.norm(face, axis=1), 1, atol=2e-4, rtol=0)):
        raise ValueError("invalid explicit source face embeddings")
    return {**source, "face": face, "face_sha256": digest, "face_encoder": encoder,
            "face_provenance_sha256": sha256_file(cache / "PROVENANCE.json"),
            "split_sha256": sha256_file(split)}


def load_v3_english_auxiliary(face_dim: int, voice_dim: int, *, face_encoder: str = "farl") -> dict:
    """Load only official MAV-Celeb v3 train English pairs for auxiliary fitting."""
    if face_encoder != "farl":
        return replace_mav_face_encoder(load_v3_english_auxiliary(512, voice_dim), "v3", face_encoder, face_dim)
    source_root = ROOT.parent / "external_mavceleb/lang01_cache"
    face_root = ROOT / "outputs/t2_v3_face_voice_benchmark"
    if face_dim != 512 or voice_dim not in (256, 768, 1536):
        raise ValueError("v3 English auxiliary supports FaRL with SimAM or WavLM-L4 only")
    voice_name = "simam.float32.npy" if voice_dim == 256 else "wavlm_l4_mean.float32.npy"
    voice_meta = pd.read_csv(source_root / "metadata.csv")
    face_meta = pd.read_csv(face_root / "face_anchors.csv")
    if set(voice_meta.language) != {"English", "German"}:
        raise ValueError("unexpected v3 source language manifest")
    if not face_meta.face_relpath.str.contains("/English/", regex=False).all():
        raise ValueError("source face anchor is not English")
    english_mask = voice_meta.language.eq("English").to_numpy()
    if not voice_meta.loc[english_mask, "relpath"].str.contains("/English/", regex=False).all():
        raise ValueError("source voice row is not English")
    if english_mask.sum() != 1422 or len(face_meta) != 379:
        raise ValueError("unexpected v3 English train row count")
    official_pairs = [line.strip().split() for line in
                      (ROOT.parent / "external_mavceleb/splits/v3/English_train.txt").open()]
    if any(len(pair) != 2 for pair in official_pairs):
        raise ValueError("malformed official v3 English train list")
    official_voice = {pair[0] for pair in official_pairs}
    official_face = {pair[1] for pair in official_pairs}
    if set(voice_meta.loc[english_mask, "relpath"]) != official_voice:
        raise ValueError("source voice cache does not match official v3 English train list")
    if not set(face_meta.face_relpath).issubset(official_face):
        raise ValueError("source face anchors are outside official v3 English train list")
    face = np.asarray(np.load(face_root / "farl_faces.float32.npy", allow_pickle=False), dtype=np.float32)
    if voice_dim == 1536:
        pool_root = ROOT / "outputs/t2_xp21_v3_meanstd"
        pool_meta = pd.read_csv(pool_root / "metadata.csv")
        if not pool_meta.equals(voice_meta.loc[english_mask].reset_index(drop=True)):
            raise ValueError("v3 mean+std source ordering differs")
        if not np.load(pool_root / "completed.npy").all():
            raise ValueError("v3 mean+std extraction incomplete")
        voice_path = pool_root / "wavlm_l4_meanstd.float32.npy"
        voice = np.asarray(np.load(voice_path, allow_pickle=False), dtype=np.float32)
    else:
        voice_path = source_root / voice_name
        voice_all = np.load(voice_path, mmap_mode="r", allow_pickle=False)
        voice = np.asarray(voice_all[english_mask], dtype=np.float32)
    if face.shape != (379, 512) or voice.shape != (1422, voice_dim):
        raise ValueError("unexpected v3 English feature shape")
    if not np.isfinite(face).all() or not np.isfinite(voice).all():
        raise ValueError("nonfinite source features")
    face_speakers = face_meta.identity.astype(str).to_numpy()
    voice_speakers = voice_meta.loc[english_mask, "identity"].astype(str).to_numpy()
    speaker_ids = sorted(set(face_speakers) & set(voice_speakers))
    if len(speaker_ids) != 50 or set(face_speakers) != set(voice_speakers):
        raise ValueError("v3 English source speakers are incomplete")
    return {
        "face": face,
        "voice": voice,
        "speaker_ids": speaker_ids,
        "face_rows": {s: np.flatnonzero(face_speakers == s) for s in speaker_ids},
        "voice_rows": {s: np.flatnonzero(voice_speakers == s) for s in speaker_ids},
        "face_sha256": sha256_file(face_root / "farl_faces.float32.npy"),
        "voice_sha256": sha256_file(voice_path),
        "face_video": face_meta.face_relpath.str.split("/").str[2].to_numpy(),
        "voice_video": voice_meta.loc[english_mask, "relpath"].str.split("/").str[2].to_numpy(),
    }


def load_v1_english_auxiliary(face_dim: int, voice_dim: int, *, face_encoder: str = "farl") -> dict:
    """Load only audited MAV-Celeb v1 English-train rows and frozen features."""
    if face_encoder != "farl":
        return replace_mav_face_encoder(load_v1_english_auxiliary(512, voice_dim), "v1", face_encoder, face_dim)
    if face_dim != 512 or voice_dim not in (256, 768, 1536):
        raise ValueError("v1 English auxiliary supports FaRL + SimAM or WavLM L4 mean/meanstd only")
    source_root = ROOT.parent / "t2_mav_v1_english_farl_wavlm_l4"
    official = ROOT.parent / "external_mavceleb/splits/v1/English_train.txt"
    provenance = json.loads((source_root / "PROVENANCE.json").read_text())
    if (provenance.get("split") != "MAV-Celeb v1 English_train.txt only"
            or provenance.get("test_data_used") is not False
            or provenance.get("urdu_data_used") is not False
            or provenance.get("split_sha256") != sha256_file(official)):
        raise ValueError("v1 source provenance does not match the official English train split")
    pairs = pd.read_csv(source_root / "v1_english_train_pairs.csv")
    faces_meta = pd.read_csv(source_root / "face_anchors.csv")
    official_pairs = [line.split() for line in official.read_text().splitlines()]
    if (len(pairs) != 5786 or pairs.identity.nunique() != 64
            or official_pairs != pairs[["voice_relpath", "face_relpath"]].astype(str).values.tolist()
            or pairs.pair_order.to_numpy(int).tolist() != list(range(len(pairs)))):
        raise ValueError("v1 source pairs do not match the official English train list")
    if (len(faces_meta) != 512 or faces_meta.identity.nunique() != 64
            or not set(faces_meta.face_relpath).issubset(set(pairs.face_relpath))
            or not faces_meta.face_relpath.str.contains("/English/", regex=False).all()):
        raise ValueError("v1 English face anchors are not valid")
    voice_path = source_root / "wavlm_l4_meanstd.float32.npy"
    if voice_dim == 256:
        sim_root = ROOT / "outputs/t2_xp22_v1_simam_cache"
        sim_provenance = json.loads((sim_root / "PROVENANCE.json").read_text())
        sim_pairs = pd.read_csv(sim_root / "v1_english_train_pairs.csv")
        if (sim_provenance.get("split_sha256") != sha256_file(official)
                or sim_provenance.get("test_data_used") is not False
                or sim_provenance.get("urdu_data_used") is not False
                or sim_provenance.get("model_weight_sha256") != "f345a9b0ac0a4e0527ea7fc7e3815c438aab001a957d81a681955430a168cef5"
                or sim_provenance.get("model_config_sha256") != "6c6592dfeb54bd77eb9bf516344f955212223c8afc9c3b309141713a8b8cb167"
                or sim_pairs[["voice_relpath", "face_relpath"]].values.tolist() != official_pairs):
            raise ValueError("v1 SimAM model/split/row provenance mismatch")
        voice_path = sim_root / "simam.float32.npy"
    full_voice = np.load(voice_path, mmap_mode="r", allow_pickle=False)
    face = np.asarray(np.load(source_root / "farl_faces.float32.npy", allow_pickle=False), dtype=np.float32)
    if full_voice.shape != (5786, 256 if voice_dim == 256 else 1536) or face.shape != (512, 512):
        raise ValueError("v1 source feature shapes differ from audited output")
    # Kaggle stored L2([mean, std]). Taking and re-normalizing the first half
    # recovers exactly the 768D L2(mean) used by XP06, up to float rounding.
    voice = np.array(full_voice[:, :voice_dim], dtype=np.float32, order="C", copy=True)
    voice /= np.linalg.norm(voice, axis=1, keepdims=True).clip(min=1e-8)
    if not np.isfinite(face).all() or not np.isfinite(voice).all():
        raise ValueError("v1 source has nonfinite features")
    face_speakers = faces_meta.identity.astype(str).to_numpy()
    voice_speakers = pairs.identity.astype(str).to_numpy()
    speaker_ids = sorted(set(face_speakers) & set(voice_speakers))
    if len(speaker_ids) != 64 or set(face_speakers) != set(voice_speakers):
        raise ValueError("v1 source speaker coverage is incomplete")
    return {
        "face": face,
        "voice": voice,
        "speaker_ids": speaker_ids,
        "face_rows": {s: np.flatnonzero(face_speakers == s) for s in speaker_ids},
        "voice_rows": {s: np.flatnonzero(voice_speakers == s) for s in speaker_ids},
        "face_sha256": sha256_file(source_root / "farl_faces.float32.npy"),
        "voice_sha256": sha256_file(voice_path),
        "split_sha256": provenance["split_sha256"],
        "face_video": faces_meta.face_relpath.str.split("/").str[2].to_numpy(),
        "voice_video": pairs.voice_relpath.str.split("/").str[2].to_numpy(),
    }


def sample_v3_english_batch(source: dict, rng: np.random.Generator, identities: int = 16,
                            samples: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample independent English face/voice views from the same source identity."""
    chosen = rng.choice(source["speaker_ids"], size=identities, replace=False)
    face_rows, voice_rows, labels = [], [], []
    class_map = {s: i for i, s in enumerate(source["speaker_ids"])}
    def choose(rows, modality):
        if not source.get("video_balanced", False):
            return int(rng.choice(rows))
        videos = source[f"{modality}_video"][rows]
        video = rng.choice(np.unique(videos))
        return int(rng.choice(rows[videos == video]))
    for speaker in chosen:
        for _ in range(samples):
            face_rows.append(choose(source["face_rows"][speaker], "face"))
            voice_rows.append(choose(source["voice_rows"][speaker], "voice"))
            labels.append(class_map[speaker])
    return np.asarray(face_rows), np.asarray(voice_rows), np.asarray(labels)


@torch.inference_mode()
def embed_all(model: DualLinearProjector, faces: np.ndarray, voices: np.ndarray, device: torch.device):
    model.eval()
    face_parts, voice_parts = [], []
    for start in range(0, len(faces), 512):
        stop = min(len(faces), start + 512)
        face, voice = model(
            torch.from_numpy(np.asarray(faces[start:stop])).to(device),
            torch.from_numpy(np.asarray(voices[start:stop])).to(device),
        )
        face_parts.append(face.cpu().numpy())
        voice_parts.append(voice.cpu().numpy())
    return np.concatenate(face_parts), np.concatenate(voice_parts)


def evaluate(
    model,
    faces,
    voices,
    row_map: np.ndarray,
    trials_root: Path,
    fold: int,
    device: torch.device,
    selection_metric: str = "speaker_balanced_eer",
):
    face_embeddings, voice_embeddings = embed_all(model, faces, voices, device)
    metrics, predictions = {}, []
    for protocol in ("standard", "same_gender"):
        frame = pd.read_csv(trials_root / f"fold_{fold}" / f"{protocol}.csv")
        face_rows = row_map[frame["face_row"].to_numpy(dtype=np.int64)]
        voice_rows = row_map[frame["voice_row"].to_numpy(dtype=np.int64)]
        if np.any(face_rows < 0) or np.any(voice_rows < 0):
            raise RuntimeError("validation trial referenced locked Group 0")
        scores = np.sum(
            face_embeddings[face_rows]
            * voice_embeddings[voice_rows],
            axis=1,
        )
        metrics[protocol] = verification_metrics(frame, scores)
        output = frame.copy()
        output["protocol"] = protocol
        output["score"] = scores
        predictions.append(output)
    if selection_metric not in ("eer", "speaker_balanced_eer"):
        raise ValueError("unknown checkpoint selection metric")
    selection = float(np.mean([metrics[name][selection_metric] for name in ("standard", "same_gender")]))
    metrics["selection_metric"] = selection_metric
    metrics["selection_score"] = selection
    return metrics, pd.concat(predictions, ignore_index=True)


def train_fold(args, config: dict, fold: int) -> dict:
    set_seed(int(config["experiment"]["seed"]))
    device = torch.device(config["training"].get("device", "cpu"))
    cache = load_cache(
        args.cache_root,
        voice_dim=int(config["data"]["voice_dim"]),
        face_dim=int(config["data"]["face_dim"]),
    )
    split = read_split(args.split)
    train_frame = canonical_train(cache)
    development = sorted(s for group in split["groups"][1:] for s in group["speakers"])
    dev_mask = train_frame["identity"].astype(str).isin(development).to_numpy()
    dev_frame = train_frame[dev_mask].reset_index(drop=True)
    row_ids = dev_frame["row_id"].to_numpy(dtype=np.int64)
    # Read only Groups 1–4. Group 0 embeddings remain unopened.
    faces = np.ascontiguousarray(cache.faces[row_ids], dtype=np.float32)
    voices = np.ascontiguousarray(cache.voices[row_ids], dtype=np.float32)
    speakers = dev_frame["identity"].astype(str).to_numpy()
    row_map = np.full(6485, -1, dtype=np.int64)
    row_map[row_ids] = np.arange(len(row_ids), dtype=np.int64)
    training_speakers, validation_speakers = fold_speakers(split, fold)
    if args.outer_fold is not None:
        if args.outer_fold == fold:
            raise ValueError("inner validation fold must differ from outer held-out fold")
        _, outer_validation = fold_speakers(split, args.outer_fold)
        training_speakers = sorted(set(training_speakers) - set(outer_validation))
        if len(training_speakers) != 28 or set(training_speakers) & set(outer_validation):
            raise ValueError("nested fit must use exactly 28 identities outside outer fold")
    selected = np.flatnonzero(np.isin(speakers, training_speakers))
    if set(speakers[selected]) & set(validation_speakers):
        raise RuntimeError("held-out identity leakage")
    face_center = None
    if config["data"].get("center_face_on_train", False):
        # One common FaRL direction, estimated without held-out speakers.
        # Speaker-balanced means prevent a frequent speaker dominating it.
        speaker_means = np.stack([
            faces[selected][speakers[selected] == speaker].mean(axis=0)
            for speaker in training_speakers
        ])
        face_center = speaker_means.mean(axis=0).astype(np.float32)
        faces = np.ascontiguousarray(faces - face_center)
        faces /= np.linalg.norm(faces, axis=1, keepdims=True).clip(min=1e-8)
    voice_center = None
    if config["data"].get("center_voice_on_train", False):
        voice_speaker_means = np.stack([
            voices[selected][speakers[selected] == speaker].mean(axis=0)
            for speaker in training_speakers
        ])
        voice_center = voice_speaker_means.mean(axis=0).astype(np.float32)
        voices = np.ascontiguousarray(voices - voice_center)
        voices /= np.linalg.norm(voices, axis=1, keepdims=True).clip(min=1e-8)
    gender_directions = None
    direction_scale = float(config["data"].get("gender_direction_attenuation", 0.0))
    if direction_scale > 0:
        if not 0 < direction_scale <= 1:
            raise ValueError("gender_direction_attenuation must be in (0, 1]")
        gender_map = split["gender_by_speaker"]
        face_direction = train_only_gender_direction(
            faces, speakers, training_speakers, gender_map
        )
        voice_direction = train_only_gender_direction(
            voices, speakers, training_speakers, gender_map
        )
        faces = attenuate_direction(faces, face_direction, direction_scale)
        voices = attenuate_direction(voices, voice_direction, direction_scale)
        gender_directions = {"face": face_direction, "voice": voice_direction}
    class_map = {speaker: index for index, speaker in enumerate(training_speakers)}
    labels = np.asarray([class_map.get(speaker, -1) for speaker in speakers], dtype=np.int64)
    simplex_weight = float(config["loss"].get("fixed_simplex_center_weight", 0.0))
    simplex_centers = None
    if simplex_weight > 0:
        simplex_centers = regular_simplex(
            classes=len(class_map), dimensions=int(config["data"]["embedding_dim"])
        ).to(device)
    model = DualLinearProjector(
        face_dim=config["data"]["face_dim"],
        voice_dim=config["data"]["voice_dim"],
        embedding_dim=config["data"]["embedding_dim"],
        face_dropout=config["model"]["face_dropout"],
        voice_dropout=config["model"]["voice_dropout"],
        voice_block_dropout=config["model"].get("voice_block_dropout", 0.0),
        voice_block_dims=tuple(config["model"].get("voice_block_dims", ())),
        input_layernorm=bool(config["model"].get("input_layernorm", True)),
        projector_type=config["model"].get("projector_type", "linear"),
        hidden_dim=int(config["model"].get("hidden_dim", 384)),
        hidden_dropout=float(config["model"].get("hidden_dropout", 0.20)),
    ).to(device)
    init_projector_sha256 = None
    if args.init_projector is not None and args.resume_run_root is not None:
        raise ValueError("external-only initialization and fold-checkpoint continuation are exclusive")
    if args.init_projector is not None:
        source = torch.load(args.init_projector, map_location="cpu", weights_only=False)
        permitted_scopes = {
            "external_mavceleb_v3_train_speakers_only": 768,
            "external_mavceleb_v3_simam_train_speakers_only": 256,
        }
        if source.get("training_scope") not in permitted_scopes:
            raise ValueError("projector initialization must come from audited external v3 pretrain")
        if source.get("voice_dim") != permitted_scopes[source["training_scope"]]:
            raise ValueError("external projector voice dimension does not match its source scope")
        if source.get("face_dim") != config["data"]["face_dim"] or source.get("voice_dim") != config["data"]["voice_dim"]:
            raise ValueError("external projector dimensions do not match cache")
        model.load_state_dict(source["model"])
        init_projector_sha256 = sha256_file(args.init_projector)
    classifier = SharedAAMClassifier(
        config["data"]["embedding_dim"],
        len(class_map),
        config["loss"]["aam_margin"],
        config["loss"]["aam_scale"],
    ).to(device)
    same_gender_aam_weight = float(config["loss"].get("same_gender_aam_weight", 0.0))
    if not 0.0 <= same_gender_aam_weight <= 1.0:
        raise ValueError("same_gender_aam_weight must be in [0, 1]")
    classifier_genders = torch.tensor(
        [int(split["gender_by_speaker"][s] == "f") for s in training_speakers], device=device,
    ) if same_gender_aam_weight else None
    resume_checkpoint = None
    resume_sha256 = None
    if args.resume_run_root is not None:
        source_path = (args.resume_run_root / f"seed_{config['experiment']['seed']}" /
                       f"fold_{fold}" / "best.pt")
        resume_checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        source_config = resume_checkpoint.get("config", {})
        if (source_config.get("experiment", {}).get("name") !=
                "t2_xp08_v1_v3_english_wavlm_v1w010"):
            raise ValueError("continuation must initialize from audited XP08 branch")
        if (int(source_config["data"]["face_dim"]) != int(config["data"]["face_dim"])
                or int(source_config["data"]["voice_dim"]) != int(config["data"]["voice_dim"])
                or int(source_config["data"]["embedding_dim"]) != int(config["data"]["embedding_dim"])):
            raise ValueError("continuation model dimensions differ")
        if resume_checkpoint.get("class_map") != class_map:
            raise ValueError("continuation classifier identities differ from fold training")
        model.load_state_dict(resume_checkpoint["model"])
        classifier.load_state_dict(resume_checkpoint["classifier"])
        resume_sha256 = sha256_file(source_path)
    external_weight = float(config["training"].get("v3_english_auxiliary_weight", 0.0))
    if not 0.0 <= external_weight <= 1.0:
        raise ValueError("v3_english_auxiliary_weight must be in [0, 1]")
    external_source = None
    external_classifier = None
    if external_weight:
        external_source = load_v3_english_auxiliary(
            int(config["data"]["face_dim"]), int(config["data"]["voice_dim"]),
            face_encoder=config["data"].get("face_encoder", "farl"))
        external_source["video_balanced"] = config["training"].get("source_video_balanced", False)
        # Keep incumbent model/dropout RNG stream unchanged by the additional head.
        with torch.random.fork_rng():
            torch.manual_seed(int(config["experiment"]["seed"]) + 85001)
            external_classifier = SharedAAMClassifier(
                config["data"]["embedding_dim"], len(external_source["speaker_ids"]),
                config["loss"]["aam_margin"], config["loss"]["aam_scale"],
            ).to(device)
    v1_weight = float(config["training"].get("v1_english_auxiliary_weight", 0.0))
    if not 0.0 <= v1_weight <= 1.0:
        raise ValueError("v1_english_auxiliary_weight must be in [0, 1]")
    v1_source = None
    v1_classifier = None
    if v1_weight:
        v1_source = load_v1_english_auxiliary(
            int(config["data"]["face_dim"]), int(config["data"]["voice_dim"]),
            face_encoder=config["data"].get("face_encoder", "farl"))
        v1_source["video_balanced"] = config["training"].get("source_video_balanced", False)
        # v1 and v3 have reused idXXXX tokens, so they require independent
        # classifier heads. The 192D projectors alone are shared.
        with torch.random.fork_rng():
            torch.manual_seed(int(config["experiment"]["seed"]) + 86001)
            v1_classifier = SharedAAMClassifier(
                config["data"]["embedding_dim"], len(v1_source["speaker_ids"]),
                config["loss"]["aam_margin"], config["loss"]["aam_scale"],
            ).to(device)
    adversarial_weight = float(config["loss"].get("gender_adversarial_weight", 0.0))
    gender_adversary = None
    if adversarial_weight > 0:
        # Keep the projector/classifier/dropout random stream paired with IMG13.
        with torch.random.fork_rng():
            torch.manual_seed(int(config["experiment"]["seed"]) + 75001)
            gender_adversary = GenderAdversary(config["data"]["embedding_dim"]).to(device)
    trainable = list(model.parameters()) + list(classifier.parameters())
    if external_classifier is not None:
        trainable += list(external_classifier.parameters())
    if v1_classifier is not None:
        trainable += list(v1_classifier.parameters())
    if gender_adversary is not None:
        trainable += list(gender_adversary.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"], eta_min=1e-6
    )
    batch_size = config["training"]["batch_identities"] * config["training"]["samples_per_identity"]
    sampler = PKSampler(
        speakers[selected],
        split["gender_by_speaker"],
        config["training"]["male_identities"],
        config["training"]["female_identities"],
        config["training"]["samples_per_identity"],
        math.ceil(len(selected) / batch_size),
        config["experiment"]["seed"],
    )
    local_faces, local_voices = faces[selected], voices[selected]
    local_labels = labels[selected]
    local_speakers = speakers[selected]
    face_view_swap_probability = float(config["training"].get("face_identity_view_swap_prob", 0.0))
    if not 0.0 <= face_view_swap_probability <= 1.0:
        raise ValueError("face_identity_view_swap_prob must be in [0, 1]")
    voice_view_swap_probability = float(config["training"].get("voice_identity_view_swap_prob", 0.0))
    if not 0.0 <= voice_view_swap_probability <= 1.0:
        raise ValueError("voice_identity_view_swap_prob must be in [0, 1]")
    identity_rows = (
        {str(speaker): np.flatnonzero(local_speakers == speaker)
         for speaker in np.unique(local_speakers)}
        if face_view_swap_probability or voice_view_swap_probability else {}
    )
    local_genders = np.asarray(
        [0 if split["gender_by_speaker"][str(speaker)] == "m" else 1
         for speaker in speakers[selected]], dtype=np.int64
    )
    age_hard_weight = float(config["loss"].get("age_matched_hard_negative_weight", 0.0))
    curriculum_weight = float(config["loss"].get("curriculum_hard_negative_weight", 0.0))
    local_face_ages = local_voice_ages = None
    if age_hard_weight > 0 or curriculum_weight > 0:
        age_root = Path(config["data"].get(
            "age_cache_root", ROOT.parent / "Track 1/age_compatibility_experiments/cache"
        ))
        face_age_cache = np.load(age_root / "train_face_age.npy", mmap_mode="r", allow_pickle=False)
        voice_age_cache = np.load(age_root / "train_voice_age.npy", mmap_mode="r", allow_pickle=False)
        if face_age_cache.shape != voice_age_cache.shape or face_age_cache.shape != (6485,):
            raise ValueError("noncanonical training age cache")
        # Only fold-training age rows are read; held-out and locked ages are
        # never used to choose the hard negatives or fit their age offset.
        training_row_ids = row_ids[selected]
        local_face_ages = np.asarray(face_age_cache[training_row_ids], dtype=np.float32)
        local_voice_ages = np.asarray(voice_age_cache[training_row_ids], dtype=np.float32)
        if not np.isfinite(local_face_ages).all() or not np.isfinite(local_voice_ages).all():
            raise ValueError("nonfinite fold-training age cache")
        if np.any((local_face_ages < 0) | (local_face_ages > 1) |
                  (local_voice_ages < 0) | (local_voice_ages > 1)):
            raise ValueError("age cache outside [0, 1]")
        speaker_offsets = [
            np.median(local_face_ages[speakers[selected] == speaker])
            - np.median(local_voice_ages[speakers[selected] == speaker])
            for speaker in training_speakers
        ]
        offset = float(np.median(speaker_offsets))
        local_voice_ages = np.clip(local_voice_ages + offset, 0, 1)
    run = args.output_root / config["experiment"]["name"] / f"seed_{config['experiment']['seed']}" / f"fold_{fold}"
    run.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    best_epoch = 0
    patience = 0
    history = []
    if resume_checkpoint is not None:
        initial_metrics, initial_predictions = evaluate(
            model, faces, voices, row_map, args.trials_root, fold, device)
        best_score = float(initial_metrics["selection_score"])
        best_epoch = 0
        if abs(best_score - float(resume_checkpoint["selection_score"])) > 1e-8:
            raise ValueError("continuation checkpoint score differs from reevaluation")
        torch.save({"model": model.state_dict(), "classifier": classifier.state_dict(),
                    "epoch": 0, "selection_score": best_score,
                    "class_map": class_map, "face_center": face_center,
                    "voice_center": voice_center, "gender_directions": gender_directions,
                    "config": config, "resume_checkpoint_sha256": resume_sha256}, run / "best.pt")
        initial_predictions.to_csv(run / "predictions.csv", index=False)
        write_json(run / "initial_metrics.json", initial_metrics)
    started = time.perf_counter()
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        classifier.train()
        if external_classifier is not None:
            external_classifier.train()
        if v1_classifier is not None:
            v1_classifier.train()
        if gender_adversary is not None:
            gender_adversary.train()
        losses = []
        for batch_number, batch in enumerate(sampler.epoch(epoch)):
            face_rows = (
                sample_other_face_view(
                    batch, local_speakers, identity_rows, face_view_swap_probability,
                    np.random.default_rng(int(config["experiment"]["seed"]) + 7919 * epoch + batch_number),
                ) if face_view_swap_probability else batch
            )
            voice_rows = (
                sample_other_voice_view(
                    batch, local_speakers, identity_rows, voice_view_swap_probability,
                    np.random.default_rng(int(config["experiment"]["seed"]) + 20533 * epoch + batch_number),
                ) if voice_view_swap_probability else batch
            )
            face_batch = torch.from_numpy(local_faces[face_rows]).to(device)
            voice_batch = torch.from_numpy(local_voices[voice_rows]).to(device)
            target = torch.from_numpy(local_labels[batch]).to(device)
            face_embedding, voice_embedding = model(face_batch, voice_batch)
            loss = 0.5 * (
                classifier.loss(face_embedding, target) + classifier.loss(voice_embedding, target)
            )
            if same_gender_aam_weight:
                loss = loss + same_gender_aam_weight * .5 * (
                    classifier.same_gender_loss(face_embedding, target, classifier_genders)
                    + classifier.same_gender_loss(voice_embedding, target, classifier_genders)
                )
            contrastive_weight = float(config["loss"].get("contrastive_weight", 0.0))
            if contrastive_weight > 0:
                temperature = float(config["loss"].get("contrastive_temperature", 0.10))
                same_gender_weight = float(config["loss"].get("same_gender_negative_weight", 1.0))
                if same_gender_weight > 1:
                    gender = torch.from_numpy(local_genders[batch]).to(device)
                    contrastive_loss = gender_weighted_cross_modal_contrastive_loss(
                        face_embedding, voice_embedding, target, gender,
                        temperature=temperature,
                        same_gender_negative_weight=same_gender_weight,
                    )
                else:
                    contrastive_loss = cross_modal_supervised_contrastive_loss(
                        face_embedding, voice_embedding, target, temperature=temperature,
                    )
                loss = loss + contrastive_weight * contrastive_loss
            conditional_weight = float(config["loss"].get("same_gender_contrastive_weight", 0.0))
            if conditional_weight > 0:
                gender = torch.from_numpy(local_genders[batch]).to(device)
                loss = loss + conditional_weight * same_gender_cross_modal_contrastive_loss(
                    face_embedding, voice_embedding, target, gender,
                    temperature=float(config["loss"].get("same_gender_contrastive_temperature", 0.10)),
                )
            if gender_adversary is not None:
                gender = torch.from_numpy(local_genders[batch]).to(device)
                gender_loss = 0.5 * (
                    torch.nn.functional.cross_entropy(gender_adversary(face_embedding), gender)
                    + torch.nn.functional.cross_entropy(gender_adversary(voice_embedding), gender)
                )
                loss = loss + adversarial_weight * gender_loss
            hard_weight = float(config["loss"].get("same_gender_hard_negative_weight", 0.0))
            if hard_weight > 0:
                gender = torch.from_numpy(local_genders[batch]).to(device)
                hard_loss = same_gender_hard_negative_loss(
                    face_embedding, voice_embedding, target, gender,
                    margin=float(config["loss"].get("same_gender_hard_negative_margin", 0.2)),
                )
                loss = loss + hard_weight * hard_loss
            if age_hard_weight > 0:
                gender = torch.from_numpy(local_genders[batch]).to(device)
                face_age_batch = torch.from_numpy(local_face_ages[batch]).to(device)
                voice_age_batch = torch.from_numpy(local_voice_ages[batch]).to(device)
                loss = loss + age_hard_weight * age_matched_same_gender_hard_negative_loss(
                    face_embedding, voice_embedding, target, gender,
                    face_age_batch, voice_age_batch,
                    nearest_k=int(config["loss"].get("age_matched_nearest_k", 8)),
                    margin=float(config["loss"].get("age_matched_margin", 0.20)),
                    temperature=float(config["loss"].get("age_matched_temperature", 0.10)),
                )
            if curriculum_weight > 0:
                gender = torch.from_numpy(local_genders[batch]).to(device)
                face_age_batch = torch.from_numpy(local_face_ages[face_rows]).to(device)
                voice_age_batch = torch.from_numpy(local_voice_ages[voice_rows]).to(device)
                loss = loss + curriculum_weight * curriculum_cross_modal_negative_loss(
                    face_embedding, voice_embedding, target, gender,
                    face_age_batch, voice_age_batch, epoch=epoch,
                    nearest_k=int(config["loss"].get("curriculum_nearest_k", 8)),
                    margin=float(config["loss"].get("curriculum_margin", 0.20)),
                    temperature=float(config["loss"].get("curriculum_temperature", 0.10)),
                )
            alignment_weight = float(config["loss"].get("paired_alignment_weight", 0.0))
            if alignment_weight > 0:
                loss = loss + alignment_weight * paired_alignment_loss(
                    face_embedding, voice_embedding
                )
            prototype_weight = float(config["loss"].get("prototype_alignment_weight", 0.0))
            if prototype_weight > 0:
                loss = loss + prototype_weight * batch_identity_prototype_alignment_loss(
                    face_embedding, voice_embedding, target
                )
            if simplex_weight > 0:
                loss = loss + simplex_weight * fixed_simplex_center_alignment_loss(
                    face_embedding, voice_embedding, target, simplex_centers
                )
            intra_weight = float(config["loss"].get("intra_modal_contrastive_weight", 0.0))
            if intra_weight > 0:
                loss = loss + intra_weight * intra_modal_supervised_contrastive_loss(
                    face_embedding, voice_embedding, target,
                    temperature=float(config["loss"].get("intra_modal_temperature", 0.10)),
                )
            voice_intra_weight = float(config["loss"].get("voice_intra_contrastive_weight", 0.0))
            if voice_intra_weight > 0:
                loss = loss + voice_intra_weight * intra_modal_supervised_contrastive_loss(
                    voice_embedding, voice_embedding, target,
                    temperature=float(config["loss"].get("voice_intra_temperature", 0.10)),
                )
            decorrelation_weight = float(config["loss"].get("decorrelation_weight", 0.0))
            if decorrelation_weight > 0:
                loss = loss + decorrelation_weight * 0.5 * (
                    covariance_decorrelation_loss(face_embedding)
                    + covariance_decorrelation_loss(voice_embedding)
                )
            if external_source is not None:
                source_rng = np.random.default_rng(
                    int(config["experiment"]["seed"]) + 791357 * epoch + batch_number)
                source_face_rows, source_voice_rows, source_labels = sample_v3_english_batch(
                    external_source, source_rng)
                source_face = torch.from_numpy(external_source["face"][source_face_rows]).to(device)
                source_voice = torch.from_numpy(external_source["voice"][source_voice_rows]).to(device)
                source_target = torch.from_numpy(source_labels).to(device)
                source_fz, source_vz = model(source_face, source_voice)
                source_loss = 0.5 * (
                    external_classifier.loss(source_fz, source_target)
                    + external_classifier.loss(source_vz, source_target)
                )
                source_loss = source_loss + 0.20 * cross_modal_supervised_contrastive_loss(
                    source_fz, source_vz, source_target, temperature=0.10)
                source_loss = source_loss + 0.30 * batch_identity_prototype_alignment_loss(
                    source_fz, source_vz, source_target)
                loss = loss + external_weight * source_loss
            if v1_source is not None:
                v1_rng = np.random.default_rng(
                    int(config["experiment"]["seed"]) + 918737 * epoch + batch_number)
                v1_face_rows, v1_voice_rows, v1_labels = sample_v3_english_batch(
                    v1_source, v1_rng)
                v1_face = torch.from_numpy(v1_source["face"][v1_face_rows]).to(device)
                v1_voice = torch.from_numpy(v1_source["voice"][v1_voice_rows]).to(device)
                v1_target = torch.from_numpy(v1_labels).to(device)
                v1_fz, v1_vz = model(v1_face, v1_voice)
                v1_loss = 0.5 * (
                    v1_classifier.loss(v1_fz, v1_target)
                    + v1_classifier.loss(v1_vz, v1_target)
                )
                v1_loss = v1_loss + 0.20 * cross_modal_supervised_contrastive_loss(
                    v1_fz, v1_vz, v1_target, temperature=0.10)
                v1_loss = v1_loss + 0.30 * batch_identity_prototype_alignment_loss(
                    v1_fz, v1_vz, v1_target)
                loss = loss + v1_weight * v1_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable,
                config["training"]["gradient_clip_norm"],
            )
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        metrics, predictions = evaluate(
            model, faces, voices, row_map, args.trials_root, fold, device,
            selection_metric=config["training"].get("selection_metric", "speaker_balanced_eer"),
        )
        score = metrics["selection_score"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "selection_score": score,
                "standard_eer": metrics["standard"]["eer"],
                "same_gender_eer": metrics["same_gender"]["eer"],
                "standard_speaker_balanced_eer": metrics["standard"]["speaker_balanced_eer"],
                "same_gender_speaker_balanced_eer": metrics["same_gender"]["speaker_balanced_eer"],
            }
        )
        if score < best_score - config["training"]["min_delta"]:
            best_score = score
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "classifier": classifier.state_dict(),
                    "epoch": epoch,
                    "selection_score": score,
                    "class_map": class_map,
                    "face_center": face_center,
                    "voice_center": voice_center,
                    "gender_directions": gender_directions,
                    "config": config,
                },
                run / "best.pt",
            )
            predictions.to_csv(run / "predictions.csv", index=False)
            write_json(run / "best_metrics.json", metrics)
        else:
            patience += 1
        if patience >= config["training"]["early_stopping_patience"]:
            break
    with (run / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_metrics, final_predictions = evaluate(
        model, faces, voices, row_map, args.trials_root, fold, device,
        selection_metric=config["training"].get("selection_metric", "speaker_balanced_eer"),
    )
    final_predictions.to_csv(run / "predictions.csv", index=False)
    summary = {
        "stage": "T2-B01",
        "fold": fold,
        "seed": config["experiment"]["seed"],
        "training_speakers": len(training_speakers),
        "validation_speakers": len(validation_speakers),
        "training_rows": int(len(selected)),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "selection_score": final_metrics["selection_score"],
        "metrics": final_metrics,
        "checkpoint": str((run / "best.pt").resolve()),
        "checkpoint_sha256": sha256_file(run / "best.pt"),
        "split_hash": split["split_hash"],
        "locked_group0_opened": False,
        "track1_scores_or_checkpoints_read": False,
        "external_init_projector_sha256": init_projector_sha256,
        "resume_checkpoint_sha256": resume_sha256,
        "v3_english_auxiliary_weight": external_weight,
        "v3_english_source_rows": None if external_source is None else {
            "face": len(external_source["face"]), "voice": len(external_source["voice"]),
            "speakers": len(external_source["speaker_ids"]),
        },
        "v3_english_source_hashes": None if external_source is None else {
            "face_sha256": external_source["face_sha256"],
            "voice_sha256": external_source["voice_sha256"],
        },
        "v1_english_auxiliary_weight": v1_weight,
        "v1_english_source_rows": None if v1_source is None else {
            "face": len(v1_source["face"]), "voice": len(v1_source["voice"]),
            "speakers": len(v1_source["speaker_ids"]),
        },
        "v1_english_source_hashes": None if v1_source is None else {
            "face_sha256": v1_source["face_sha256"],
            "voice_sha256": v1_source["voice_sha256"],
            "split_sha256": v1_source["split_sha256"],
        },
        "training_seconds": time.perf_counter() - started,
    }
    summary["summary_hash"] = stable_hash(summary)
    write_json(run / "summary.json", summary)
    return summary


def write_report(summary: dict, path: Path) -> None:
    experiment_id = summary["experiment"].get("id", "T2-B01")
    lines = [
        f"# {experiment_id} — {summary.get('projector_type', 'linear')} 192D projector development run",
        "",
        "Decision: **development measurement; see experiment registry for promotion status**",
        "",
        "| Fold | Best epoch | Standard EER | Same-gender EER | Selection EER |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["folds"]:
        lines.append(
            f"| {row['fold']} | {row['best_epoch']} | {100 * row['metrics']['standard']['eer']:.2f}% | "
            f"{100 * row['metrics']['same_gender']['eer']:.2f}% | {100 * row['selection_score']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Mean selection EER: **{100 * summary['mean_selection_score']:.2f}%**",
            f"Fold standard deviation: **{100 * summary['std_selection_score']:.2f} pp**",
            f"Worst fold: **{100 * summary['worst_selection_score']:.2f}%**",
            "",
            "This run uses only fold-training identities for fitting and fold-validation identities for model selection; no Group 0, Track 1 scores/checkpoints or public labels were used for that selection.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Track 2 T2-B01")
    parser.add_argument("--cache-root", type=Path, default=ROOT.parent / "flag_raw_s01_v3_adaface_simam100")
    parser.add_argument("--split", type=Path, default=ROOT.parent / "Track 1/data_manifests/speaker_splits.json")
    parser.add_argument("--trials-root", type=Path, default=ROOT.parent / "Track 1/data_manifests/trials/folds")
    parser.add_argument("--audit", type=Path, default=ROOT / "artifacts/t2_00_audit.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/t2_b01_linear_192.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--experiment-name", help="Isolate a feature-cache ablation without changing the frozen recipe")
    parser.add_argument("--seed", type=int, help="Independent initialization for a confirmation run")
    parser.add_argument("--init-projector", type=Path, default=None,
                        help="Audited external-only projector initialization; classifier remains new")
    parser.add_argument("--resume-run-root", type=Path, default=None,
                        help="Continue matching XP08 fold checkpoints with train-only FLAG batches")
    parser.add_argument("--fold", type=int, choices=range(4), default=None)
    parser.add_argument("--outer-fold", type=int, choices=range(4), default=None,
                        help="Exclude this outer validation group from inner-fold fitting; requires --fold")
    args = parser.parse_args()
    if args.outer_fold is not None and args.fold is None:
        parser.error("--outer-fold requires one --fold inner validation")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit["status"] != "PASS":
        raise RuntimeError("T2-B01 requires a passing T2-00 audit")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.experiment_name:
        config["experiment"]["name"] = args.experiment_name
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    folds = [args.fold] if args.fold is not None else list(range(4))
    results = [train_fold(args, config, fold) for fold in folds]
    if len(folds) == 4:
        scores = np.asarray([row["selection_score"] for row in results])
        aggregate = {
            "stage": "T2-B01",
            "experiment": config["experiment"],
            "config": str(args.config.resolve()),
            "config_hash": stable_hash(config),
            "projector_type": config["model"].get("projector_type", "linear"),
            "audit_hash": audit["audit_hash"],
            "folds": results,
            "mean_selection_score": float(scores.mean()),
            "std_selection_score": float(scores.std()),
            "worst_selection_score": float(scores.max()),
            "locked_group0_opened": False,
            "public_labels_used": False,
            "track1_scores_or_checkpoints_read": False,
        }
        aggregate["summary_hash"] = stable_hash(aggregate)
        base = args.output_root / config["experiment"]["name"] / f"seed_{config['experiment']['seed']}"
        write_json(base / "cv_summary.json", aggregate)
        write_report(aggregate, base / "T2_B01_REPORT.md")
        print(json.dumps({
            "mean_selection_eer_percent": 100 * aggregate["mean_selection_score"],
            "std_percent_points": 100 * aggregate["std_selection_score"],
            "worst_fold_percent": 100 * aggregate["worst_selection_score"],
        }, indent=2))


if __name__ == "__main__":
    main()
