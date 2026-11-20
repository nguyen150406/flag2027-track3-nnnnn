"""Pair-local speaker embeddings for every FLAG voice row with a frozen WeSpeaker model.

Kaldi 80-bin fbank (25/10 ms), per-utterance mean normalisation, whole utterance
split into <=30 s chunks averaged (same as RAW-S01 SimAM). No cross-file stats.
Usage: python extract_wespeaker.py <hf_repo> <out_name> [--device mps|cpu]
"""
import sys, wave, json
from pathlib import Path
import numpy as np, torch, torchaudio, yaml
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
import types  # noqa: E402
_pkg = types.ModuleType("wespeaker"); _pkg.__path__ = [str(ROOT / ".tools/wespeaker/wespeaker")]
sys.modules["wespeaker"] = _pkg  # skip package __init__ (CLI deps)
_fe = types.ModuleType("wespeaker.frontend"); _fe.__path__ = [_pkg.__path__[0] + "/frontend"]
sys.modules["wespeaker.frontend"] = _fe  # skip optional s3prl/whisper frontends
from wespeaker.models.speaker_model import get_speaker_model  # noqa: E402
import pandas as pd  # noqa: E402

meta = pd.read_csv(ROOT / "flag_raw_s01_v3_adaface_simam100/metadata.csv")


def voice_path(r):
    if r.partition == "train":
        return ROOT / "data/train_set/train_set" / r.voice_relpath
    return ROOT / "data/dev_set" / r.protocol / r.voice_relpath


def load_pcm(path):
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, path
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    x = torch.from_numpy(x.astype(np.float32))[None]
    if sr != 16000:
        x = torchaudio.functional.resample(x, sr, 16000)
    return x  # int16 scale, as in WeSpeaker recipes


def main(repo, name, device, rows=None, out_dir=None):  # rows: table with a 'path' column (t3_eval_extract.py), out_dir: where to write
    if repo == "local:simam100":  # official WeSpeaker VoxBlink2 SimAM-R100 used by the FLAG caches
        cfg = dict(model="SimAM_ResNet100_ASP", model_args=dict(in_planes=64, embed_dim=256, acoustic_dim=80))
        weights = ROOT / ".tools/simam/unpacked/voxblink2_samresnet100/avg_model.pt"
    else:
        cfg = yaml.safe_load(open(hf_hub_download(repo, "config.yaml")))
        weights = hf_hub_download(repo, "avg_model.pt")
    model = get_speaker_model(cfg["model"])(**cfg["model_args"])
    state = torch.load(weights, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not [k for k in missing if "projection" not in k], missing
    model.to(device).eval()
    nmel = cfg.get("dataset_args", {}).get("fbank_args", {}).get("num_mel_bins", 80)
    out = []
    table = meta if rows is None else rows
    for i, r in enumerate(table.itertuples()):
        x = load_pcm(r.path if hasattr(r, "path") else voice_path(r))
        chunks = [x[:, s:s + 30 * 16000] for s in range(0, x.shape[1], 30 * 16000)]
        chunks = [c for c in chunks if c.shape[1] >= 16000] or [x]
        embs = []
        for c in chunks:
            fb = torchaudio.compliance.kaldi.fbank(c, num_mel_bins=nmel, frame_length=25, frame_shift=10,
                                                   dither=0.0, sample_frequency=16000, window_type="hamming")
            fb = fb - fb.mean(0, keepdim=True)
            with torch.no_grad():
                e = model(fb[None].to(device))
            e = e[-1] if isinstance(e, (tuple, list)) else e
            embs.append(e[0].float().cpu())
        e = torch.stack(embs).mean(0)
        out.append((e / e.norm()).numpy())
        if i % 1000 == 0:
            print(i, flush=True)
    d = Path(out_dir) if out_dir else ROOT / "Track 3/caches/voice" / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "voice_embeddings.float32.npy", np.stack(out).astype(np.float32))
    json.dump(dict(repo=repo, model=cfg["model"], rows=len(out), pair_local=True), open(d / "PROVENANCE.json", "w"))


if __name__ == "__main__":
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    main(sys.argv[1], sys.argv[2], dev)
