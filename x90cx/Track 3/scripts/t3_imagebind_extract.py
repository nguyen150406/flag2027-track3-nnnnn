"""Frozen ImageBind-huge face (vision) / voice (audio) embeddings on the Mac, R2 (2026-11-12).
Same pinned public source (commit 53680b02) and checkpoint as Track 2 XP65, same preprocessing (224 bicubic + centre crop + CLIP
normalisation; audio = three fixed 2-s crops start/middle/end -> 128-bin kaldi fbank, 204 frames, (x + 4.268) / 9.138), L2 output.
No training, no corpus statistic.
  python t3_imagebind_extract.py <list.csv with a 'path' column> <face|voice> <out.npy> [--device mps]
  python t3_imagebind_extract.py check [--n 32]   -> cosine vs the XP65 FLAG-train caches (port check)"""
import io, math, sys, types, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
T3 = Path(__file__).resolve().parents[1]; IB = T3 / "caches/imagebind"; COMMIT = "53680b02d7e37b19b124fa37bae4b6c98c38f5be"
sys.path.append(str(IB / "pylib")); REPO = IB / f"ImageBind-{COMMIT}"; sys.path.insert(0, str(REPO))
pkg = types.ModuleType("imagebind"); pkg.__path__ = [str(REPO / "imagebind")]; pkg.__package__ = "imagebind"; sys.modules["imagebind"] = pkg  # skip data.py
from imagebind.models.imagebind_model import imagebind_huge, ModalityType
from torchvision import transforms
from PIL import Image
import soundfile as sf, torchaudio
from scipy.signal import resample_poly

VT = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224), transforms.ToTensor(),
                         transforms.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])


def vision_input(path):  # JPEG decoded with OpenCV's libjpeg-turbo = the Kaggle decoder of the XP65 / ibv train caches (Mac Pillow: IJG, cos .964)
    import cv2; return VT(Image.fromarray(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)))


def audio_input(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True); a = x.mean(1)
    assert len(a) and np.isfinite(a).all(), path
    if sr != 16000: g = math.gcd(int(sr), 16000); a = resample_poly(a, 16000 // g, int(sr) // g).astype("float32")
    if len(a) < 32000: a = np.pad(a, (0, 32000 - len(a)))
    mx = len(a) - 32000; clips = []
    for st in (0, mx // 2, mx):
        w = torch.from_numpy(a[st:st + 32000].copy()).unsqueeze(0); w -= w.mean()
        mel = torchaudio.compliance.kaldi.fbank(w, htk_compat=True, sample_frequency=16000, use_energy=False, window_type="hanning", num_mel_bins=128,
                                                dither=0.0, frame_length=25, frame_shift=10).T
        mel = torch.nn.functional.pad(mel, (0, 204 - mel.shape[1])) if mel.shape[1] < 204 else mel[:, :204]
        clips.append((mel.unsqueeze(0) + 4.268) / 9.138)
    return torch.stack(clips)


def model(dev):
    m = imagebind_huge(pretrained=False); m.load_state_dict(torch.load(IB / "imagebind_huge.pth", map_location="cpu", weights_only=True), strict=True)
    for col in (m.modality_preprocessors, m.modality_trunks, m.modality_heads, m.modality_postprocessors):
        for k in list(col):
            if k not in (ModalityType.VISION, ModalityType.AUDIO): del col[k]
    return m.eval().requires_grad_(False).to(dev)


@torch.inference_mode()
def encode(m, paths, kind, dev, bs=16):
    mod = ModalityType.VISION if kind == "face" else ModalityType.AUDIO; fn = vision_input if kind == "face" else audio_input; out = []
    for i in range(0, len(paths), bs):
        x = m({mod: torch.stack([fn(p) for p in paths[i:i + bs]]).to(dev)})[mod].float()
        x = torch.nn.functional.normalize(x, dim=1); assert torch.isfinite(x).all(); out.append(x.cpu().numpy())
        if i % (bs * 20) == 0: print(f"  {kind} {i + len(x)}/{len(paths)}", flush=True)
    return np.vstack(out)


if __name__ == "__main__":
    a = sys.argv[1:]; dev = torch.device(a[a.index("--device") + 1] if "--device" in a else "mps")
    if a[0] == "check":
        n = int(a[a.index("--n") + 1]) if "--n" in a else 32
        sys.path.insert(0, str(T3 / "scripts")); from t3lib import META
        X = T3.parent / "Track 2/outputs/t2_xp65_imagebind_train/extraction"; ids = open(X / "pair_ids.txt").read().split(); pos = {s: i for i, s in enumerate(ids)}
        tr = META[META.partition.eq("train")].sort_values("row_id"); j = np.random.default_rng(0).choice(len(tr), n, replace=False); base = T3.parent / "data/train_set/train_set"
        m = model(dev); r = np.array([pos[s] for s in tr.pair_id.astype(str).iloc[j]])
        for kind, col, ref in (("face", "face_relpath", "face"), ("voice", "voice_relpath", "voice")):
            e = encode(m, [base / p for p in tr[col].iloc[j]], kind, dev); R = np.load(X / f"{ref}_embeddings.float32.npy")[r]
            c = (e * R / np.linalg.norm(R, axis=1, keepdims=True)).sum(1); print(f"CHECK {kind}: cos vs XP65 mean {c.mean():.4f} min {c.min():.4f}")
    else:
        paths = pd.read_csv(a[0]).path.tolist(); m = model(dev); e = encode(m, paths, a[1], dev); np.save(a[2], e); print("wrote", a[2], e.shape)
