"""ImageBind-huge for the TP reproduction (user 2026-11-05): official code @53680b02 + public checkpoint (CC BY-NC 4.0),
loaded through a namespace bootstrap (the package __init__ imports unused video helpers; recipe of Track 2 XP65).
  load(dev, modalities)  vision and/or audio towers only, frozen
  audio_input(x)         (3, 1, 128, 204): three fixed 2-s clips (start / middle / end) of a 16-kHz waveform, kaldi fbank 128,
                         official normalisation ((x + 4.268) / 9.138); the model averages the three clip embeddings
  add_lora(model)        LoRA (rank 8, alpha 16) on the q and v rows of every audio-trunk attention in_proj_weight
  train                  the RF 'ib' LoRA: symmetric InfoNCE (tau .05) between frozen ImageBind face embeddings and LoRA audio
                         embeddings of the same identity (face from another video when there is one), AdamW lr 1e-4, 1200 steps,
                         batch 64 identities, external clips only (no FLAG audio), Bengali-excluded identities removed.
Usage: python t3_imagebind.py train --rows rows_vox2.csv --faces vox2_face_ibv.float32.npy --face_ok ...ok.npy --wav_root DIR
                                    --exclude exclude_bengali_v3.csv --out lora.pt [--steps 1200] [--device cuda]
       python t3_imagebind.py encode --list voices.csv [--lora lora.pt] [--prefix OLD=NEW] --out emb.npy   (no t3data needed)
Encode with the LoRA inside the Track tree: t3_extract_wavlmsv.py --name ibl --model lora.pt (frozen tower: --name iba)."""
import math, os, sys, time, types, urllib.request, zipfile
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

COMMIT = "53680b02d7e37b19b124fa37bae4b6c98c38f5be"
CKPT_URL = "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth"
CACHE = Path(os.environ.get("IB_CACHE", Path.home() / ".cache/imagebind"))
VISION, AUDIO = "vision", "audio"


def _source():
    repo = CACHE / f"ImageBind-{COMMIT}"
    if not repo.exists():
        CACHE.mkdir(parents=True, exist_ok=True); z = CACHE / "src.zip"
        urllib.request.urlretrieve(f"https://codeload.github.com/facebookresearch/ImageBind/zip/{COMMIT}", z)
        with zipfile.ZipFile(z) as f: f.extractall(CACHE)
        z.unlink()
    if "imagebind" not in sys.modules:
        pkg = types.ModuleType("imagebind"); pkg.__path__ = [str(repo / "imagebind")]; pkg.__package__ = "imagebind"; sys.modules["imagebind"] = pkg
        sys.path.insert(0, str(repo))
    return repo


def load(dev, modalities=(VISION, AUDIO), pretrained=True):
    pretrained = pretrained and not os.environ.get("IB_RANDOM")  # IB_RANDOM=1: random weights, code-path tests only
    _source()
    from imagebind.models.imagebind_model import imagebind_huge
    m = imagebind_huge(pretrained=False)
    if pretrained:
        ck = CACHE / "imagebind_huge.pth"
        if not ck.exists():
            urllib.request.urlretrieve(CKPT_URL, str(ck) + ".part"); os.replace(str(ck) + ".part", ck)
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True), strict=True)
    for coll in (m.modality_preprocessors, m.modality_trunks, m.modality_heads, m.modality_postprocessors):
        for k in list(coll):
            if k not in modalities: del coll[k]
    return m.eval().requires_grad_(False).to(dev)


def audio_input(x):
    import torchaudio
    x = np.asarray(x, np.float32)
    if len(x) < 32000: x = np.pad(x, (0, 32000 - len(x)))
    mx = len(x) - 32000; clips = []
    for s in (0, mx // 2, mx):
        w = torch.from_numpy(x[s:s + 32000].copy())[None]; w = w - w.mean()
        mel = torchaudio.compliance.kaldi.fbank(w, htk_compat=True, sample_frequency=16000, use_energy=False, window_type="hanning",
                                                num_mel_bins=128, dither=0.0, frame_length=25, frame_shift=10).T
        mel = F.pad(mel, (0, 204 - mel.shape[1])) if mel.shape[1] < 204 else mel[:, :204]
        clips.append(((mel + 4.268) / 9.138)[None])
    return torch.stack(clips)


def embed(m, modality, x):  # -> L2-normalised embeddings (B, 1024)
    return F.normalize(m({modality: x})[modality].float(), dim=-1)


class QVLoRA(nn.Module):
    """W + s * [Bq Aq ; 0 ; Bv Av] on an (3D x D) in_proj_weight: low-rank updates of the query and value projections."""
    def __init__(s, d, r=8, alpha=16):
        super().__init__(); s.s = alpha / r
        s.Aq, s.Av = (nn.Parameter(torch.empty(r, d)) for _ in range(2)); s.Bq, s.Bv = (nn.Parameter(torch.zeros(d, r)) for _ in range(2))
        for a in (s.Aq, s.Av): nn.init.kaiming_uniform_(a, a=math.sqrt(5))
    def forward(s, W):
        d = W.shape[1]; z = torch.zeros(d, d, dtype=W.dtype, device=W.device)
        return W + torch.cat([s.s * (s.Bq @ s.Aq), z, s.s * (s.Bv @ s.Av)]).to(W.dtype)


def add_lora(m, r=8, alpha=16):
    import torch.nn.utils.parametrize as P
    n = 0
    for mod in m.modality_trunks[AUDIO].modules():
        if isinstance(mod, nn.MultiheadAttention):
            assert mod._qkv_same_embed_dim, "needs a packed in_proj_weight"
            P.register_parametrization(mod, "in_proj_weight", QVLoRA(mod.embed_dim, r, alpha).to(mod.in_proj_weight.device)); n += 1
    assert n >= 12, n
    return [p for k, p in m.named_parameters() if ".parametrizations.in_proj_weight." in k and k.split(".")[-1] in ("Aq", "Av", "Bq", "Bv")]


def lora_state(m): return {k: v.detach().cpu() for k, v in m.state_dict().items() if k.split(".")[-1] in ("Aq", "Av", "Bq", "Bv")}


def load_lora(m, path):
    add_lora(m); st = torch.load(path, map_location="cpu", weights_only=True); miss = m.load_state_dict(st, strict=False)
    assert not miss.unexpected_keys and not [k for k in st if k not in m.state_dict()], miss
    return m


def train(a):
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, str(Path(__file__).parent)); from t3_lid_audit import read16
    dev = torch.device(a.device); T0 = time.time(); log = lambda s: print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m] lora:", s, flush=True)
    rows = pd.read_csv(a.rows); FE = np.load(a.faces, mmap_mode="r"); fok = np.load(a.face_ok)
    ex = pd.read_csv(a.exclude); ex = set(ex.identity[ex.src.eq(a.src)].astype(str))
    wav = np.array([str(Path(a.wav_root) / f"{ci}.wav") for ci in rows.clip_index]); wok = np.array([Path(p).exists() for p in wav])
    keep = ~rows.identity.astype(str).isin(ex).to_numpy()
    vid = rows.voice.str.split("/").str[-2].to_numpy(); ids = rows.identity.astype(str).to_numpy()
    F_, V_ = {}, {}
    for i in np.flatnonzero(keep & fok): F_.setdefault(ids[i], []).append(i)
    for i in np.flatnonzero(keep & wok): V_.setdefault(ids[i], []).append(i)
    P = sorted(set(F_) & set(V_)); log(f"{len(P)} identities, {sum(len(V_[p]) for p in P)} voice rows (excluded {len(ex)} ids of {a.src})")
    m = load(dev, (AUDIO,)); params = add_lora(m); opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    m.train(); scaler = torch.amp.GradScaler(enabled=dev.type == "cuda"); rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    pool = ThreadPoolExecutor(8)
    def batch():
        sel = rng.choice(len(P), a.batch, replace=False); fi, vi = [], []
        for k in sel:
            p = P[k]; v = V_[p][rng.integers(len(V_[p]))]; other = [f for f in F_[p] if vid[f] != vid[v]] or F_[p]
            fi.append(other[rng.integers(len(other))]); vi.append(v)
        return torch.from_numpy(np.asarray(FE[fi], np.float32)), list(pool.map(lambda i: audio_input(read16(wav[i])), vi))
    nxt = pool.submit(batch)
    for step in range(1, a.steps + 1):
        f, au = nxt.result(); nxt = pool.submit(batch)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            v = embed(m, AUDIO, torch.stack(au).to(dev))
        f = F.normalize(f.to(dev), dim=-1); lg = f @ v.float().T / a.tau; y = torch.arange(len(f), device=dev)
        loss = (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if step % 50 == 0 or step == 1: log(f"step {step}/{a.steps} loss {loss.item():.3f}")
    torch.save(lora_state(m), a.out); log(f"saved {a.out}")


@torch.no_grad()
def encode(a):  # voice list -> ImageBind audio embeddings (frozen, or with --lora), one file at a time through the tower
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).parent)); from t3_lid_audit import read16
    dev = torch.device(a.device); m = load(dev, (AUDIO,))
    if a.lora: load_lora(m, a.lora); m.eval()
    c = pd.read_csv(a.list); ps = (c.voice if "voice" in c else c.path).astype(str).tolist()
    if a.prefix: old, new = a.prefix.split("=", 1); ps = [new + p[len(old):] if p.startswith(old) else p for p in ps]
    out = np.zeros((len(ps), 1024), np.float32)
    for i in range(0, len(ps), 32):
        x = torch.stack([audio_input(read16(p)) for p in ps[i:i + 32]]).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"): out[i:i + len(x)] = embed(m, AUDIO, x).cpu().numpy()
        if i % 3200 == 0: print(time.strftime("%H:%M:%S"), f"encoded {i + len(x)}/{len(ps)}", flush=True)
    np.save(a.out, out); print("saved", a.out, out.shape)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("train", "encode"))
    ap.add_argument("--list", default=""); ap.add_argument("--lora", default=""); ap.add_argument("--prefix", default="", help="old=new path prefix swap")
    ap.add_argument("--rows"); ap.add_argument("--faces"); ap.add_argument("--face_ok")
    ap.add_argument("--wav_root"); ap.add_argument("--exclude"); ap.add_argument("--src", default="vox2")
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--tau", type=float, default=.05); ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); (train if a.cmd == "train" else encode)(a)
