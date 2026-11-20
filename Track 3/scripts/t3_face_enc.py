"""Frozen face-encoder screen (2026-10-27, user "oke đồng ý" to the idea sweep): embeds every FLAG face (train + dev, META row order)
and every external face row (rows_<src>.csv of the sources whose images are mounted) with new frozen encoders, in the layout the
projector code reads: <out>/flag_face_<tag>.float32.npy and <out>/<src>_face_<tag>.{float32,ok}.npy (install = copy to caches/<tag>/
face_embeddings.float32.npy and caches/ext/). Each image is encoded independently (pair-local), L2-normalised.
  c336 = OpenAI CLIP ViT-L/14@336 (image_embeds)          pel  = Meta PE-Core-L/14-336 (timm, pooled embedding)
  mc2h = MetaCLIP 2 worldwide ViT-H/14 (image_embeds)       cl16 = CLIP ViT-L/14 (224) CLS token after block 16 (intermediate layer)
  clipmouth / clipnose / cliplow (2026-11-04, research direction A2: voice predicts nose/mouth shape within gender, Li et al. ACM MM 2023,
  Cross-Modal Perceptionist CVPR 2022): frozen CLIP ViT-L/14 image_embeds of a square region crop placed from YuNet's 5 landmarks of
  THIS image (eyes, nose tip, mouth corners; iod = inter-ocular distance): mouth = centre mouth, side 1.3 iod; nose = centre nose tip,
  side 1.0 iod; low = nose-to-chin incl. jaw, side 2.0 iod. No face found -> fixed landmarks (FLAG-train medians), pair-local.
  ibv = ImageBind-huge vision tower (RF reproduction 2026-11-05; t3_imagebind.py, official 224 CLIP-style transform), 1024-d.
  --list csv (column 'face'): encode exactly these images -> <out>/list_face_<tag>.float32.npy, nothing else (eval phase).
Usage: python t3_face_enc.py --out DIR --enc c336,cl16 [--roots flag=DIR flagdev=DIR vox=DIR vox2=DIR vb=DIR lrs3=DIR] [--device cuda:0]
       [--limit N: first N FLAG train rows only (smoke)]"""
import argparse, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from PIL import Image
from t3lib import META
import t3data as D

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--enc", required=True); ap.add_argument("--roots", nargs="+", default=[])
ap.add_argument("--device", default="cuda"); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--list", default=""); ap.add_argument("--sources", default="vox,vox2,vb,lrs3"); ap.add_argument("--yunet", default=str(D.T3 / "caches/vb2/kaggle_meta/yunet.onnx"))
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); dev = torch.device(a.device); T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m] {a.enc}:", m, flush=True)
ROOTS = {k: [Path(p) for p in v.split(",")] for k, v in (r.split("=", 1) for r in a.roots)}
def find(src, rel):  # same lookup as t3_ft_clip.find
    for r in ROOTS.get(src, []):
        for p in (r / rel, r / src / rel, r / "train" / rel, r / "test" / rel, r / "dev" / rel):
            if p.exists(): return p
    return None
CLIP_MS = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
amp = torch.float16 if dev.type == "cuda" else torch.float32  # T4: fp16 (bf16 is emulated)


REGION = {"clipmouth": "mouth", "clipnose": "nose", "cliplow": "low"}
LMK0 = np.array([[.373, .288], [.628, .284], [.498, .419], [.391, .537], [.615, .536]])  # FLAG-train median YuNet landmarks (fraction of w, h)
import threading; _tl = threading.local(); LMK = {}  # path -> landmarks (computed once per image, reused by the 3 region tags)


def landmarks(path, im):
    if path in LMK: return LMK[path]
    import cv2
    if not hasattr(_tl, "det"): _tl.det = cv2.FaceDetectorYN.create(a.yunet, "", (224, 224), 0.5, 0.3, 5000)
    w, h = im.size; _tl.det.setInputSize((w, h)); _, f = _tl.det.detect(np.asarray(im)[:, :, ::-1].copy())
    if f is None: out = LMK0 * np.array([w, h]); found = False
    else:  # most confident face, ties broken towards the image centre
        c = np.array([w / 2, h / 2]); j = int(np.argmax(f[:, 14] - 0.002 * np.linalg.norm(f[:, :2] + f[:, 2:4] / 2 - c, axis=1)))
        out = f[j, 4:14].reshape(5, 2).astype(np.float64); found = True
    LMK[path] = (out, found); return LMK[path]


def region_box(lm, region):
    re_, le, nose, rm, lmc = lm; eye = (re_ + le) / 2; iod = max(np.linalg.norm(le - re_), 8.0); mc = (rm + lmc) / 2
    c, side = {"mouth": (mc, 1.3 * iod), "nose": (nose, 1.0 * iod), "low": ((nose + mc) / 2 + .35 * (mc - eye), 2.0 * iod)}[region]
    return tuple(int(round(v)) for v in (c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2))


def model(tag):
    """-> (forward(pixels) -> embeddings, input size, (mean, std))"""
    if tag in REGION:
        from transformers import CLIPVisionModelWithProjection
        m = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
        return (lambda x: m(pixel_values=x).image_embeds), 224, CLIP_MS
    if tag in ("c336", "cl16"):
        from transformers import CLIPVisionModelWithProjection
        m = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14-336" if tag == "c336" else "openai/clip-vit-large-patch14")
        m = m.to(dev).eval()
        if tag == "c336": return (lambda x: m(pixel_values=x).image_embeds), 336, CLIP_MS
        return (lambda x: m(pixel_values=x, output_hidden_states=True).hidden_states[16][:, 0]), 224, CLIP_MS
    if tag == "mc2h":
        from transformers import MetaClip2VisionModelWithProjection
        m = MetaClip2VisionModelWithProjection.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu").to(dev).eval()
        return (lambda x: m(pixel_values=x).image_embeds), 224, CLIP_MS
    if tag == "pel":
        import timm
        m = timm.create_model("vit_pe_core_large_patch14_336.fb", pretrained=True, num_classes=0).to(dev).eval()
        c = m.pretrained_cfg; assert c["input_size"][-1] == 336, c["input_size"]
        return m, 336, (list(c["mean"]), list(c["std"]))
    if tag == "ibv":
        import t3_imagebind as IB
        m = IB.load(dev, (IB.VISION,))
        return (lambda x: m({IB.VISION: x})[IB.VISION]), 224, CLIP_MS
    raise ValueError(tag)


def loader(size, ms, region=None):
    mean, std = (torch.tensor(v).view(3, 1, 1) for v in ms)
    def load(path):  # shorter side -> size (bicubic), centre crop, normalise (the CLIP recipe of t3_ft_clip.load at any size)
        im = Image.open(path).convert("RGB")
        if region: im = im.crop(region_box(landmarks(path, im)[0], region)).resize((size, size), Image.BICUBIC)  # outside the image -> black
        w, h = im.size; s = size / min(w, h)
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC); w, h = im.size
        im = im.crop(((w - size) // 2, (h - size) // 2, (w - size) // 2 + size, (h - size) // 2 + size))
        return (torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1) - mean) / std
    return load


@torch.no_grad()
def encode(fwd, load, paths, pool):
    out = []; t = time.time()
    for s in range(0, len(paths), a.bs):
        x = torch.stack(list(pool.map(load, paths[s:s + a.bs]))).to(dev)
        with torch.autocast(dev.type, dtype=amp, enabled=dev.type == "cuda"):
            out.append(F.normalize(fwd(x).float(), dim=1).cpu())
        if (s // a.bs) % 100 == 0: log(f"  {s + len(x)}/{len(paths)} ({(s + len(x)) / (time.time() - t):.0f} img/s)")
    return torch.cat(out).numpy().astype(np.float32)


if a.list:  # eval phase: only the listed images
    _c = pd.read_csv(a.list); ps = (_c.face if "face" in _c else _c.path).astype(str).tolist(); miss = [p for p in ps if not Path(p).exists()]; assert not miss, f"{len(miss)} missing, e.g. {miss[:2]}"
    with ThreadPoolExecutor(max(2, (os.cpu_count() or 4) // 2)) as pool:
        for tag in a.enc.split(","):
            fwd, size, ms = model(tag); E = encode(fwd, loader(size, ms, REGION.get(tag)), ps, pool)
            np.save(OUT / f"list_face_{tag}.float32.npy", E); log(f"list {E.shape}")
    raise SystemExit
TR = META.sort_values("row_id")
flag = [find("flag", r.face_relpath) if r.partition == "train" else find("flagdev", f"{r.protocol}/{r.face_relpath}") for r in TR.itertuples()]
if a.limit:  # N evenly spaced FLAG train rows (many identities)
    ntr = int(TR.partition.eq("train").sum()); SEL = np.linspace(0, ntr - 1, a.limit).astype(int); flag = [flag[i] for i in SEL]
miss = sum(p is None for p in flag); assert not miss, f"{miss} FLAG images missing"
with ThreadPoolExecutor(max(2, (os.cpu_count() or 4) // 2)) as pool:
    for tag in a.enc.split(","):
        fwd, size, ms = model(tag); load = loader(size, ms, {"clipmouth": "mouth", "clipnose": "nose", "cliplow": "low"}.get(tag)); log(f"loaded, input {size}, mean/std {ms}")
        E = encode(fwd, load, flag, pool); np.save(OUT / f"flag_face_{tag}.float32.npy", E); log(f"FLAG {E.shape}")
        if a.limit:  # smoke: identity sanity on the encoded FLAG train rows (same-ID cos must exceed different-ID cos)
            spk = TR[TR.partition.eq("train")].identity.astype(str).to_numpy()[SEL]; S = E @ E.T; same = spk[:, None] == spk[None, :]
            np.fill_diagonal(same, False); off = ~np.eye(len(E), dtype=bool) & ~same
            log(f"SMOKE same-ID cos {S[same].mean():.3f} vs different-ID {S[off].mean():.3f}")
            assert tag in REGION or S[same].mean() > S[off].mean() + .02  # region crops: logged only (weak identity signal is expected)
            continue
        for s in a.sources.split(","):
            rows = pd.read_csv(D.EXT / f"rows_{s}.csv"); ps = [find(s, f) for f in rows.face]; ok = np.array([p is not None for p in ps])
            if not ok.any(): log(f"{s}: no images, skipped"); continue
            e = np.zeros((len(ps), E.shape[1]), np.float32); e[ok] = encode(fwd, load, [p for p in ps if p is not None], pool)
            np.save(OUT / f"{s}_face_{tag}.float32.npy", e); np.save(OUT / f"{s}_face_{tag}.ok.npy", ok); log(f"{s} encoded {ok.sum()}/{len(ok)}")
        if tag in REGION: log(f"landmarks: {len(LMK)} images, face found in {sum(v[1] for v in LMK.values())}")
        del fwd; torch.cuda.empty_cache() if dev.type == "cuda" else None
log("DONE")
