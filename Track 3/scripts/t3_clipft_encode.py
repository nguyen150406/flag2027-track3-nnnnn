"""Encode face images with CLIP ViT-L/14 (frozen, or + a t3_ft_clip.py LoRA checkpoint) exactly as t3_ft_clip.py does
(same LoRA layout, same PIL preprocessing, L2-normalised image_embeds). --tta averages the image and its mirror.
  python t3_clipft_encode.py <ckpt.pt|none> v1 <name> [--tta] [--device mps]
      -> caches/ext/v1_face_<name>.float32.npy + .ok.npy (rows_v1.csv order; images under external_mavceleb/v1_complete)
  python t3_clipft_encode.py <ckpt.pt|none> list <paths.csv> --out <emb.npy> [--tta]
      -> embeddings of the csv's 'path' column, in order
  python t3_clipft_encode.py <ckpt.pt|none> check <ref.npy> [--n 48]
      -> cosine of fresh FLAG train-row encodings vs a reference FLAG face cache (decoder / port check)"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]; T3 = ROOT / "Track 3"
V1IMG = ROOT / "external_mavceleb/v1_complete/mavceleb_v1_complete-001/v1/faces"
MEAN, STD = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1), torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


class LoRA(nn.Module):  # same parameter names as t3_ft_clip.py so its state dict loads unchanged
    def __init__(self, base, r):
        super().__init__(); self.base = base; self.A = nn.Parameter(torch.zeros(r, base.in_features)); self.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x): return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))


def build(ckpt, dev, alpha=1.0):  # alpha: WiSE-FT blend, LoRA update B@A scaled by alpha (t3_ft_clip.py --alpha; research direction A3)
    from transformers import CLIPVisionModelWithProjection
    ck = torch.load(ckpt, map_location="cpu", weights_only=False) if ckpt != "none" else {}
    if ck and alpha != 1.0: ck = dict(ck, lora={k: v * alpha if k.endswith(".B") else v for k, v in ck["lora"].items()})
    farl = ck.get("encoder") == "farl"  # t3_ft_clip.py --encoder farl checkpoints sit on the HF FaRL conversion
    clip = CLIPVisionModelWithProjection.from_pretrained(str(ROOT / ".tools/farl/farl_hf") if farl else "openai/clip-vit-large-patch14")
    if ck:
        for layer in clip.vision_model.encoder.layers[-ck["k"]:]:
            for mod, names in ((layer.self_attn, ("q_proj", "k_proj", "v_proj", "out_proj")), (layer.mlp, ("fc1", "fc2"))):
                for n in names: setattr(mod, n, LoRA(getattr(mod, n), ck["rank"]))
        missing, unexpected = clip.load_state_dict(ck["lora"], strict=False)
        assert not unexpected and len(ck["lora"]) == 12 * ck["k"], (len(ck["lora"]), unexpected[:3])
    return clip.to(dev).eval()


def rgb(path, turbo=False):  # turbo: decode with OpenCV's libjpeg-turbo = the Kaggle Pillow decoder that made the LoRA caches
    if not turbo: return Image.open(path).convert("RGB")  # (anaconda Pillow on the Mac uses IJG libjpeg 9: pixels differ by up to ~13 levels)
    import cv2; return Image.fromarray(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))


def load(path, flip=False, turbo=False):  # copy of t3_ft_clip.load
    im = rgb(path, turbo); w, h = im.size; s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size
    im = im.crop(((w - 224) // 2, (h - 224) // 2, (w - 224) // 2 + 224, (h - 224) // 2 + 224))
    if flip: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    x = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)
    return (x - MEAN) / STD


@torch.no_grad()
def encode(clip, paths, dev, tta=False, bs=32, turbo=False):
    pool, out, t0 = ThreadPoolExecutor(8), [], time.time()
    emb = lambda px: F.normalize(clip(pixel_values=px.to(dev)).image_embeds.float(), dim=1).cpu()
    for s in range(0, len(paths), bs):
        ch = paths[s:s + bs]; e = emb(torch.stack(list(pool.map(lambda p: load(p, False, turbo), ch))))
        if tta: e = F.normalize(e + emb(torch.stack(list(pool.map(lambda p: load(p, True), ch)))), dim=1)
        out.append(e)
        if (s // bs) % 50 == 0: print(f"  {s + len(ch)}/{len(paths)}  {(s + len(ch)) / (time.time() - t0):.1f} img/s", flush=True)
    return torch.cat(out).numpy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("mode"); ap.add_argument("arg")
    ap.add_argument("--tta", action="store_true"); ap.add_argument("--device", default="mps"); ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--turbo", action="store_true", help="libjpeg-turbo decode (match Kaggle-made caches)")
    ap.add_argument("--out")
    a = ap.parse_args(); dev = torch.device(a.device); clip = build(a.ckpt, dev)
    if a.mode == "check":
        meta = pd.read_csv(ROOT / "flag_raw_s01_v3_adaface_simam100/metadata.csv").sort_values("row_id")
        tr = meta[meta.partition.eq("train")]; j = np.random.default_rng(0).choice(len(tr), a.n, replace=False)
        e = encode(clip, [ROOT / "data/train_set/train_set" / p for p in tr.face_relpath.iloc[j]], dev, a.tta)
        ref = np.load(a.arg, mmap_mode="r")[tr.row_id.to_numpy()[j]]; ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
        c = (e * ref).sum(1); print(f"CHECK cos vs {Path(a.arg).name}: mean {c.mean():.4f} min {c.min():.4f}")
    elif a.mode == "list":
        ps = pd.read_csv(a.arg).path.tolist(); print(f"{len(ps)} images, {a.ckpt} tta={a.tta}", flush=True)
        np.save(a.out, encode(clip, ps, dev, a.tta, turbo=a.turbo)); print("wrote", a.out, flush=True)
    else:
        rows = pd.read_csv(T3 / "caches/ext" / f"rows_{a.mode}.csv"); ps = [V1IMG / f for f in rows.face]
        ok = np.array([p.exists() for p in ps]); e = np.zeros((len(ps), clip.config.projection_dim), np.float32)
        print(f"{a.mode}: {ok.sum()}/{len(ok)} face images found; encoding with {a.ckpt} tta={a.tta}", flush=True)
        e[ok] = encode(clip, [p for p, k in zip(ps, ok) if k], dev, a.tta, turbo=a.turbo)  # 12/11: --turbo = Kaggle decoder of the LoRA caches
        np.save(T3 / "caches/ext" / f"{a.mode}_face_{a.arg}.float32.npy", e); np.save(T3 / "caches/ext" / f"{a.mode}_face_{a.arg}.ok.npy", ok)
        print("wrote", f"{a.mode}_face_{a.arg}", flush=True)
