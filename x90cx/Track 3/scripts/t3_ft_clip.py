"""Face-side fine-tuning for the clip:cat4 branch: CLIP ViT-L/14 with LoRA on the last K blocks, trained end-to-end with the
SUB21a projector recipe (DualLinear 512, dropout face .70 / voice .40, AAM per source + cross-modal SupCon + prototype alignment,
same-gender SupCon on the FLAG batch). Voice side = cached frozen cat4 embeddings. External sources: images located through
IMG_ROOTS; rows without an image are skipped.
  --fold 0..3 : fit on that fold's fit identities, write OOF branch scores for the canonical CV cells -> <out>/oof_fold<f>.pkl
  --fold -1   : fit on all 70 train identities, save LoRA+projector and the fine-tuned CLIP embeddings of every FLAG row
                (train+dev) and every external face row -> <out>/<enc>ft_*.npy (input for the normal projector pipeline)
  --encoder farl : same recipe on FaRL ViT-B/16 (HF conversion by t3_farl_to_hf.py, .tools/farl/farl_hf) -> farlft_*
  --encoder siglip : same recipe on SigLIP2 so400m/14 @224 (google/siglip2-so400m-patch14-224, or $T3_SIGLIP), MAP-head output
                   1152-d; no cached frozen SigLIP exists -> external row tables come from the CLIP caches and the image<->row
                   check is an identity check (same-ID cos > different-ID cos) instead of a cache comparison
Usage: python t3_ft_clip.py --fold 0 --out DIR --roots flag=DIR vox=DIR,DIR vox2=DIR vb=DIR lrs3=DIR [--steps N] [--device cuda]"""
import argparse, io, os, pickle, time, threading, queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageFilter
import t3data as D
from t3lib import SPK, META, ROOT, fold_ids, cell, GENDER, GROUPS
from t3train import aam_logits
from track2.model import DualLinearProjector, cross_modal_supervised_contrastive_loss, batch_identity_prototype_alignment_loss

ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--roots", nargs="+", default=[])
ap.add_argument("--steps", type=int, default=0); ap.add_argument("--k", type=int, default=6); ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr_lora", type=float, default=1e-4); ap.add_argument("--seed", type=int, default=2027); ap.add_argument("--device", default="cuda")
ap.add_argument("--sources", default="vox,vox2,vb,lrs3"); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--proj", default="sim,ecapa,sam100ft,r293,cat4", help="fold>=0: after fine-tuning, re-encode all faces and run the normal "
                "3-seed projector CV (t3train, recipe from T3_* env as in t3_j_par) for these clip:<voice> branches; empty = skip")
ap.add_argument("--seeds", default="2027,42,3407")
ap.add_argument("--encoder", default="clip", choices=["clip", "farl", "siglip"])
ap.add_argument("--no_flag", action="store_true", help="fine-tune on the external sources only (no FLAG identities): FLAG train faces "
                "stay unseen by the encoder, like dev faces -> no in-sample stacking bias in the projectors trained on them")
ap.add_argument("--tag", default="", help="output tag (default <encoder>ft): <tag>.pt, flag_face_<tag>, <src>_face_<tag>")
ap.add_argument("--domain", default="", help="csv (src, identity): these identities of the external sources also form an extra "
                "source 'sa' with its own head and a batch every step (domain oversampling, e.g. South-Asian speakers)")
ap.add_argument("--nocache", default="", help="15/11 (90c-X): comma list of external sources without a frozen-encoder face cache (e.g. mav): "
                "their image<->row check is the identity check (same-ID cos > different-ID cos) instead of the cache comparison")
ap.add_argument("--xlang", default="", help="15/11 (90c-X, option A): comma list of external sources whose identities with voices in >= 2 "
                "languages draw their per_id voices from DIFFERENT languages (forced cross-language positives); others sample as before")
ap.add_argument("--load", default="", help="encode only: load the LoRA weights of an earlier --fold -1 checkpoint, no training")
ap.add_argument("--alpha", type=float, default=1.0, help="with --load: WiSE-FT blend, the LoRA update B@A is scaled by alpha (0 = frozen CLIP, 1 = the LoRA); research direction A3, 2026-11-04")
ap.add_argument("--no_ft", action="store_true", help="control: skip fine-tuning (frozen CLIP), still re-encode every face in this run")
ap.add_argument("--aug", type=float, default=0.0, help="training-time degradation probability per face (0 = off): down-up "
                "scaling, Gaussian blur, JPEG re-encode (FLAG faces are upscaled, re-compressed YouTube crops); encoding stays plain")
ap.add_argument("--holdout", type=int, default=-1, help="cross-fit (with --fold -1): leave FLAG identity group 0..4 (t3lib.GROUPS, 14 IDs)\n"
                "out of the LoRA training; its train rows then get out-of-fold embeddings from this LoRA")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); dev = torch.device(a.device); T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m] fold {a.fold}:", m, flush=True)
ROOTS = {k: [Path(p) for p in v.split(",")] for k, v in (r.split("=", 1) for r in a.roots)}
CFG = dict(dim=512, face_dropout=.70, voice_dropout=.40, margin=.1, scale=30., w_con=.5, temp=.2, w_proto=.6, w_sg=.5, ids=32, per_id=2, lr=3e-4, wd=1e-4)

def find(src, rel):  # image path of a rows-csv face path (or FLAG relpath) under the configured roots, else None
    for r in ROOTS.get(src, []):
        for p in (r / rel, r / src / rel, r / "train" / rel, r / "test" / rel, r / "dev" / rel):
            if p.exists(): return p
    return None

# ------------------------------------------------------------------ CLIP + LoRA (zero-init: starts exactly as the frozen encoder)
class LoRA(nn.Module):
    def __init__(self, base, r):
        super().__init__(); self.base = base; self.A = nn.Parameter(torch.randn(r, base.in_features) / r); self.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x): return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))
ENC = a.encoder; FT = a.tag or ENC + "ft"  # output tag: clipft / farlft / clipnf / siglipft
BASE = "clip" if ENC == "siglip" else ENC  # cached face arrays that define the external row tables (siglip has none of its own)
if ENC == "siglip":
    from transformers import SiglipVisionModel
    clip = SiglipVisionModel.from_pretrained(os.environ.get("T3_SIGLIP", "google/siglip2-so400m-patch14-224")).to(dev)
    FD = clip.config.hidden_size  # 1152 (so400m): the attention-pooling (MAP) head output = SigLIP image embedding
else:
    from transformers import CLIPVisionModelWithProjection
    clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14" if ENC == "clip" else str(ROOT / ".tools/farl/farl_hf")).to(dev)
    FD = clip.config.projection_dim  # 768 CLIP ViT-L / 512 FaRL
for p in clip.parameters(): p.requires_grad = False
VM = getattr(clip, "vision_model", clip)  # transformers 5 flattened SiglipVisionModel (no .vision_model); 4.x and CLIP keep it
for layer in VM.encoder.layers[-a.k:]:
    at, mlp = layer.self_attn, layer.mlp
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"): setattr(at, name, LoRA(getattr(at, name), a.rank).to(dev))
    for name in ("fc1", "fc2"): setattr(mlp, name, LoRA(getattr(mlp, name), a.rank).to(dev))
clip.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
lora = [p for p in clip.parameters() if p.requires_grad]
MEAN, STD = ((torch.full((3, 1, 1), .5), torch.full((3, 1, 1), .5)) if ENC == "siglip" else  # SigLIP processor: (x - .5) / .5
             (torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1), torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)))
def load(path, flip=False, aug=None):  # CLIP preprocessing: shorter side 224 (bicubic), centre crop 224, normalise
    im = Image.open(path).convert("RGB"); w, h = im.size; s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC); w, h = im.size
    im = im.crop(((w - 224) // 2, (h - 224) // 2, (w - 224) // 2 + 224, (h - 224) // 2 + 224))
    if aug:  # (scale, jpeg quality, blur radius), each None = skipped; training batches only
        sc, q, r = aug
        if sc: m = round(224 * sc); im = im.resize((m, m), Image.BICUBIC).resize((224, 224), Image.BICUBIC)
        if r: im = im.filter(ImageFilter.GaussianBlur(r))
        if q: b = io.BytesIO(); im.save(b, "JPEG", quality=q); im = Image.open(b).convert("RGB")
    if flip: im = im.transpose(Image.FLIP_LEFT_RIGHT)
    x = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)
    return (x - MEAN) / STD
# bf16 only with native support (compute capability >= 8): torch.cuda.is_bf16_supported() is True on a T4 through slow emulation
amp = torch.bfloat16 if dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and amp == torch.float16)  # T4: fp16 autocast -> loss scaling
def embed(px):
    with torch.autocast(dev.type, dtype=amp, enabled=dev.type == "cuda"):
        o = clip(pixel_values=px); e = o.pooler_output if ENC == "siglip" else o.image_embeds
    return F.normalize(e.float(), dim=1)

# ------------------------------------------------------------------ data: FLAG (fit identities) + external sources (face image paths)
Vflag = np.asarray(D.voice_full("cat4"), np.float32)
fit = np.unique(SPK) if a.fold < 0 else np.array(fold_ids(a.fold)[0])
if a.holdout >= 0:  # cross-fit LoRA: train on the other 4 groups (56 IDs)
    assert a.fold < 0; fit = np.array(sorted(set(SPK) - set(GROUPS[a.holdout])))
rows = np.flatnonzero(np.isin(SPK, fit))
TR = META[META.partition.eq("train")].sort_values("manifest_order")
flag_img = [find("flag", rp) for rp in TR.face_relpath]
rows = np.array([r for r in rows if flag_img[r] is not None]); ids = sorted(set(SPK[rows])); idx = {s: i for i, s in enumerate(ids)}
by_id = {s: rows[SPK[rows] == s] for s in ids}
male = [s for s in ids if GENDER.get(s) == "m"]; female = [s for s in ids if GENDER.get(s) == "f"]
log(f"FLAG fit identities {len(ids)}, rows {len(rows)} (images found {sum(p is not None for p in flag_img)}/{len(flag_img)})")
SRC = []
for s in a.sources.split(","):
    x = D.ext_source(s, BASE, "cat4"); rcsv = pd.read_csv(D.EXT / f"rows_{s}.csv")
    paths = [find(s, f) for f in rcsv.face.iloc[x["face_rows"]]]
    keep = np.array([p is not None for p in paths])
    if keep.sum() == 0: log(f"source {s}: no images found, skipped"); continue
    fl = x["face_labels"][keep]; fp = [p for p, k in zip(paths, keep) if k]
    labs = np.unique(np.intersect1d(fl, x["labels"]))
    fg = {int(c): np.flatnonzero(fl == c) for c in labs}; vg = {int(c): np.flatnonzero(x["labels"] == c) for c in labs}
    SRC.append(dict(name=s, paths=fp, fg=fg, vg=vg, labs=labs, V=np.asarray(x["voices"], np.float32), n=int(x["labels"].max()) + 1, fl=fl,
                    cached=np.asarray(x["faces"], np.float32)[keep]))
    log(f"source {s}: {len(labs)} identities, {keep.sum()}/{len(keep)} face rows with images")
    if s in a.xlang.split(","):   # option A: voice indices of each identity grouped by language
        vl = np.asarray(x["voice_lang"]).astype(str); SRC[-1]["vlg"] = {c: {g: vg[c][vl[vg[c]] == g] for g in np.unique(vl[vg[c]])} for c in vg}
        log(f"source {s}: forced cross-language voice pairs for {sum(len(d) >= CFG['per_id'] for d in SRC[-1]['vlg'].values())}/{len(labs)} identities "
            f"(languages {pd.Series(vl).value_counts().to_dict()})")
if a.domain:  # extra source "sa": the listed identities (already in vox2/vb...) get their own AAM head and a batch every step
    dom = pd.read_csv(a.domain); lab = {}; P, Lf, Vs, Lv, C = [], [], [], [], []
    for s in dom.src.unique():
        want = set(dom.identity[dom.src.eq(s)].astype(str))
        x = D.ext_source(s, BASE, "cat4"); rcsv = pd.read_csv(D.EXT / f"rows_{s}.csv"); fn, vn = x["face_names"].astype(str), x["voice_names"].astype(str)
        fi = np.flatnonzero(np.isin(fn, list(want))); paths = [find(s, f) for f in rcsv.face.iloc[x["face_rows"][fi]]]
        fi = np.array([i for i, q in zip(fi, paths) if q is not None], int); paths = [q for q in paths if q is not None]
        vi = np.flatnonzero(np.isin(vn, list(set(fn[fi]))))
        for n in sorted(set(fn[fi]) & set(vn[vi])): lab.setdefault(n, len(lab))
        fk = np.array([n in lab for n in fn[fi]], bool); vk = np.array([n in lab for n in vn[vi]], bool)
        P += [q for q, k in zip(paths, fk) if k]; Lf += [lab[n] for n in fn[fi][fk]]
        Vs.append(np.asarray(x["voices"], np.float32)[vi[vk]]); Lv += [lab[n] for n in vn[vi][vk]]
        C.append(np.asarray(x["faces"], np.float32)[fi[fk]])
    Lf, Lv = np.array(Lf), np.array(Lv); labs = np.unique(Lf)
    SRC.append(dict(name="sa", domain=True, paths=P, fg={int(c): np.flatnonzero(Lf == c) for c in labs}, vg={int(c): np.flatnonzero(Lv == c) for c in labs},
                    labs=labs, V=np.vstack(Vs), n=len(lab), cached=np.vstack(C), fl=Lf))
    log(f"domain source sa ({a.domain}): {len(labs)} identities, {len(P)} face rows, {len(Lv)} voice rows")

@torch.no_grad()
def check(name, paths, cached):
    j = np.random.default_rng(0).choice(len(paths), min(48, len(paths)), replace=False)
    clip.eval(); e = embed(torch.stack([load(paths[i]) for i in j]).to(dev)).cpu().numpy()
    c = cached[j] / np.linalg.norm(cached[j], axis=1, keepdims=True); cs = (e * c).sum(1)
    log(f"check {name}: cos(zero-init model, cached CLIP) mean {cs.mean():.4f} min {cs.min():.4f}")
    # CLIP ViT-L moves to cos ~.92 under +-1 grey-level noise (a different JPEG decoder / re-encoded frame gives ~.9-.95);
    # only a wrong person/row mapping (cos ~.3-.6) should stop the run
    assert cs.mean() > .75, f"{name}: images do not match the cached embeddings"
@torch.no_grad()
def check_id(name, paths, labels):  # encoders without a cache: frozen same-identity faces must be closer than different-identity faces
    r = np.random.default_rng(0); labels = np.asarray(labels); u, c = np.unique(labels, return_counts=True)
    j = np.concatenate([r.choice(np.flatnonzero(labels == l), 2, replace=False) for l in r.choice(u[c >= 2], min(24, (c >= 2).sum()), replace=False)])
    clip.eval(); e = embed(torch.stack([load(paths[i]) for i in j]).to(dev)).cpu().numpy(); S = e @ e.T; L = labels[j]
    same = (L[:, None] == L[None]) & ~np.eye(len(j), dtype=bool); diff = L[:, None] != L[None]
    log(f"check {name}: frozen {ENC} cos same identity {S[same].mean():.4f} vs different {S[diff].mean():.4f} ({len(j)} faces)")
    assert S[same].mean() > S[diff].mean() + .02 or os.environ.get("T3_IDCHECK") == "0", f"{name}: images do not look row-aligned"
if ENC == "siglip":
    check_id("flag", [flag_img[r] for r in rows], SPK[rows])
    for s in SRC: s.pop("cached"); check_id(s["name"], s["paths"], s["fl"])
else:
    check("flag", [flag_img[r] for r in rows], np.asarray(D.face(ENC)[:6485], np.float32)[rows])
    NOC = set(filter(None, a.nocache.split(",")))
    for s in SRC:
        if s["name"] in NOC: s.pop("cached"); check_id(s["name"], s["paths"], s["fl"])
        else: check(s["name"], s["paths"], s.pop("cached"))
# --load: the image<->row checks above ran on the zero-init model; now put the trained LoRA in place (encode only)
if a.load:
    ck = torch.load(a.load, map_location="cpu", weights_only=False); assert ck["k"] == a.k and ck["rank"] == a.rank and ck.get("encoder", "clip") == ENC
    sd = {(("vision_model." if VM is not clip else "") + k.removeprefix("vision_model.") if ENC == "siglip" else k): v for k, v in ck["lora"].items()}
    bad = clip.load_state_dict(sd, strict=False).unexpected_keys; assert not bad and len(sd) == 12 * a.k, bad
    if a.alpha != 1.0:  # W = W0 + alpha * B @ A (weight-space interpolation between frozen CLIP and the fine-tuned model)
        nb = 0
        for m in clip.modules():
            if isinstance(m, LoRA): m.B.data.mul_(a.alpha); nb += 1
        assert nb == 6 * a.k, nb; log(f"WiSE-FT: LoRA updates of {nb} layers scaled by alpha {a.alpha}")

# ------------------------------------------------------------------ model, heads, optimiser
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
proj = DualLinearProjector(FD, Vflag.shape[1], CFG["dim"], CFG["face_dropout"], CFG["voice_dropout"]).to(dev)
W = nn.Parameter((torch.randn(len(ids), CFG["dim"]) * .01).to(dev))
heads = [nn.Parameter((torch.randn(s["n"], CFG["dim"]) * .01).to(dev)) for s in SRC]
gmask = torch.tensor([GENDER.get(s) == "m" for s in ids]).to(dev)
opt = torch.optim.AdamW([{"params": list(proj.parameters()) + [W] + heads, "lr": CFG["lr"], "weight_decay": CFG["wd"]},
                         {"params": lora, "lr": a.lr_lora, "weight_decay": 0.0}])
steps = 0 if (a.no_ft or a.load) else (a.steps or 15 * max(1, len(rows) // (CFG["ids"] * CFG["per_id"])))
def loss_on(fe, ve, lab, head):
    l = .5 * (F.cross_entropy(aam_logits(fe, head, lab, CFG["margin"], CFG["scale"]), lab) + F.cross_entropy(aam_logits(ve, head, lab, CFG["margin"], CFG["scale"]), lab))
    return l + CFG["w_con"] * cross_modal_supervised_contrastive_loss(fe, ve, lab, CFG["temp"]) + CFG["w_proto"] * batch_identity_prototype_alignment_loss(fe, ve, lab)

def plan():  # one step's sample lists (same sampling scheme as t3train)
    half = CFG["ids"] // 2
    pick = list(rng.choice(male, min(half, len(male)), replace=False)) + list(rng.choice(female, min(half, len(female)), replace=False))
    b = np.concatenate([rng.choice(by_id[s], CFG["per_id"], replace=len(by_id[s]) < CFG["per_id"]) for s in pick])
    out = [] if a.no_flag else [("flag", [flag_img[r] for r in b], Vflag[b], [idx[s] for s in SPK[b]])]
    for s in SRC:
        ch = rng.choice(s["labs"], min(CFG["ids"], len(s["labs"])), replace=False); fb, vb, lb = [], [], []
        for c in ch:
            fb += list(rng.choice(s["fg"][int(c)], CFG["per_id"])); lb += [int(c)] * CFG["per_id"]
            d = s.get("vlg", {}).get(int(c))
            if d is not None and len(d) >= CFG["per_id"]: vb += [int(rng.choice(d[g])) for g in rng.choice(sorted(d), CFG["per_id"], replace=False)]
            else: vb += list(rng.choice(s["vg"][int(c)], CFG["per_id"]))
        out.append((s["name"], [s["paths"][i] for i in fb], s["V"][vb], lb))
    return out
SRCI = {s["name"]: i for i, s in enumerate(SRC)}
arng = np.random.default_rng(a.seed + 7919)  # own stream: with --aug the batch plan and flips stay identical to the run without it
def augs(n):
    if not a.aug: return [None] * n
    return [None if u[0] >= a.aug else (.35 + .45 * u[2] if u[1] < .5 else None, int(30 + 60 * u[4]) if u[3] < .5 else None, .3 + .9 * u[6] if u[5] < .25 else None)
            for u in arng.random((n, 7)).tolist()]  # Python floats: Pillow+numpy 2 GaussianBlur fails on np.float64 radii
pool = ThreadPoolExecutor(a.workers); Q = queue.Queue(maxsize=3)
def producer():  # a loader error is handed to the training loop (raised there) instead of leaving it blocked on Q.get() forever
    try:
        for _ in range(steps):
            P = plan(); Q.put([(n, torch.stack(list(pool.map(lambda pf: load(*pf), zip(ps, rng.random(len(ps)) < .5, augs(len(ps)))))), v, l) for n, ps, v, l in P])
    except BaseException as e: Q.put(e)
threading.Thread(target=producer, daemon=True).start()

proj.train(); clip.train()
for step in range(steps):
    batch = Q.get(); loss = 0.0
    if isinstance(batch, BaseException): raise batch
    for name, px, v, lab in batch:
        fe, ve = proj(embed(px.to(dev, non_blocking=True)), torch.from_numpy(v).to(dev)); lt = torch.tensor(lab).to(dev)
        if name == "flag":
            loss = loss + loss_on(fe, ve, lt, W)
            gm = gmask[lt]
            for sel in (gm, ~gm):
                if sel.sum() > 2: loss = loss + CFG["w_sg"] * cross_modal_supervised_contrastive_loss(fe[sel], ve[sel], lt[sel], CFG["temp"])
        else:
            loss = loss + loss_on(fe, ve, lt, heads[SRCI[name]])
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(list(proj.parameters()) + [W] + heads + lora, 5.0); scaler.step(opt); scaler.update()
    if step % 50 == 0 or step == steps - 1: log(f"step {step}/{steps} loss {float(loss):.3f}")

# ------------------------------------------------------------------ outputs
proj.eval(); clip.eval()
@torch.no_grad()
def encode(paths, bs=128):
    n = int(os.environ.get("T3_FT_DEBUG_N", "0"))  # smoke tests only: embed the first n paths, repeat them for the rest
    if n and len(paths) > n: return np.resize(encode(paths[:n], bs), (len(paths), FD))
    out = []
    for s in range(0, len(paths), bs):
        out.append(embed(torch.stack(list(pool.map(load, paths[s:s + bs]))).to(dev)).cpu())
        if (s // bs) % 25 == 0: log(f"  encoded {s + len(out[-1])}/{len(paths)}")
    return torch.cat(out).numpy()
def encode_ext():  # fine-tuned embeddings of every face row of every source rows csv (+ which rows have an image)
    E = {}
    for s in [s for s in SRC if not s.get("domain")]:  # the domain source's rows are encoded with their own source
        rcsv = pd.read_csv(D.EXT / f"rows_{s['name']}.csv"); ps = [find(s["name"], f) for f in rcsv.face]; ok = np.array([p is not None for p in ps])
        e = np.zeros((len(ps), FD), np.float32); e[ok] = encode([p for p in ps if p is not None]); E[s["name"]] = (e, ok)
        log(f"{s['name']} encoded {ok.sum()}/{len(ok)}")
    return E
@torch.no_grad()
def scores(fr):
    fe = encode([flag_img[r] for r in fr.face_row]); f, v = proj(torch.from_numpy(fe).to(dev), torch.from_numpy(Vflag[fr.voice_row.to_numpy()]).to(dev))
    return (F.normalize(f, dim=1) * F.normalize(v, dim=1)).sum(1).cpu().numpy().astype(np.float64)
if a.fold >= 0:
    if not a.no_ft:
        oof = {(a.fold, p): scores(cell(a.fold, p)[0]) for p in ("standard", "same_gender")}
        pickle.dump({"clip_cat4": oof}, open(OUT / f"oof_fold{a.fold}_joint.pkl", "wb")); log("joint-projector OOF written")
    if a.proj:  # the deployable path: fine-tuned faces -> normal projector recipe, 3 seeds, same sources as the baseline CV
        del opt, heads, W; torch.cuda.empty_cache()
        assert all(p is not None for p in flag_img), "FLAG train images missing"
        Ff = encode(flag_img); E = encode_ext(); log("all faces re-encoded")
        np.save(OUT / f"flag_train_{FT}_fold{a.fold}.float16.npy", Ff.astype(np.float16))
        from t3train import train_projector, pair_cos
        from t3_j_par import kw_from_env
        KW = kw_from_env(); log(f"projector recipe {KW}, device {os.environ.get('T3_TRAIN_DEVICE', 'cpu')}")
        fit_rows = np.flatnonzero(np.isin(SPK, fit)); res = {}
        for vn in a.proj.split(","):
            V = np.asarray(D.voice_full(vn)[:6485], np.float32); aux = []
            for s in [s for s in SRC if not s.get("domain")]:
                x = D.ext_source(s["name"], BASE, vn); e, ok = E[s["name"]]; k = ok[x["face_rows"]]
                x = {key: (val[k] if key.startswith("face") else val) for key, val in x.items()}
                vk = np.isin(x["labels"], np.unique(x["face_labels"]))  # identities that lost every face image
                x = {key: (val[vk] if key.startswith("voice") or key == "labels" else val) for key, val in x.items()}
                x["faces"] = e[x["face_rows"]]; aux.append(x)
            log(f"{ENC}_{vn}: external face rows {[len(x['faces']) for x in aux]}, voice rows {[len(x['voices']) for x in aux]}")
            outs = []
            for seed in map(int, a.seeds.split(",")):
                m = train_projector(Ff, V, fit_rows, SPK, seed=seed, aux=aux, **KW)
                outs.append({p: pair_cos(m, Ff, V, cell(a.fold, p)[0].face_row.to_numpy(), cell(a.fold, p)[0].voice_row.to_numpy())
                             for p in ("standard", "same_gender")})
                log(f"{ENC}_{vn} seed {seed} trained")
            res[f"{ENC}_{vn}"] = {(a.fold, p): np.mean([o[p] for o in outs], 0) for p in ("standard", "same_gender")}
            from t3lib import eer
            log(f"RESULT {'ctl' if a.no_ft else 'ft'} {ENC}_{vn} fold {a.fold} branch EER (no age/gender): "
                + " ".join(f"{p} {eer(cell(a.fold, p)[0], res[f'{ENC}_{vn}'][a.fold, p]):.2f}" for p in ("standard", "same_gender")))
            pickle.dump(res, open(OUT / f"oof_fold{a.fold}_proj.pkl", "wb"))
        log("projector OOF written")
else:
    torch.save({"lora": {(k.removeprefix("vision_model.") if ENC == "siglip" else k): v for k, v in clip.state_dict().items() if k.endswith((".A", ".B"))}, "proj": proj.state_dict(), "k": a.k, "rank": a.rank, "encoder": ENC, "no_flag": a.no_flag, "holdout": a.holdout, "aug": a.aug}, OUT / f"{FT}.pt")
    allp = [find("flag", r.face_relpath) if r.partition == "train" else find("flagdev", f"{r.protocol}/{r.face_relpath}")
            for r in META.sort_values("row_id").itertuples()]  # dev relpaths repeat across protocols -> prefix the protocol
    miss = [i for i, p in enumerate(allp) if p is None]; assert not miss, f"{len(miss)} FLAG images missing"
    np.save(OUT / f"flag_face_{FT}.float32.npy", encode(allp).astype(np.float32)); log("FLAG train+dev encoded")
    for name, (e, ok) in encode_ext().items():
        np.save(OUT / f"{name}_face_{FT}.float32.npy", e); np.save(OUT / f"{name}_face_{FT}.ok.npy", ok)
log("DONE")
