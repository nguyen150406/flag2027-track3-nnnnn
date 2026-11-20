"""IB-LoRA (16/11, prereg PREREG_gate_all.txt 20:10 "PREREG sub118a"): a second English system = ImageBind-huge with LoRA adapters trained for
face<->voice identity matching on the XJ external pool (MAV xjc set + VoxCeleb1, no FLAG people). Own code; ImageBind = public commit 53680b02 +
checkpoint as scripts/t3_imagebind_extract.py (same preprocessing for every scored item).
LoRA = weight parametrisation W + (alpha/r) * B @ A (B = 0 at start, so step 0 = frozen ImageBind) on MHA in_proj / out_proj and MLP fc1 / fc2 of
the last NV vision blocks and the last NA audio blocks, plus both head linears. Loss = symmetric InfoNCE face<->voice, one face + one voice per
identity per batch, learnable temperature; every second step a gender-pure batch. Checkpoint choice = best same-gender EER on held-out external ids.
  python t3_iblora.py selftest                                   (CPU: frozen parity, audio crop parity, LoRA merge equivalence)
  python t3_iblora.py train <items.csv> <outdir> [--steps N] [--smoke]
        items.csv: kind (face|voice), k (identity), gender (m|f|u), split (train|val|ctr... see builder), path, ref (row in the frozen-IB ref array or -1)
  python t3_iblora.py extract <lora.pt> <list.csv with 'path'> <face|voice> <out.npy> [--device mps]   (LoRA merged, fp32, our extractor)"""
import os, sys, math, time, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.utils.parametrize as PZ
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import t3_imagebind_extract as T
from imagebind.models.imagebind_model import ModalityType
VIS, AUD = ModalityType.VISION, ModalityType.AUDIO
T0 = time.time()
try:
    import cv2; cv2.setNumThreads(0)   # review 16/11: no OpenCV thread pool in the parent before DataLoader workers fork
except ImportError: pass


def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", m, flush=True)


# ------------------------------------------------------------------ LoRA
class LoRA(nn.Module):
    def __init__(self, shape, r, alpha):
        super().__init__(); o, i = shape
        self.A = nn.Parameter(torch.randn(r, i, dtype=torch.float32) / math.sqrt(i)); self.B = nn.Parameter(torch.zeros(o, r, dtype=torch.float32)); self.s = alpha / r

    def forward(self, W): return W + (self.s * (self.B @ self.A)).to(W.dtype)


def targets(m, nv, na):
    """(module, attribute) that get a LoRA, fixed order: per modality the last n blocks (attn in_proj, attn out_proj, fc1, fc2), then the head linear"""
    out = []
    for mod, n in ((VIS, nv), (AUD, na)):
        bl = list(m.modality_trunks[mod].blocks); assert 0 <= n <= len(bl)
        for b in bl[len(bl) - n:]:
            assert isinstance(b.attn, nn.MultiheadAttention) and b.attn._qkv_same_embed_dim
            out += [(b.attn, "in_proj_weight"), (b.attn.out_proj, "weight"), (b.mlp.fc1, "weight"), (b.mlp.fc2, "weight")]
        hd = m.modality_heads[mod][-1]; assert isinstance(hd, nn.Linear) and hd.bias is None; out.append((hd, "weight"))
    return out


def add_lora(m, cfg):
    torch.manual_seed(int(cfg.get("seed", 2027)))
    for mod_, attr in targets(m, cfg["nv"], cfg["na"]): PZ.register_parametrization(mod_, attr, LoRA(tuple(getattr(mod_, attr).shape), cfg["r"], cfg["alpha"]))
    return lora_mods(m, cfg)


def lora_mods(m, cfg): return [mod_.parametrizations[attr][0] for mod_, attr in targets(m, cfg["nv"], cfg["na"])]


def lora_state(mods): return {f"{i}.{n}": getattr(q, n).detach().float().cpu().clone() for i, q in enumerate(mods) for n in ("A", "B")}


def load_lora(mods, st):
    assert len(st) == 2 * len(mods), (len(st), len(mods))
    with torch.no_grad():
        for i, q in enumerate(mods):
            for n in ("A", "B"): getattr(q, n).copy_(st[f"{i}.{n}"])


def merge(m, cfg):
    for mod_, attr in targets(m, cfg["nv"], cfg["na"]): PZ.remove_parametrizations(mod_, attr, leave_parametrized=True)
    return m.requires_grad_(False)


def place(m, dv, da):
    for col in (m.modality_preprocessors, m.modality_trunks, m.modality_heads, m.modality_postprocessors): col[VIS].to(dv); col[AUD].to(da)
    return m


# ------------------------------------------------------------------ forward = ImageBindModel.forward for one modality (+ checkpointing of LoRA blocks)
def emb(m, mod, x, ck=False):
    red = x.ndim >= 5
    if red: B_, S_ = x.shape[:2]; x = x.reshape(B_ * S_, *x.shape[2:])
    v = m.modality_preprocessors[mod](**{mod: x}); tr = m.modality_trunks[mod]
    ids = [i for i, b in enumerate(tr.blocks) if any(isinstance(q, LoRA) for q in b.modules())] if ck else []
    t = tr(**v["trunk"], use_checkpoint=bool(ids), checkpoint_blk_ids=ids or None)
    h = F.normalize(m.modality_heads[mod](t, **v["head"]).float(), dim=-1)          # postprocessor = normalise (audio: x20, irrelevant after the next L2)
    if red: h = F.normalize(h.reshape(B_, S_, -1).mean(1), dim=-1)                  # clip mean, then our extractor's L2
    return h


# ------------------------------------------------------------------ inputs (scored items: T.vision_input / T.audio_input; training: random views)
def audio_from(path, starts=None, rng=None):
    """= T.audio_input with crop starts chosen by `starts(mx)` (default: T's fixed start / middle / end)"""
    import math as _m, soundfile as sf, torchaudio
    from scipy.signal import resample_poly
    x, sr = sf.read(str(path), dtype="float32", always_2d=True); a = x.mean(1)
    assert len(a) and np.isfinite(a).all(), path
    if sr != 16000: g = _m.gcd(int(sr), 16000); a = resample_poly(a, 16000 // g, int(sr) // g).astype("float32")
    if len(a) < 32000: a = np.pad(a, (0, 32000 - len(a)))
    mx = len(a) - 32000; st = (0, mx // 2, mx) if rng is None else sorted(int(s) for s in rng.integers(0, mx + 1, 3)); clips = []
    for s in st:
        w = torch.from_numpy(a[s:s + 32000].copy()).unsqueeze(0); w -= w.mean()
        mel = torchaudio.compliance.kaldi.fbank(w, htk_compat=True, sample_frequency=16000, use_energy=False, window_type="hanning", num_mel_bins=128,
                                                dither=0.0, frame_length=25, frame_shift=10).T
        mel = torch.nn.functional.pad(mel, (0, 204 - mel.shape[1])) if mel.shape[1] < 204 else mel[:, :204]
        clips.append((mel.unsqueeze(0) + 4.268) / 9.138)
    return torch.stack(clips)


def vision_train(path, rng):
    import cv2
    from PIL import Image
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB); h, w = img.shape[:2]; sc = rng.uniform(.8, 1.0); ch, cw = max(1, int(h * sc)), max(1, int(w * sc))
    y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1)); img = img[y0:y0 + ch, x0:x0 + cw]
    if rng.random() < .5: img = img[:, ::-1]
    return T.VT(Image.fromarray(np.ascontiguousarray(img)))


class StepDS(torch.utils.data.Dataset):
    """item s = the whole batch of step s (faces P x 3 x 224 x 224, voices P x 3 x 1 x 128 x 204), views seeded by s"""
    def __init__(self, fp, vp, plan, seed): self.fp, self.vp, self.plan, self.seed = fp, vp, plan, seed
    def __len__(self): return len(self.plan)
    def __getitem__(self, s):
        rng = np.random.default_rng(self.seed * 100003 + 7 * s + 1)
        return torch.stack([vision_train(self.fp[i], rng) for i, _ in self.plan[s]]), torch.stack([audio_from(self.vp[j], rng=rng) for _, j in self.plan[s]])


def eer(y, s):
    from sklearn.metrics import roc_curve
    y = np.asarray(y).astype(int)
    if y.min() == y.max(): return float('nan')
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return 100 * float(fpr[i] + fnr[i]) / 2


@torch.no_grad()
def embed_items(m, paths, kind, dev, amp, bs=32):
    fn, mod, out = (T.vision_input, VIS, []) if kind == "face" else (T.audio_input, AUD, [])
    for i in range(0, len(paths), bs):
        x = torch.stack([fn(p) for p in paths[i:i + bs]]).to(dev)
        with torch.autocast("cuda", torch.float16, enabled=amp): out.append(emb(m, mod, x).float().cpu())
    return torch.cat(out).numpy()


def train(items, out, steps=1600, P_=64, lr=2e-4, val_every=200, nw=4, seed=2027, smoke=False, ref=None):
    import pandas as pd
    out = Path(out); out.mkdir(parents=True, exist_ok=True); L = pd.read_csv(items, keep_default_na=False)
    cfg = dict(r=8, alpha=16, nv=8, na=12, seed=seed, P=P_, steps=steps, lr=lr, wd=.01, warmup=100, temp0=.05, gpure="odd steps", ext="MAV xjc + VoxCeleb1, no FLAG")
    if smoke: cfg.update(nv=1, na=1, P=4, steps=steps, warmup=1); P_ = 4
    cuda = torch.cuda.is_available() and not smoke; ng = torch.cuda.device_count() if cuda else 0
    dv = torch.device("cuda:0") if cuda else torch.device("cpu"); da = torch.device("cuda:1") if ng >= 2 else dv
    tr = L[L.split.eq("train")]; fT, vT = tr[tr.kind.eq("face")], tr[tr.kind.eq("voice")]
    ks = sorted(set(fT.k) & set(vT.k)); fidx = {k: g.index.values for k, g in fT.groupby("k")}; vidx = {k: g.index.values for k, g in vT.groupby("k")}
    gk = {k: g for k, g in zip(tr.k, tr.gender)}; byg = {g: [k for k in ks if gk[k] == g] for g in ("m", "f")}
    log(f"train ids {len(ks)} (m {len(byg['m'])} f {len(byg['f'])}) faces {len(fT)} voices {len(vT)} | device {dv}/{da} gpus {ng} | cfg {cfg}")
    plan = []
    for s in range(steps):
        rng = np.random.default_rng(seed * 1000003 + s)
        if s % 2 == 1 and min(len(byg["m"]), len(byg["f"])) >= P_:
            g = "m" if rng.random() < len(byg["m"]) / (len(byg["m"]) + len(byg["f"])) else "f"; pool = byg[g]
        else: pool = ks
        kk = rng.choice(pool, P_, replace=False); plan.append([(int(rng.choice(fidx[k])), int(rng.choice(vidx[k]))) for k in kk])
    paths = L.path.tolist()
    # ---- model
    m = T.model(torch.device("cpu"))
    if cuda: m.half()
    mods = add_lora(m, cfg); lp = [p for q in mods for p in (q.A, q.B)]; place(m, dv, da)
    log(f"LoRA modules {len(mods)} params {sum(p.numel() for p in lp) / 1e6:.2f}M | trainable total {sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6:.2f}M")
    assert sum(p.numel() for p in m.parameters() if p.requires_grad) == sum(p.numel() for p in lp), "only LoRA may train"
    lt = nn.Parameter(torch.tensor(math.log(1 / cfg["temp0"]), device=dv))
    opt = torch.optim.AdamW([{"params": lp, "weight_decay": cfg["wd"]}, {"params": [lt], "weight_decay": 0.0}], lr=lr)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    # ---- validation sets (fixed items, our extractor preprocessing)
    V = {sp: L[L.split.eq(sp)] for sp in ("val", "ctr")}
    vf, vv = V["val"][V["val"].kind.eq("face")], V["val"][V["val"].kind.eq("voice")]; cf, cv = V["ctr"][V["ctr"].kind.eq("face")], V["ctr"][V["ctr"].kind.eq("voice")]
    def validate(s):
        m.eval(); t = time.time()
        Fe, Ve = embed_items(m, vf.path.tolist(), "face", dv, cuda), embed_items(m, vv.path.tolist(), "voice", da, cuda)
        Fc, Vc = embed_items(m, cf.path.tolist(), "face", dv, cuda), embed_items(m, cv.path.tolist(), "voice", da, cuda)
        u = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True); S = u(Fe - Fc.mean(0)) @ u(Ve - Vc.mean(0)).T
        same = vf.k.values[:, None] == vv.k.values[None, :]; sg = vf.gender.values[:, None] == vv.gender.values[None, :]
        r = dict(step=s, val_ng=eer(same.ravel(), S.ravel()), val_g=eer(same[sg], S[sg]), n_pos=int(same.sum()), sec=round(time.time() - t, 1))
        return r, Fe, Ve
    hist, best = [], None
    r0, Fe0, Ve0 = validate(0); hist.append(r0); log(f"VAL step 0 (= frozen ImageBind) {r0}")
    if ref is not None:   # parity: step-0 embeddings of the val items vs our frozen xj1ib arrays (same files, same extractor)
        u = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
        pf = (u(Fe0[:len(ref["face"])]) * u(ref["face"])).sum(1); pv = (u(Ve0[:len(ref["voice"])]) * u(ref["voice"])).sum(1)   # first val items
        par = dict(face_mean=float(pf.mean()), face_min=float(pf.min()), voice_mean=float(pv.mean()), voice_min=float(pv.min())); log(f"PARITY step 0 vs frozen xj1ib arrays {par}")
        json.dump(par, open(out / "parity.json", "w")); assert pf.mean() > .99 and pv.mean() > .99, f"step-0 parity failed {par}"
    def save(tag, s, r, Fe, Ve):
        torch.save(dict(cfg=cfg, state=lora_state(mods), temp=float(lt.exp()), step=s, val=r), out / f"lora_{tag}.pt")
        np.save(out / f"val_face_{tag}.npy", Fe.astype(np.float32)); np.save(out / f"val_voice_{tag}.npy", Ve.astype(np.float32))
    dl = torch.utils.data.DataLoader(StepDS(paths, paths, plan, seed), batch_size=None, shuffle=False, num_workers=nw,
                                     prefetch_factor=3 if nw else None, persistent_workers=bool(nw), pin_memory=cuda)
    m.eval(); t0 = time.time(); run = []
    for s, (Fi, Vi) in enumerate(dl):
        with torch.autocast("cuda", torch.float16, enabled=cuda):
            f = emb(m, VIS, Fi.to(dv, non_blocking=True), ck=cuda); v = emb(m, AUD, Vi.to(da, non_blocking=True), ck=cuda).to(dv)
        lg = lt.exp().clamp(max=100) * (f.float() @ v.float().T); y = torch.arange(len(f), device=dv)
        loss = (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(lp, 1.0); scaler.step(opt); scaler.update(); sch.step()
        assert torch.isfinite(loss), f"loss not finite at step {s}"
        run.append((loss.item(), (lg.argmax(1) == y).float().mean().item()))
        if s % 20 == 0 or s == steps - 1:
            a = np.mean(run, 0); run = []
            log(f"step {s:5d} loss {a[0]:.4f} acc {a[1]:.3f} temp {1 / float(lt.exp()):.4f} lr {sch.get_last_lr()[0]:.2e} gnorm {float(gn):.3f} "
                f"{(time.time() - t0) / (s + 1):.2f}s/step" + (f" mem {torch.cuda.max_memory_allocated(dv) / 2 ** 30:.1f}/{torch.cuda.max_memory_allocated(da) / 2 ** 30:.1f}GB" if cuda else ""))
        if (s + 1) % val_every == 0 or s == steps - 1:
            r, Fe, Ve = validate(s + 1); hist.append(r); log(f"VAL step {s + 1} {r}")
            if best is None or r["val_g"] <= best["val_g"]: best = r; save("best", s + 1, r, Fe, Ve); log(f"  -> best (val_g {r['val_g']:.2f})")
            json.dump(dict(cfg=cfg, hist=hist, best=best), open(out / "val_hist.json", "w"), indent=1)
    save("last", steps, hist[-1], Fe, Ve)
    log(f"TRAIN DONE best {best} | step 0 {hist[0]}"); return hist, best


# ------------------------------------------------------------------ extraction with a trained LoRA (merged weights, fp32)
def load_merged(ck, dev):
    C = torch.load(ck, map_location="cpu", weights_only=True); cfg = C["cfg"]; m = T.model(torch.device("cpu"))
    mods = add_lora(m, cfg); load_lora(mods, C["state"]); merge(m, cfg)
    return m.eval().to(dev), C


def selftest():
    import pandas as pd
    sys.path.insert(0, str(HERE)); from t3lib import META
    tr = META[META.partition.eq("train")].sort_values("row_id").iloc[::1500]; base = T.T3.parent / "data/train_set/train_set"
    fp, vp = [base / p for p in tr.face_relpath], [base / p for p in tr.voice_relpath]; dev = torch.device("cpu")
    for p in vp: assert torch.equal(audio_from(p), T.audio_input(p)), "audio_from != T.audio_input"
    m = T.model(dev); ef0, ev0 = T.encode(m, fp, "face", dev), T.encode(m, vp, "voice", dev)
    with torch.no_grad():
        ef = emb(m, VIS, torch.stack([T.vision_input(p) for p in fp])).numpy(); ev = emb(m, AUD, torch.stack([T.audio_input(p) for p in vp])).numpy()
    print("frozen parity emb vs T.encode: face %.6f voice %.6f" % ((ef * ef0).sum(1).min(), (ev * ev0).sum(1).min())); assert (ef * ef0).sum(1).min() > .9999 and (ev * ev0).sum(1).min() > .9999
    cfg = dict(r=8, alpha=16, nv=2, na=2, seed=1); mods = add_lora(m, cfg)
    with torch.no_grad():
        for q in mods: q.B.normal_(0, .02)
        e1 = emb(m, VIS, torch.stack([T.vision_input(p) for p in fp]), ck=False).numpy(); a1 = emb(m, AUD, torch.stack([T.audio_input(p) for p in vp])).numpy()
    print("LoRA changed outputs: face cos to frozen %.4f voice %.4f" % ((e1 * ef0).sum(1).mean(), (a1 * ev0).sum(1).mean()))
    st = lora_state(mods); merge(m, cfg); e2, a2 = T.encode(m, fp, "face", dev), T.encode(m, vp, "voice", dev)
    print("merge equivalence: face %.6f voice %.6f" % ((e1 * e2).sum(1).min(), (a1 * a2).sum(1).min())); assert (e1 * e2).sum(1).min() > .9999 and (a1 * a2).sum(1).min() > .9999
    ck = Path("/tmp/iblora_selftest.pt"); torch.save(dict(cfg=cfg, state=st, temp=20.0, step=0, val={}), ck); m3, _ = load_merged(ck, dev); e3 = T.encode(m3, fp, "face", dev)
    print("save/load/merge: face %.6f" % (e1 * e3).sum(1).min()); assert (e1 * e3).sum(1).min() > .9999; ck.unlink(); print("SELFTEST OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "selftest": selftest()
    elif a[0] == "train":
        ref = None
        if "--ref" in a: z = np.load(a[a.index("--ref") + 1]); ref = dict(face=z["face"], voice=z["voice"])
        train(a[1], a[2], steps=int(a[a.index("--steps") + 1]) if "--steps" in a else 1600, smoke="--smoke" in a,
              nw=int(a[a.index("--nw") + 1]) if "--nw" in a else 4, val_every=int(a[a.index("--val") + 1]) if "--val" in a else 200, ref=ref)
    elif a[0] == "extract":
        import pandas as pd
        dev = torch.device(a[a.index("--device") + 1] if "--device" in a else "mps"); m, C = load_merged(a[1], dev)
        log(f"LoRA {a[1]} step {C['step']} val {C['val']}"); e = T.encode(m, pd.read_csv(a[2]).path.tolist(), a[3], dev); np.save(a[4], e.astype(np.float32)); log(f"wrote {a[4]} {e.shape}")
