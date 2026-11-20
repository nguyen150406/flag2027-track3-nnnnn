"""Voice-side fine-tuning, the mirror of t3_ft_clip.py (candidate 1, user-approved 2026-10-15): the 'sim' voice encoder (WeSpeaker
SimAM-ResNet100, VoxBlink2) with layer4 + ASP pooling + bottleneck trainable (conv1..layer3 frozen, every BatchNorm in eval mode, so
step 0 is exactly the frozen encoder), trained end-to-end with the SUB21a projector recipe (DualLinear 512, dropout face .70 / voice
.40, AAM per source + cross-modal SupCon + prototype alignment, same-gender SupCon on the FLAG batch) against FROZEN faces
(L2 farl (+) L2 clip, 1280-d). Sources: FLAG train (70 English identities) + vox2 / vb / lrs3 (one wav per rows-csv voice row).
No Bengali speech (organiser FAQ): FLAG train is English, vox2 Bengali speakers are dropped by t3data.ext_source (exclude_bengali.csv),
VoxBlink speakers labelled Bengali/Assamese were never extracted (official spk2lan), LRS3 is English TED.
Front-end = extract_wespeaker.py: int16-scale 16 kHz, Kaldi fbank 80 (25/10 ms, hamming, dither 0), mean norm over time.
Training: random 2 s crops (repeat-padded if shorter). Encoding: whole utterance in <=30 s chunks (>=1 s), chunk embeddings averaged, L2.
Speed (L4: 3 s crops were 4.96 s/step): channels_last + cudnn.benchmark for the fixed-size training crops, bf16 autocast everywhere on
cuda incl. encoding (same model and maths; ~1e-3 cosine numerics, identical for every encoded row).
Diagnostics (not a gate - proxies misled before): frozen-model parity vs the cached sim embeddings per source (row mapping / audio
decoding, must be ~1), and on 300 held-out VoxBlink identities (never trained on) the speaker-verification EER frozen vs fine-tuned and
the cross-modal EER of the joint model (compare with a --lr_enc 0 control run).
Outputs (<out>): <tag>.pt, flag_voice_<tag>.float32.npy (11349 META rows), <src>_voice_<tag>.{float32,ok}.npy (rows csv order), diag.json
Usage: python t3_ft_voice.py --out DIR --tag simfta --seed 2027 --roots vox2=DIR vb=DIR lrs3=DIR [--steps N] [--device cuda]
       python t3_ft_voice.py --out DIR --tag simfta --load DIR/simfta.pt --roots ...        (encode only)
Wav layout: vox2 / vb <root>/<clip_index>.wav, lrs3 <root>/<video>_<utt>.wav, FLAG under ROOT/data (META voice_relpath)."""
import argparse, json, os, sys, time, threading, queue, wave, types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torchaudio
import t3data as D
from t3lib import SPK, META, ROOT, GENDER
from t3train import aam_logits
from track2.model import DualLinearProjector, cross_modal_supervised_contrastive_loss, batch_identity_prototype_alignment_loss
_pkg = types.ModuleType("wespeaker"); _pkg.__path__ = [str(ROOT / ".tools/wespeaker/wespeaker")]; sys.modules["wespeaker"] = _pkg
_fe = types.ModuleType("wespeaker.frontend"); _fe.__path__ = [_pkg.__path__[0] + "/frontend"]; sys.modules["wespeaker.frontend"] = _fe
from wespeaker.models.speaker_model import get_speaker_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--roots", nargs="+", default=[])
ap.add_argument("--steps", type=int, default=0); ap.add_argument("--seed", type=int, default=2027); ap.add_argument("--device", default="cuda")
ap.add_argument("--lr_enc", type=float, default=3e-5, help="0 = control run (encoder frozen, only the joint projector trains)")
ap.add_argument("--unfreeze", type=int, default=4, choices=[3, 4], help="first trainable ResNet stage (4 = layer4 only)")
ap.add_argument("--sources", default="vox2,vb,lrs3"); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--crop", type=float, default=2.0, help="training crop seconds (2 s = 200 frames, the WeSpeaker training length)")
ap.add_argument("--heldout", type=int, default=300, help="identities of --heldout_src kept out of training for the diagnostics")
ap.add_argument("--heldout_src", default="vb")
ap.add_argument("--load", default="", help="encode only: trainable weights of an earlier run (<tag>.pt)")
ap.add_argument("--no_encode", action="store_true", help="diagnostics only (e.g. the --lr_enc 0 control): skip the FLAG / external encodes")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); dev = torch.device(a.device); T0 = time.time(); DIAG = {"args": vars(a)}
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m] {a.tag}:", m, flush=True)
ROOTS = {k: Path(v) for k, v in (r.split("=", 1) for r in a.roots)}
CFG = dict(dim=512, face_dropout=.70, voice_dropout=.40, margin=.1, scale=30., w_con=.5, temp=.2, w_proto=.6, w_sg=.5, ids=32, per_id=2, lr=3e-4, wd=1e-4)
def l2(x): x = np.asarray(x, np.float32); return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

# ------------------------------------------------------------------ audio front-end (identical to extract_wespeaker.py)
def read_wav(p):
    with wave.open(str(p)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, p
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    x = torch.from_numpy(x.astype(np.float32))[None]
    return torchaudio.functional.resample(x, sr, 16000) if sr != 16000 else x
def fbank(x):
    fb = torchaudio.compliance.kaldi.fbank(x, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_frequency=16000, window_type="hamming")
    return fb - fb.mean(0, keepdim=True)
def train_crop(p, seed):  # random a.crop seconds, repeat-padded when the utterance is shorter
    x = read_wav(p); n = int(a.crop * 16000)
    if x.shape[1] < n: x = x.repeat(1, n // x.shape[1] + 1)
    s = int(np.random.default_rng(seed).integers(0, x.shape[1] - n + 1)); return fbank(x[:, s:s + n])
def chunks(p):  # inference: <=30 s chunks, drop chunks under 1 s unless that is all there is
    x = read_wav(p); c = [x[:, s:s + 30 * 16000] for s in range(0, x.shape[1], 30 * 16000)]
    return [fbank(q) for q in ([q for q in c if q.shape[1] >= 16000] or [x])]
def flag_wav(r): return ROOT / "data/train_set/train_set" / r.voice_relpath if r.partition == "train" else ROOT / "data/dev_set" / r.protocol / r.voice_relpath
def ext_wav(src, rel, ci):
    if src in ("vox2", "vb"): return ROOTS[src] / f"{ci}.wav"
    if src == "lrs3": p = rel.split("/"); return ROOTS[src] / f"{p[1]}_{Path(p[2]).stem}.wav"
    raise ValueError(src)

# ------------------------------------------------------------------ SimAM-ResNet100 (VoxBlink2) = the cached 'sim' encoder
net = get_speaker_model("SimAM_ResNet100_ASP")(in_planes=64, embed_dim=256, acoustic_dim=80)
miss, _ = net.load_state_dict(torch.load(ROOT / ".tools/simam/unpacked/voxblink2_samresnet100/avg_model.pt", map_location="cpu"), strict=False)
assert not [k for k in miss if "projection" not in k], miss
net.to(dev).eval()  # eval for good: BatchNorm keeps the pretrained running statistics
if dev.type == "cuda": net.to(memory_format=torch.channels_last)  # NHWC convs on tensor cores (same maths)
for p in net.parameters(): p.requires_grad = False
f = net.front; TRAIN = ([f.layer3] if a.unfreeze == 3 else []) + [f.layer4, net.pooling, net.bottleneck]
enc = [p for m in TRAIN for p in m.parameters()]
for p in enc: p.requires_grad = a.lr_enc > 0
log(f"SimAM-ResNet100: {sum(p.numel() for p in enc) / 1e6:.1f}M trainable params from layer{a.unfreeze} (lr {a.lr_enc})")
amp = torch.bfloat16 if dev.type == "cuda" and torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16
use_amp = dev.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp == torch.float16)
def nchw(fb):  # (B, T, 80) fbank -> (B, 1, 80, T) network input
    x = fb.permute(0, 2, 1).unsqueeze(1); return x.contiguous(memory_format=torch.channels_last) if dev.type == "cuda" else x
def embed_train(fb):  # (B, T, 80) -> (B, 256); frozen stages without autograd
    x = nchw(fb)
    with torch.no_grad(), torch.autocast(dev.type, dtype=amp, enabled=use_amp):
        x = f.relu(f.bn1(f.conv1(x))); x = f.layer2(f.layer1(x))
        if a.unfreeze == 4: x = f.layer3(x)
    with torch.autocast(dev.type, dtype=amp, enabled=use_amp):
        if a.unfreeze == 3: x = f.layer3(x)
        x = net.bottleneck(net.pooling(f.layer4(x)))
    return x.float()
pool = ThreadPoolExecutor(a.workers)
@torch.no_grad()
def encode(paths, block=256, debug=True):  # fp32 inference, one utterance at a time (variable lengths), CPU front-end prefetched per block
    n = int(os.environ.get("T3_FT_DEBUG_N", "0")) if debug else 0  # smoke tests only: embed the first n, repeat them for the rest
    if n and len(paths) > n: return np.resize(encode(paths[:n], block), (len(paths), 256))
    out = np.zeros((len(paths), 256), np.float32); torch.backends.cudnn.benchmark = False  # variable lengths: no per-shape autotuning
    def one(fb):
        with torch.autocast(dev.type, dtype=amp, enabled=use_amp):
            x = nchw(fb[None].to(dev)); x = f.layer4(f.layer3(f.layer2(f.layer1(f.relu(f.bn1(f.conv1(x)))))))
            return net.bottleneck(net.pooling(x))[0].float()  # = net(fb[None]) with the channels_last input
    for s in range(0, len(paths), block):
        for i, fbs in enumerate(pool.map(chunks, paths[s:s + block])):
            e = torch.stack([one(fb) for fb in fbs]).mean(0); out[s + i] = F.normalize(e, dim=0).cpu().numpy()
        if (s // block) % 20 == 0: log(f"  encoded {min(s + block, len(paths))}/{len(paths)}")
    return out

# ------------------------------------------------------------------ data: FLAG train + external sources (frozen faces, wav paths)
TRM = META.sort_values("row_id"); TRM = TRM[TRM.partition.eq("train")]; assert len(TRM) == 6485 == len(SPK)
flag_wavs = [flag_wav(r) for r in TRM.itertuples()]; assert all(p.exists() for p in flag_wavs), "FLAG train wavs missing"
Fflag = np.hstack([l2(D.face("farl")[:6485]), l2(D.face("clip")[:6485])])
ids = sorted(set(SPK)); idx = {s: i for i, s in enumerate(ids)}; by_id = {s: np.flatnonzero(SPK == s) for s in ids}
male = [s for s in ids if GENDER.get(s) == "m"]; female = [s for s in ids if GENDER.get(s) == "f"]
rng0 = np.random.default_rng(0); SRC, HELD = [], None
for s in a.sources.split(","):
    xa, xc = D.ext_source(s, "farl", "sim"), D.ext_source(s, "clip", "sim")
    assert (xa["face_rows"] == xc["face_rows"]).all() and (xa["voice_rows"] == xc["voice_rows"]).all(), s
    rc = pd.read_csv(D.EXT / f"rows_{s}.csv"); vr = xa["voice_rows"]
    vw = [ext_wav(s, rc.voice.iat[i], rc.clip_index.iat[i]) for i in vr]; vk = np.array([p.exists() for p in vw])
    if vk.sum() == 0: log(f"source {s}: no wavs found, skipped"); continue
    fl, vl = xa["face_labels"], xa["labels"][vk]; labs = np.intersect1d(np.unique(fl), np.unique(vl))
    src = dict(name=s, faces=np.hstack([l2(xa["faces"]), l2(xc["faces"])]), fl=fl, vl=vl, wavs=[p for p, k in zip(vw, vk) if k],
               cached=l2(xa["voices"][vk]), n=int(xa["labels"].max()) + 1)
    if s == a.heldout_src and a.heldout:  # unseen identities for the diagnostics: never sampled in training
        nw = pd.Series([str(q) for q in src["wavs"]]).groupby(vl).nunique(); multi = np.intersect1d(labs, nw.index[nw >= 2].to_numpy())
        HELD = dict(src=src, labs=np.sort(rng0.choice(multi, min(a.heldout, len(multi)), replace=False))); labs = np.setdiff1d(labs, HELD["labs"])
    src.update(labs=labs, fg={int(c): np.flatnonzero(fl == c) for c in labs}, vg={int(c): np.flatnonzero(vl == c) for c in labs}); SRC.append(src)
    log(f"source {s}: {len(labs)} training identities, {len(fl)} face rows, {vk.sum()}/{len(vk)} voice rows with a wav")

@torch.no_grad()
def check(name, paths, cached, k=32):  # frozen model vs the cached sim embedding of the same row: decoding / row-mapping check
    j = np.random.default_rng(1).choice(len(paths), min(k, len(paths)), replace=False)
    cs = (encode([paths[i] for i in j], debug=False) * cached[j]).sum(1); DIAG[f"parity_{name}"] = [float(cs.mean()), float(cs.min())]
    log(f"check {name}: cos(frozen model, cached sim) mean {cs.mean():.4f} min {cs.min():.4f}")
    assert cs.mean() > .95, f"{name}: wavs do not match the cached sim embeddings (wrong rows or decoding)"
if not a.load:  # the trained weights would (rightly) move away from the cached embeddings
    check("flag", flag_wavs, l2(D.voice("sim")[:6485]))
    for s in SRC: check(s["name"], s["wavs"], s["cached"])

def eer(pos, neg):
    t = np.sort(np.concatenate([pos, neg])); fr = np.searchsorted(np.sort(pos), t) / len(pos); fa = 1 - np.searchsorted(np.sort(neg), t) / len(neg)
    i = np.argmin(np.abs(fr - fa)); return float(100 * (fr[i] + fa[i]) / 2)
def held_pairs():  # one cross-video positive voice pair and one negative per held-out identity (same draw for every run)
    s, r = HELD["src"], np.random.default_rng(2); P, N = [], []
    for c in HELD["labs"]:
        v = np.flatnonzero(s["vl"] == c); w = np.array([str(x) for x in np.array(s["wavs"], object)[v]])
        i = v[0]; j = v[np.flatnonzero(w != w[0])[0]]; o = r.choice(HELD["labs"][HELD["labs"] != c])
        P.append((i, j)); N.append((i, int(r.choice(np.flatnonzero(s["vl"] == o)))))
    return np.array(P), np.array(N)
def diagnose(stage, proj=None):
    if HELD is None: return
    s = HELD["src"]; P, N = held_pairs(); rows = np.unique(np.concatenate([P.ravel(), N.ravel()]))
    E = np.zeros((len(s["vl"]), 256), np.float32); E[rows] = encode([s["wavs"][i] for i in rows], debug=False)
    sv = lambda X: eer((X[P[:, 0]] * X[P[:, 1]]).sum(1), (X[N[:, 0]] * X[N[:, 1]]).sum(1))
    d = {"sv_eer_frozen_cached": sv(s["cached"]), "sv_eer_model": sv(E)}
    if proj is not None:  # cross-modal on unseen identities: voice row vs a face of the same / another identity
        r = np.random.default_rng(3); fpos = [int(r.choice(np.flatnonzero(s["fl"] == s["vl"][i]))) for i in P[:, 0]]
        fneg = [int(r.choice(np.flatnonzero(s["fl"] == s["vl"][j]))) for j in N[:, 1]]
        with torch.no_grad():
            fe, ve = proj(torch.from_numpy(s["faces"][fpos + fneg]).to(dev), torch.from_numpy(E[np.concatenate([P[:, 0], N[:, 0]])]).to(dev))
            c = (F.normalize(fe, dim=1) * F.normalize(ve, dim=1)).sum(1).cpu().numpy()
        d["xmodal_eer_joint"] = eer(c[:len(P)], c[len(P):])
    DIAG[f"heldout_{stage}"] = d; log(f"held-out {a.heldout_src} ({len(HELD['labs'])} ids) {stage}: " + " ".join(f"{k} {v:.2f}" for k, v in d.items()))
if not a.load: diagnose("step0")

# ------------------------------------------------------------------ joint model, heads, optimiser (as t3_ft_clip.py)
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
proj = DualLinearProjector(Fflag.shape[1], 256, CFG["dim"], CFG["face_dropout"], CFG["voice_dropout"]).to(dev)
W = nn.Parameter((torch.randn(len(ids), CFG["dim"]) * .01).to(dev))
heads = [nn.Parameter((torch.randn(s["n"], CFG["dim"]) * .01).to(dev)) for s in SRC]
gmask = torch.tensor([GENDER.get(s) == "m" for s in ids]).to(dev)
groups = [{"params": list(proj.parameters()) + [W] + heads, "lr": CFG["lr"], "weight_decay": CFG["wd"]}]
if a.lr_enc > 0: groups.append({"params": enc, "lr": a.lr_enc, "weight_decay": 0.0})
opt = torch.optim.AdamW(groups)
if a.load:
    ck = torch.load(a.load, map_location="cpu", weights_only=False); assert ck["unfreeze"] == a.unfreeze
    for m, sd in zip(TRAIN, ck["train"]): m.load_state_dict(sd)
    proj.load_state_dict(ck["proj"]); log(f"loaded {a.load}")
steps = 0 if a.load else (a.steps or 15 * (6485 // (CFG["ids"] * CFG["per_id"])))
def loss_on(fe, ve, lab, head):
    l = .5 * (F.cross_entropy(aam_logits(fe, head, lab, CFG["margin"], CFG["scale"]), lab) + F.cross_entropy(aam_logits(ve, head, lab, CFG["margin"], CFG["scale"]), lab))
    return l + CFG["w_con"] * cross_modal_supervised_contrastive_loss(fe, ve, lab, CFG["temp"]) + CFG["w_proto"] * batch_identity_prototype_alignment_loss(fe, ve, lab)
def plan():  # one step: FLAG gender-balanced 32 ids x 2 rows (paired face/voice rows), each source 32 ids x 2 (independent face / voice rows)
    half = CFG["ids"] // 2
    pick = list(rng.choice(male, min(half, len(male)), replace=False)) + list(rng.choice(female, min(half, len(female)), replace=False))
    b = np.concatenate([rng.choice(by_id[s], CFG["per_id"], replace=len(by_id[s]) < CFG["per_id"]) for s in pick])
    out = [("flag", [flag_wavs[r] for r in b], Fflag[b], [idx[s] for s in SPK[b]])]
    for s in SRC:
        ch = rng.choice(s["labs"], min(CFG["ids"], len(s["labs"])), replace=False); fb, vb, lb = [], [], []
        for c in ch:
            fb += list(rng.choice(s["fg"][int(c)], CFG["per_id"])); vb += list(rng.choice(s["vg"][int(c)], CFG["per_id"])); lb += [int(c)] * CFG["per_id"]
        out.append((s["name"], [s["wavs"][i] for i in vb], s["faces"][fb], lb))
    return out
SRCI = {s["name"]: i for i, s in enumerate(SRC)}; Q = queue.Queue(maxsize=4)
def producer():
    for _ in range(steps):
        Q.put([(n, torch.stack(list(pool.map(train_crop, ps, rng.integers(0, 2 ** 31, len(ps))))), fc, l) for n, ps, fc, l in plan()])
threading.Thread(target=producer, daemon=True).start()
proj.train(); t1 = time.time(); torch.backends.cudnn.benchmark = dev.type == "cuda"  # fixed crop shapes
for step in range(steps):
    batch = Q.get(); loss = 0.0
    for name, fb, fc, lab in batch:
        fe, ve = proj(torch.from_numpy(fc).to(dev), embed_train(fb.to(dev, non_blocking=True))); lt = torch.tensor(lab).to(dev)
        if name == "flag":
            loss = loss + loss_on(fe, ve, lt, W); gm = gmask[lt]
            for sel in (gm, ~gm):
                if sel.sum() > 2: loss = loss + CFG["w_sg"] * cross_modal_supervised_contrastive_loss(fe[sel], ve[sel], lt[sel], CFG["temp"])
        else:
            loss = loss + loss_on(fe, ve, lt, heads[SRCI[name]])
    opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 5.0); scaler.step(opt); scaler.update()
    assert torch.isfinite(loss), f"loss {float(loss)} at step {step}"
    if step % 50 == 0 or step == steps - 1: log(f"step {step}/{steps} loss {float(loss.detach()):.3f} ({(time.time() - t1) / (step + 1):.2f} s/step)")
proj.eval(); torch.backends.cudnn.benchmark = False
if steps: DIAG["final_loss"] = float(loss.detach())

# ------------------------------------------------------------------ outputs
if not a.load:
    torch.save({"train": [m.state_dict() for m in TRAIN], "proj": proj.state_dict(), "unfreeze": a.unfreeze, "seed": a.seed, "lr_enc": a.lr_enc,
                "encoder": "SimAM_ResNet100_ASP voxblink2", "sources": a.sources}, OUT / f"{a.tag}.pt")
    diagnose("final", proj)
if a.no_encode: json.dump(DIAG, open(OUT / f"diag_{a.tag}.json", "w"), indent=1, default=str); log("DONE (no encode)"); sys.exit(0)
allw = [flag_wav(r) for r in META.sort_values("row_id").itertuples()]; miss = [str(p) for p in allw if not p.exists()]
assert not miss, f"{len(miss)} FLAG wavs missing, e.g. {miss[:2]}"
uw = sorted(set(map(str, allw))); E = dict(zip(uw, encode([Path(p) for p in uw])))
np.save(OUT / f"flag_voice_{a.tag}.float32.npy", np.stack([E[str(p)] for p in allw]).astype(np.float32)); log("FLAG train+dev voices encoded")
for s in a.sources.split(","):
    rc = pd.read_csv(D.EXT / f"rows_{s}.csv"); ps = [ext_wav(s, v, c) for v, c in zip(rc.voice, rc.clip_index)]; ok = np.array([p.exists() for p in ps])
    uw = sorted({str(p) for p, k in zip(ps, ok) if k}); E = dict(zip(uw, encode([Path(p) for p in uw])))
    e = np.zeros((len(ps), 256), np.float32); e[ok] = np.stack([E[str(p)] for p, k in zip(ps, ok) if k])
    np.save(OUT / f"{s}_voice_{a.tag}.float32.npy", e); np.save(OUT / f"{s}_voice_{a.tag}.ok.npy", ok); log(f"{s} voices encoded {ok.sum()}/{len(ok)} rows ({len(uw)} files)")
json.dump(DIAG, open(OUT / f"diag_{a.tag}.json", "w"), indent=1, default=str); log("DONE")
