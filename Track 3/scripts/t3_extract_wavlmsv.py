"""WavLM-SV voice embeddings ('wsv', user-approved 2026-10-15): microsoft/wavlm-base-plus-sv = WavLM Base+ (self-supervised on 94k h
of English: LibriLight, GigaSpeech, VoxPopuli-EN; no Bengali) with an x-vector head fine-tuned on VoxCeleb1 for speaker verification.
Pair-local: one file at a time, never batch-padded (the Track 2 WavLM cache bug), per-utterance normalisation by the model's feature
extractor, whole utterance in <=30 s chunks (>=1 s), chunk embeddings averaged, L2.
Outputs (<out>): flag_voice_wsv.float32.npy (11349 META rows), <src>_voice_wsv.{float32,ok}.npy (rows csv order), diag_wsv.json with
the speaker-verification EER on cross-video VoxBlink pairs (extraction sanity: a working SV model gives a few %).
--name wll (user-approved 2026-10-16): WavLM-Large + ECAPA-TDNN (UniSpeech wavlm_large_finetune, VoxCeleb1-O EER 0.431%), torch-native
mirror prj-beatrice/unispeech-wavlm-large-ecapa-tdnn-torch-native pinned at 13ad4fcc (safetensors; the official Google-Drive file is
over its download quota); WavLM Large is pretrained on English only, the SV head on VoxCeleb2. 256-d, raw waveform (the model layer-norms it).
--name rdn (2026-10-18 overnight): ReDimNet b6 (IDRnD, torch.hub IDRnD/ReDimNet, train_type ft_lm, dataset vox2 = VoxCeleb2 only,
VoxCeleb1-O EER ~0.4%), 192-d, raw 16 kHz waveform (the model computes its own features).
--name whs (user-approved 2026-11-03): frozen Whisper encoder openai/whisper-large-v3-turbo @41f01f3f, hidden state 2 (2nd block),
mean | std over the frames that cover real audio, each <=30 s chunk padded to Whisper's fixed 30 s window (t3_whisper_screen.py), 2560-d.
RF reproduction (2026-11-05, the encoders of TP): --name r221 = WeSpeaker ResNet221-LM (VoxCeleb, 256-d, t3extract recipe: balanced
<=30 s segments, fbank 80, CMN); sbe = SpeechBrain spkrec-ecapa-voxceleb (192-d, <=30 s chunks averaged); wl3 = Whisper large-v3
encoder, LAST layer, mean over the frames of real audio in the FIRST 30 s (1280-d); iba / ibl = ImageBind-huge audio tower frozen /
with the RF LoRA (--model lora.pt from t3_imagebind.py train), three fixed 2-s clips, 1024-d.
Usage: python t3_extract_wavlmsv.py --out DIR --flag_data <dir holding train_set/ and dev_set/> --roots vox2=DIR vb=DIR [--device cuda] [--name wsv|wll]
Wav layout: vox2 / vb <root>/<clip_index>.wav (as t3_ft_voice.py); T3_FT_DEBUG_N=n encodes only the first n files of each set (smoke)."""
import argparse, json, os, time, wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F, torchaudio
import t3data as D
from t3lib import META

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--flag_data", required=True); ap.add_argument("--roots", nargs="+", default=[])
ap.add_argument("--sources", default="vox2,vb"); ap.add_argument("--device", default="cuda"); ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--model", default=""); ap.add_argument("--name", default="wsv", choices=("wsv", "wll", "rdn", "whs", "r221", "sbe", "wl3", "iba", "ibl"))
ap.add_argument("--skip_flag", action="store_true", help="external sources only (FLAG voices already extracted)")
ap.add_argument("--shard", default="0/1", help="k/n: encode only every n-th external file starting at k (outputs <src>_voice_<name>.s<k>.*; merge = sum of arrays, OR of ok)")
ap.add_argument("--list", default="", help="csv with a 'voice' column: encode exactly these files in order -> <out>/list_voice_<name>.float32.npy, nothing else (eval phase)")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True); dev = torch.device(a.device); T0 = time.time(); FD = Path(a.flag_data)
ROOTS = {k: Path(v) for k, v in (r.split("=", 1) for r in a.roots)}; ONE = None
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m] {a.name}:", m, flush=True)
if a.name == "wsv":
    from transformers import AutoFeatureExtractor, WavLMForXVector
    a.model = a.model or "microsoft/wavlm-base-plus-sv"; DIM = 512
    fe = AutoFeatureExtractor.from_pretrained(a.model); model = WavLMForXVector.from_pretrained(a.model).to(dev).eval()
    emb = lambda x: model(input_values=x.to(dev)).embeddings[0]
elif a.name == "rdn":
    a.model = a.model or "IDRnD/ReDimNet b6 ft_lm vox2"; DIM = 192; fe = None
    model = torch.hub.load("IDRnD/ReDimNet", "ReDimNet", model_name="b6", train_type="ft_lm", dataset="vox2", trust_repo=True).to(dev).eval()
    emb = lambda x: model(x.to(dev))[0]
elif a.name == "whs":
    from transformers import WhisperFeatureExtractor, WhisperModel
    a.model = a.model or "openai/whisper-large-v3-turbo"; DIM = 2560; WREV = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    wfe = WhisperFeatureExtractor.from_pretrained(a.model, revision=WREV)
    wenc = WhisperModel.from_pretrained(a.model, revision=WREV).encoder.to(dev).eval()
    if dev.type != "cpu": wenc = wenc.half()
    fe = None
    def chunk_in(q): return wfe(q, sampling_rate=16000, return_tensors="pt").input_features, min(1500, -(-len(q) // 320))  # frames with real audio
    def emb(x):
        f, k = x; h = wenc(f.to(dev, next(wenc.parameters()).dtype), output_hidden_states=True).hidden_states[2][0, :k].float()
        return torch.cat([h.mean(0), h.std(0) if k > 1 else torch.zeros_like(h[0])])
elif a.name == "r221":
    os.environ.setdefault("T3_DEVICE", a.device); import t3extract
    a.model = a.model or t3extract.HF_SPK["r221"]; DIM = 256; fe = None; ONE = t3extract._wespeaker("r221")
elif a.name == "sbe":
    from speechbrain.inference.speaker import EncoderClassifier
    a.model = a.model or "speechbrain/spkrec-ecapa-voxceleb"; DIM = 192; fe = None
    sbm = EncoderClassifier.from_hparams(source=a.model, savedir=str(Path.home() / ".cache/sb_ecapa"), run_opts={"device": a.device})
    emb = lambda x: sbm.encode_batch(x.to(dev))[0, 0]
elif a.name == "wl3":
    from transformers import WhisperFeatureExtractor, WhisperModel
    a.model = a.model or "openai/whisper-large-v3"; DIM = 1280; WREV = "06f233fe06e710322aca913c1bc4249a0d71fce1"
    wfe = WhisperFeatureExtractor.from_pretrained(a.model, revision=WREV); wenc = WhisperModel.from_pretrained(a.model, revision=WREV).encoder.to(dev).eval()
    if dev.type != "cpu": wenc = wenc.half()
    fe = None
    def chunk_in(q): return wfe(q, sampling_rate=16000, return_tensors="pt").input_features, min(1500, -(-len(q) // 320))
    def emb(x):
        f, k = x; return wenc(f.to(dev, next(wenc.parameters()).dtype)).last_hidden_state[0, :k].float().mean(0)
elif a.name in ("iba", "ibl"):
    import t3_imagebind as IB
    ibm = IB.load(dev, (IB.AUDIO,)); DIM = 1024; fe = None
    if a.name == "ibl": assert a.model, "--model lora.pt required"; IB.load_lora(ibm, a.model); ibm.eval()
    else: a.model = f"ImageBind-huge @{IB.COMMIT}"
    emb = lambda x: IB.embed(ibm, IB.AUDIO, x[None].to(dev))[0]
else:
    from transformers import AutoModel
    a.model = a.model or "prj-beatrice/unispeech-wavlm-large-ecapa-tdnn-torch-native"; DIM = 256; fe = None
    model = AutoModel.from_pretrained(a.model, revision=None if Path(a.model).exists() else "13ad4fcc740cbd47992bcc2aaa59186d4072b7c9", trust_remote_code=True).to(dev).eval()
    emb = lambda x: model(x[0].to(dev), sampling_rate=16000).embeddings[0]
amp = dev.type == "cuda" and a.name != "sbe"  # fp16 autocast (T4); SpeechBrain ECAPA stays fp32; the x-vector statistics pooling runs on short sequences, fine in half precision

def read(p):  # float waveform in [-1, 1] at 16 kHz
    with wave.open(str(p)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, p
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return torchaudio.functional.resample(torch.from_numpy(x)[None], sr, 16000)[0].numpy() if sr != 16000 else x
def prep(p):  # <=30 s chunks, drop chunks under 1 s unless that is all there is; the feature extractor normalises each chunk
    x = read(p); c = [x[s:s + 30 * 16000] for s in range(0, len(x), 30 * 16000)]
    c = [q for q in c if len(q) >= 16000] or [x]
    if a.name == "whs": return [chunk_in(q) for q in c]
    if a.name == "wl3": return [chunk_in(c[0])]  # first 30 s only
    if a.name in ("iba", "ibl"): return [IB.audio_input(x)]  # whole clip -> three fixed 2-s clips
    return [fe(q, sampling_rate=16000, return_tensors="pt").input_values if fe else torch.from_numpy(q)[None] for q in c]
pool = ThreadPoolExecutor(a.workers)
@torch.no_grad()
def encode(paths, block=256):
    n = int(os.environ.get("T3_FT_DEBUG_N", "0"))
    if n and len(paths) > n: return np.resize(encode(paths[:n], block), (len(paths), DIM))
    out = np.zeros((len(paths), DIM), np.float32)
    if ONE:  # encoders with their own file reader (WeSpeaker via t3extract)
        for i, p in enumerate(paths):
            out[i] = ONE(p)
            if i % 5000 == 0: log(f"  encoded {i + 1}/{len(paths)}")
        return out
    for s in range(0, len(paths), block):
        for i, xs in enumerate(pool.map(prep, paths[s:s + block])):
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                e = torch.stack([emb(x).float() for x in xs]).mean(0)
            out[s + i] = F.normalize(e, dim=0).cpu().numpy()
        if (s // block) % 20 == 0: log(f"  encoded {min(s + block, len(paths))}/{len(paths)}")
    return out

def flag_wav(r): return FD / "train_set/train_set" / r.voice_relpath if r.partition == "train" else FD / "dev_set" / r.protocol / r.voice_relpath
def ext_wav(src, r):  # vox2 / vb / avs / avs2: <root>/<clip_index>.wav; lrs3: <root>/<video>_<utt>.wav; vox (no clip_index): <root>/<voice relpath>
    if src == "lrs3": p = r.voice.split("/"); return ROOTS[src] / f"{p[1]}_{Path(p[2]).stem}.wav"
    return ROOTS[src] / (f"{r.clip_index}.wav" if hasattr(r, "clip_index") else r.voice)
if a.list:
    ps = pd.read_csv(a.list).voice.astype(str).tolist(); uw = sorted(set(ps)); E = dict(zip(uw, encode([Path(p) for p in uw])))
    np.save(OUT / f"list_voice_{a.name}.float32.npy", np.stack([E[p] for p in ps]).astype(np.float32)); log(f"list encoded ({len(uw)} files)"); raise SystemExit
if not a.skip_flag:
    allw =[flag_wav(r) for r in META.sort_values("row_id").itertuples()]; miss = [str(p) for p in allw if not p.exists()]
    assert not miss, f"{len(miss)} FLAG wavs missing, e.g. {miss[:2]}"
    uw = sorted(set(map(str, allw))); E = dict(zip(uw, encode([Path(p) for p in uw])))
    np.save(OUT / f"flag_voice_{a.name}.float32.npy", np.stack([E[str(p)] for p in allw]).astype(np.float32)); log(f"FLAG train+dev encoded ({len(uw)} files)")
DIAG = {"model": a.model}
for s in [x for x in a.sources.split(",") if x]:
    rc = pd.read_csv(D.EXT / f"rows_{s}.csv"); ps = [ext_wav(s, r) for r in rc.itertuples()]; ok = np.array([p.exists() for p in ps])
    sk, sn = map(int, a.shard.split("/")); uw = sorted({str(p) for p, k in zip(ps, ok) if k})[sk::sn]; ok &= np.isin([str(p) for p in ps], uw)
    E = dict(zip(uw, encode([Path(p) for p in uw])))
    e = np.zeros((len(ps), DIM), np.float32); e[ok] = np.stack([E[str(p)] for p, k in zip(ps, ok) if k])
    sfx = "" if sn == 1 else f".s{sk}"
    np.save(OUT / f"{s}_voice_{a.name}{sfx}.float32.npy", e); np.save(OUT / f"{s}_voice_{a.name}{sfx}.ok.npy", ok); log(f"{s} voices encoded {ok.sum()}/{len(ok)} rows ({len(uw)} files, shard {a.shard})")
    if s == "vb":  # sanity: SV EER on cross-video pairs of 300 identities (one positive + one negative pair each)
        r = np.random.default_rng(0); g = pd.DataFrame(dict(id=rc.identity, w=[str(p) for p in ps], ok=ok))[ok].drop_duplicates("w")
        ids = [i for i, n in g.groupby("id").w.nunique().items() if n >= 2]; ids = list(r.choice(ids, min(300, len(ids)), replace=False))
        first = {i: g.index[g.id == i][:2] for i in ids}; P = [(first[i][0], first[i][1]) for i in ids]
        N = [(first[i][0], first[ids[(k + 1 + r.integers(len(ids) - 1)) % len(ids)]][0]) for k, i in enumerate(ids)]
        sc = lambda pr: np.array([float(e[x] @ e[y]) for x, y in pr]); pos, neg = sc(P), sc(N)
        t = np.sort(np.concatenate([pos, neg])); fr = np.searchsorted(np.sort(pos), t) / len(pos); fa = 1 - np.searchsorted(np.sort(neg), t) / len(neg)
        j = np.argmin(np.abs(fr - fa)); DIAG["vb_sv_eer"] = float(100 * (fr[j] + fa[j]) / 2); log(f"sanity: VoxBlink cross-video SV EER {DIAG['vb_sv_eer']:.2f}% ({len(ids)} ids)")
json.dump(DIAG, open(OUT / f"diag_{a.name}.json", "w"), indent=1); log("DONE")
