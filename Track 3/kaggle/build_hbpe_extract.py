"""HB + PE-AV VOICE block (17/11 evening, user "thử hết đi"): Kaggle notebook ncamtu06/flag-t3-hbpe-extract. Frozen PE-AV (facebook/pe-av-base
rev 1c4c329b, Apache-2.0, public; weights sha256 c87718fb...) audio embeddings (audio_model head = the audio side of PE-AV's audio-video contrastive space)
of every HB pool clip -> caches/ext/*_voice_pe. FLAG / eval PE voices = the Mac files of the 17/11 PE-AV screen (scratchpad peav/, same extractor code:
transformers 5.1.0, manual state-dict load, 48 kHz resample_poly, centre crop 20 s, bs 1, fp32).
Audio: identical to ncamtu06/flag-t3-hbibv-extract (VoxCeleb2 via HF-signed links made at build time, ffmpeg 16 kHz mono; VoxBlink wav from vb_media.zip).
GATE (fail fast in each worker): two synthetic wavs (deterministic numpy, PCM_16) must reproduce the Mac CPU embeddings, min cos > .999.
Two workers (one per T4) encode the two halves of the clip list. Output hbpe.zip: pe_voice.f16.npy, hbpe_voices.csv (src, clip_index, good), parity json, PROVENANCE.
usage: python build_hbpe_extract.py"""
import os, json, hashlib, re, sys, shutil
from pathlib import Path
import numpy as np
K = Path(__file__).resolve().parent; sys.path.insert(0, str(K)); T3 = K.parent; CX = T3 / "caches/ext"
import build_xj1ib_extract as B
USER = "ncamtu06"; PEV = "/tmp/scratch/peav"; PESHA = "c87718fb93e00f3b2e5c886b3a6a589ee3bc3d29ea0c1a2e1f6b1e141c19170e"; PEREV = "1c4c329b78d80a18d860ebfb05daebff4bc44518"
SYN = r'''
def synth_wavs(d):
    import numpy as np, soundfile as sf
    out = []
    for i, dur in enumerate((3.0, 5.5)):
        rng = np.random.default_rng(1234 + i); t = np.arange(int(16000 * dur)) / 16000
        x = sum(.2 * np.sin(2 * np.pi * f * t + p) for f, p in zip(rng.uniform(90, 400, 5), rng.uniform(0, 6.28, 5))) + .05 * rng.standard_normal(len(t))
        x = (np.clip(x / np.abs(x).max() * .8, -1, 1) * 32767).astype(np.int16); p = f"{d}/synth{i}.wav"; sf.write(p, x, 16000, subtype="PCM_16"); out.append(p)
    return out
'''


def mac_refs():
    """PE-AV voice embeddings of the synthetic wavs on the Mac CPU with the screen's extractor (venv python: transformers 5.1.0, soundfile)"""
    import tempfile, subprocess
    d = tempfile.mkdtemp()
    code = SYN + ("\nimport sys, json, torch; sys.path.insert(0, %r); import peav_extract as P; ps = synth_wavs(%r); E = P.Enc(torch.device('cpu'));"
                  "print(json.dumps([E.voice(P.load_audio(p, 20))[0].tolist() for p in ps]))") % (PEV, d)
    out = subprocess.run([PEV + "/.venv/bin/python", "-c", code], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    return json.loads(out)


WORKER = r'''
import os, sys, time, json, math
from pathlib import Path
import numpy as np, pandas as pd, torch, soundfile as sf
from scipy.signal import resample_poly
from PIL import Image
from transformers import AutoProcessor, PeAudioVideoModel
PART, NPART = int(sys.argv[1]), int(sys.argv[2]); T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", f"w{PART}", m, flush=True)
W = Path("/kaggle/temp/hbpe"); OUT = Path("/kaggle/working/hbpe_out"); MID = str(W / "model"); SR = 48000
def load_audio(p, max_s):
    x, sr = sf.read(str(p), dtype="float32", always_2d=True); a = x.mean(1); assert len(a) and np.isfinite(a).all(), p
    if sr != SR: g = math.gcd(int(sr), SR); a = resample_poly(a, SR // g, int(sr) // g).astype("float32")
    if len(a) > max_s * SR: s = (len(a) - max_s * SR) // 2; a = a[s:s + max_s * SR]   # centre crop of long clips (deterministic)
    return a


def load_face(p, frames):
    im = np.asarray(Image.open(p).convert("RGB")); return np.repeat(im[None], frames, 0)   # (T, H, W, 3) static video


class Enc:
    def __init__(self, dev):
        # from_pretrained of transformers 5.1/5.23 breaks on this checkpoint (tied-weight bug 'bands'; timm head size) -> build from config and
        # load the state dict ourselves; the joint audio_video_encoder.embedder copies are filled from audio_model / video_model (= tying)
        from transformers import PeAudioVideoConfig
        from safetensors.torch import load_file
        self.dev = dev; self.proc = AutoProcessor.from_pretrained(MID); m = PeAudioVideoModel(PeAudioVideoConfig.from_pretrained(MID)).eval()
        sd = load_file(os.path.join(MID, "model.safetensors")); msd = m.state_dict()
        for k in msd:
            for a, b in (("audio_video_encoder.embedder.audio_encoder.", "audio_model.audio_encoder."), ("audio_video_encoder.embedder.video_encoder.", "video_model.video_encoder."),
                         ("audio_model.text_model.", "text_model."), ("video_model.text_model.", "text_model.")):
                if k.startswith(a) and k not in sd and b + k[len(a):] in sd: sd[k] = sd[b + k[len(a):]]
        assert all(sd[k].shape == msd[k].shape for k in sd if k in msd) and not [k for k in sd if k not in msd]
        need = [k for k in msd if k.startswith(("audio_model.audio_encoder", "audio_model.audio_head", "video_model.video_encoder", "video_model.video_head"))]
        miss = [k for k in need if k not in sd]; assert not miss, miss[:5]
        r = m.load_state_dict(sd, strict=False); self.unloaded = [k for k in r.missing_keys if not k.endswith(("bands", "inv_freq"))]
        self.m = m.float().to(dev)

    @torch.inference_mode()
    def voice(self, a):
        f = self.proc.feature_extractor([a], sampling_rate=SR, return_tensors="pt"); kw = {k: v.to(self.dev) for k, v in f.items()}
        e = self.m.get_audio_embeds(kw["input_values"], kw.get("padding_mask")); return e.float().cpu().numpy()

    @torch.inference_mode()
    def face(self, v):
        f = self.proc.video_processor([v], return_tensors="pt"); kw = {k: v_.to(self.dev) for k, v_ in f.items()}
        vm = self.m.video_model; o = vm.video_encoder(pixel_values_videos=kw["pixel_values_videos"], padding_mask_videos=kw.get("padding_mask_videos"), return_dict=True)
        e = vm.video_head(o.pooler_output); return e.float().cpu().numpy()   # PeVideoModel.get_video_features is nested in __init__ in transformers 5.x



if PART > 0:   # RAM: load after worker 0 has its model on the GPU
    t = time.time()
    while not (W / "w0_ready").exists() and time.time() - t < 1800: time.sleep(10)
dev = torch.device("cuda"); E = Enc(dev); log(f"model loaded, unloaded params {len(E.unloaded)}")
if PART == 0: (W / "w0_ready").touch()
u = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12); ref = np.array(json.load(open(W / "refs.json")), np.float32)
e = np.stack([E.voice(load_audio(str(W / f"synth{i}.wav"), 20))[0] for i in range(2)]); c = (u(e) * u(ref)).sum(1); log(f"PARITY vs Mac CPU on synthetic wavs: {c.round(6).tolist()}")
json.dump(dict(min=float(c.min())), open(OUT / f"parity_{PART}.json", "w")); assert c.min() > .999, f"parity failed {c}"
L = pd.read_csv(W / "voices.csv"); idx = np.array_split(np.arange(len(L)), NPART)[PART]; out = np.zeros((len(idx), 1024), np.float32); t = time.time()
for j, i in enumerate(idx):
    out[j] = E.voice(load_audio(L.path[i], 20))[0]
    if j % 1000 == 0: log(f"voice {j + 1}/{len(idx)} ({(j + 1) / max(time.time() - t, 1e-9):.2f}/s)")
assert np.isfinite(out).all(); np.save(OUT / f"pe_voice_{PART}.f16.npy", out.astype(np.float16)); log(f"SAVED {len(idx)} in {(time.time() - t) / 60:.1f} min")
'''

BODY = r'''try:
    import numpy as np
    import torch; ng = torch.cuda.device_count(); log(f"GPUs {ng}; cpus {os.cpu_count()}"); assert ng >= 1, "no GPU"
    for k_, u_ in list(URLS.items()) + [(f"hf{i}", h_) for i, h_ in enumerate(__HFURLS__)]:   # fail fast: temporary links still alive?
        r_ = requests.get(u_, headers={"Range": "bytes=0-3"}, timeout=120); log(f"link {k_}: HTTP {r_.status_code}"); assert r_.status_code == 206 and r_.content[:2] == b"PK", f"link {k_} dead"
    EXT = T3 / "caches/ext"; EXT.mkdir(parents=True, exist_ok=True); WB = Path("/kaggle/temp/hbpe"); WB.mkdir(parents=True, exist_ok=True)
    OB = Path("/kaggle/working/hbpe_out"); OB.mkdir(parents=True, exist_ok=True)
    for name, dst in (("vox2_manifest.csv", EXT), ("rows_vox2.csv", EXT), ("rows_vb.csv", EXT)):
        hit = [h for h in find(name) if "flag-t3-hbib-files" in str(h)]; assert hit, f"{name} missing"; shutil.copyfile(hit[0], dst / name)
    import hashlib as hl_
    open(EXT / "exclude_bengali.csv", "w").write("identity,src\n")   # fetch ALL rows_vox2 clips (HB applies LID + exclusion itself)
    sh("pip install -q transformers==5.1.0 timm==1.0.31 2>&1 | tail -2", "extract.log")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"; os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download
    snapshot_download("facebook/pe-av-base", revision="__PEREV__", local_dir=str(WB / "model"))
    h = hl_.sha256(open(WB / "model/model.safetensors", "rb").read()).hexdigest(); log(f"PE-AV weights sha {h[:16]}"); assert h == "__PESHA__"
    exec(r"""__SYN__"""); synth_wavs(str(WB)); json.dump(json.loads(r"""__REFS__"""), open(WB / "refs.json", "w"))
    log("STAGE setup done")
__VOX2__    fetch(URLS["vb_media.zip"], DL / "vb_media.zip")
    with zipfile.ZipFile(DL / "vb_media.zip") as z: z.extractall(TMP / "vbm", [m for m in z.namelist() if m.startswith("wav/")])
    shutil.move(str(TMP / "vbm/wav"), str(AUD / "vb")); (DL / "vb_media.zip").unlink(); shutil.rmtree(TMP / "vbm")
    import soundfile as sf_
    def wok(p):
        try: return sf_.info(p).frames > 0
        except Exception: return False
    rows = []
    for s in ("vox2", "vb"):
        for p in sorted((AUD / s).glob("*.wav"), key=lambda q: int(q.stem)): rows.append((s, int(p.stem), str(p)))
    import pandas as pd_
    V = pd_.DataFrame(rows, columns=["src", "clip_index", "path"]); V["good"] = [wok(p) for p in V.path]; sub = V.path[V.good].iloc[0]
    log(f"STAGE audio done: {len(V)} wav ({int((V.src == 'vox2').sum())} vox2), readable {int(V.good.sum())}"); assert V.good.mean() > .98
    V.assign(path=np.where(V.good, V.path, sub)).to_csv(WB / "voices.csv", index=False); V[["src", "clip_index", "good"]].to_csv(OB / "hbpe_voices.csv", index=False)
    open(WB / "worker.py", "w").write(r"""__WORKER__""")
    procs = []; NP_ = 2 if ng >= 2 else 1
    for g in range(NP_):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
        procs.append(subprocess.Popen([sys.executable, str(WB / "worker.py"), str(g), str(NP_)], stdout=open(LG / f"w_{g}.log", "w"), stderr=subprocess.STDOUT, env=env, cwd=str(WB)))
    while any(p.poll() is None for p in procs):
        time.sleep(120)
        if any(p.poll() not in (None, 0) for p in procs):
            for p in procs:
                if p.poll() is None: p.kill()
        for g in range(NP_):
            f = LG / f"w_{g}.log"
            if f.exists(): log(f"[w{g}] " + ((f.read_text().strip().splitlines() or ["-"])[-1])[-160:])
    for g in range(NP_): print(f"===== w{g}\n" + "\n".join((LG / f"w_{g}.log").read_text().splitlines()[-25:]), flush=True)
    assert all(p.returncode == 0 for p in procs), [p.returncode for p in procs]
    np.save(OB / "pe_voice.f16.npy", np.concatenate([np.load(OB / f"pe_voice_{g}.f16.npy") for g in range(NP_)])); [os.remove(OB / f"pe_voice_{g}.f16.npy") for g in range(NP_)]
    json.dump(dict(pair_local=True, labels_used=False, test_statistics=False, n_voices=len(V),
                   parity={g: json.load(open(OB / f"parity_{g}.json")) for g in range(NP_)},
                   model="facebook/pe-av-base@__PEREV__, model.safetensors sha256 __PESHA__, transformers 5.1.0 manual state-dict load, 48 kHz, centre crop 20 s, bs 1, fp32",
                   audio="vox2: HF mirror ProgramComputer/voxceleb mp4 -> ffmpeg 16 kHz mono; vb: vb_media.zip wav"), open(OB / "PROVENANCE.json", "w"), indent=1)
    with zipfile.ZipFile(OUT / "hbpe.zip", "w", zipfile.ZIP_STORED) as zz:
        for p in sorted(OB.iterdir()): zz.write(p, p.name)
    shutil.rmtree(OB); log(f"JOB DONE {(OUT / 'hbpe.zip').stat().st_size / 1e6:.0f} MB")
'''

if __name__ == "__main__":
    import types
    _w = (K / "build_t3_whs_kernel.py").read_text(); ns = {"__file__": str(K / "build_t3_whs_kernel.py")}
    exec(_w[:_w.index("BODY = r\'\'\'")], ns); exec(_w[_w.index("def links():"):_w.index("code = (HEAD")], ns)
    HEAD, TAIL, VOX2 = ns["HEAD"], ns["TAIL"], ns["VOX2"]
    assert compile(WORKER, "w", "exec") and '"""' not in WORKER and '"""' not in SYN
    refs = mac_refs(); print("mac refs done", [round(sum(x * x for x in r) ** .5, 4) for r in refs])
    body = BODY.replace("__VOX2__", VOX2)
    rep = {"__PEREV__": PEREV, "__PESHA__": PESHA, "__SYN__": SYN, "__REFS__": json.dumps(refs), "__WORKER__": WORKER}
    for a_, b_ in rep.items(): assert a_ in body, a_; body = body.replace(a_, b_)
    urls = {k: v for k, v in ns["links"]().items() if k == "vb_media.zip"}
    code = (HEAD + body + TAIL).replace("__URLS__", json.dumps(urls)).replace("__HFURLS__", json.dumps(ns["hf_urls"]()))
    code = code.replace("import os, sys, json, time", "import pandas as pd\nimport os, sys, json, time", 1)
    left = set(re.findall(r"__[A-Z0-9]+__", code)); assert not left, left
    compile(code, "hbpe", "exec")
    d = K / "hbpe_extract"; d.mkdir(exist_ok=True)
    json.dump({"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
               "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}, open(d / "hbpe_extract.ipynb", "w"))
    json.dump({"id": f"{USER}/flag-t3-hbpe-extract", "title": "FLAG T3 hbpe extract", "code_file": "hbpe_extract.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [f"{USER}/flag-t3-hbib-files"], "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
    print(f"built {d} | notebook {(d / 'hbpe_extract.ipynb').stat().st_size / 1e6:.2f} MB")
