"""HB + PE-AV FACE block (17/11 evening, user "thử hết đi" / "thử thêm các hướng tốt đột biến cho eng luôn"): Kaggle notebook trnlhvy/flag-t3-hbpef-extract.
Frozen PE-AV (facebook/pe-av-base rev 1c4c329b, public, weights sha256 c87718fb...) video-side embeddings of every HB pool face (still image as a 1-frame
video, = the 17/11 Mac PE-AV screen extractor: transformers 5.1.0 manual state-dict load, PIL RGB, bs 1, fp32) for VoxCeleb2 (vox2_faces.zip) and VoxBlink
(vb_media.zip img/) from trncnguyn kernel outputs via temporary links made at build time. FLAG / eval PE faces = the Mac screen files.
GATE (fail fast, each worker): 2 deterministic synthetic PNG faces must reproduce the Mac CPU embeddings (min cos > .999).
Output hbpef.zip: hbpef_face.f16.npy (N x 1024, raw), hbpef_face_names.csv (zip member names, same order), parity json, PROVENANCE.
usage: python build_hbpef_extract.py -> kaggle/hbpef_extract/ ; then push with the trnlhvy token at once."""
import os, json, hashlib, re, sys, gzip, base64
from pathlib import Path
import numpy as np, pandas as pd
K = Path(__file__).resolve().parent; sys.path.insert(0, str(K)); T3 = K.parent; CX = T3 / "caches/ext"
import build_xj1ib_extract as B   # COMMON helpers, ImageBind pins, extractor source
USER = "trnlhvy"; PEV = "/tmp/scratch/peav"; PESHA = "c87718fb93e00f3b2e5c886b3a6a589ee3bc3d29ea0c1a2e1f6b1e141c19170e"; PEREV = "1c4c329b78d80a18d860ebfb05daebff4bc44518"
COMMON = B.COMMON.replace('"/kaggle/temp/xj1ib"', '"/kaggle/temp/hbpef"').replace('"/kaggle/working/xj1ib_out"', '"/kaggle/working/hbpef_out"')
assert COMMON.count("hbpef") == 2
SYNI = r'''
def synth_faces(d):
    import numpy as np
    from PIL import Image
    out = []
    for i in range(2):
        rng = np.random.default_rng(4321 + i); yy, xx = np.mgrid[0:256, 0:256] / 255.
        im = np.stack([np.sin(6 * xx * (1 + i) + rng.uniform(0, 6)) * .5 + .5, np.cos(5 * yy + rng.uniform(0, 6)) * .5 + .5, (xx * yy + .2 * rng.random((256, 256)))], -1)
        p = f"{d}/synthface{i}.png"; Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8)).save(p); out.append(p)
    return out
'''


def mac_refs():
    import tempfile, subprocess
    d = tempfile.mkdtemp()
    code = SYNI + ("\nimport sys, json, torch; sys.path.insert(0, %r); import peav_extract as P; ps = synth_faces(%r); E = P.Enc(torch.device('cpu'));"
                   "print(json.dumps([E.face(P.load_face(p, 1))[0].tolist() for p in ps]))") % (PEV, d)
    out = subprocess.run([PEV + "/.venv/bin/python", "-c", code], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    return json.loads(out)


WORKER = r'''
import os, sys, time, json, math
from pathlib import Path
import numpy as np, pandas as pd, torch, soundfile as sf
from scipy.signal import resample_poly
from PIL import Image
from transformers import AutoProcessor, PeAudioVideoModel
SH = int(sys.argv[1]); T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", f"shard{SH}", m, flush=True)
W = Path("/kaggle/temp/hbpef"); OUT = Path("/kaggle/working/hbpef_out"); MID = str(W / "model"); SR = 48000
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



if SH > 0:
    t = time.time()
    while not (W / "w0_ready").exists() and time.time() - t < 1800: time.sleep(10)
dev = torch.device("cuda"); E = Enc(dev); log(f"model loaded, unloaded {len(E.unloaded)}")
if SH == 0: (W / "w0_ready").touch()
u = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12); ref = np.array(json.load(open(W / "refs.json")), np.float32)
e = np.stack([E.face(load_face(str(W / f"synthface{i}.png"), 1))[0] for i in range(2)]); c = (u(e) * u(ref)).sum(1); log(f"PARITY vs Mac CPU on synthetic faces: {c.round(6).tolist()}")
json.dump(dict(min=float(c.min())), open(OUT / f"parity_{SH}.json", "w")); assert c.min() > .999, f"parity failed {c}"
L = pd.read_csv(W / "faces.csv"); L = L.iloc[SH::2]; paths = L.path.tolist(); out = np.zeros((len(paths), 1024), np.float32); t = time.time()
for j, p in enumerate(paths):
    out[j] = E.face(load_face(p, 1))[0]
    if j % 2000 == 0: log(f"face {j + 1}/{len(paths)} ({(j + 1) / max(time.time() - t, 1e-9):.2f}/s)")
assert np.isfinite(out).all(); np.save(OUT / f"shard{SH}.f16.npy", out.astype(np.float16)); log(f"SAVED shard {out.shape} in {(time.time() - t) / 60:.1f} min")
'''

MAIN = r'''
try:
    import requests
    from concurrent.futures import ThreadPoolExecutor
    URLS = json.loads(r"""__URLS__""")
    for k, u_ in URLS.items():   # fail fast if the temporary links already expired (rebuild + re-push)
        r_ = requests.get(u_, headers={"Range": "bytes=0-3"}, timeout=120); log(f"link {k}: HTTP {r_.status_code} {r_.content[:2]!r}"); assert r_.status_code == 206 and r_.content[:2] == b"PK", f"link {k} dead"
    run("pip install -q transformers==5.1.0 timm==1.0.31 2>&1 | tail -2")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"; os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download
    snapshot_download("facebook/pe-av-base", revision="__PEREV__", local_dir=str(W / "model")); h = fsha(W / "model/model.safetensors"); log(f"PE-AV weights sha {h[:16]}"); assert h == "__PESHA__"
    exec(r"""__SYNI__"""); synth_faces(str(W)); json.dump(json.loads(r"""__REFS__"""), open(W / "refs.json", "w"))
    # ---- media (parallel range download, faces only)
    DL = Path("/kaggle/temp/dl"); DL.mkdir(exist_ok=True); MED = W / "media"; MED.mkdir(exist_ok=True)
    def fetch(url, out, n=8, chunk=64 << 20):
        r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=120); total = int(r.headers["Content-Range"].split("/")[1])
        with open(out, "wb") as f: f.truncate(total)
        def part(a):
            b = min(a + chunk, total) - 1
            for k in range(6):
                try:
                    d = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=600).content; assert len(d) == b - a + 1
                    with open(out, "r+b") as f: f.seek(a); f.write(d)
                    return len(d)
                except Exception: time.sleep(5 * (k + 1))
            raise RuntimeError(f"range {a}-{b} failed")
        t = time.time()
        with ThreadPoolExecutor(n) as ex: got = sum(ex.map(part, range(0, total, chunk)))
        log(f"downloaded {out.name}: {got / 1e9:.2f} GB in {time.time() - t:.0f} s"); assert got == total
    names = []
    for k in ("vox2_faces.zip", "vb_media.zip"):
        fetch(URLS[k], DL / k)
        with zipfile.ZipFile(DL / k) as z:
            mem = [m for m in z.namelist() if m.endswith(".jpg")]; z.extractall(MED, mem); names += mem; log(f"{k}: {len(mem)} faces extracted")
        (DL / k).unlink()
    import cv2
    good = np.array([cv2.imread(str(MED / n)) is not None for n in names]); log(f"faces {len(names)} decodable {int(good.sum())}"); assert good.mean() > .995
    sub = str(MED / names[int(np.argmax(good))])
    pd.DataFrame(dict(path=[str(MED / n) if g else sub for n, g in zip(names, good)], good=good)).to_csv(W / "faces.csv", index=False)
    pd.DataFrame(dict(name=names, good=good)).to_csv(OUT / "hbpef_face_names.csv", index=False)
    open(W / "worker.py", "w").write(r"""__WORKER__""")
    for i in (0, 1): open(W / f"w{i}.py", "w").write(f"import sys; sys.argv = ['w', '{i}']; exec(open('/kaggle/temp/hbpef/worker.py').read())")
    S.mkdir(parents=True, exist_ok=True)   # v2: run_workers uses cwd=S (no repo extracted in this kernel)
    log("setup done -> workers"); run_workers([("w0", W / "w0.py", 0), ("w1", W / "w1.py", 1)])
    E = np.empty((len(names), 1024), np.float16); E[0::2] = np.load(OUT / "shard0.f16.npy"); E[1::2] = np.load(OUT / "shard1.f16.npy")
    np.save(OUT / "hbpef_face.f16.npy", E); (OUT / "shard0.f16.npy").unlink(); (OUT / "shard1.f16.npy").unlink()
    json.dump(dict(pair_local=True, labels_used=False, test_statistics=False, n_faces=len(names), model="facebook/pe-av-base@__PEREV__ sha256 __PESHA__, 1-frame video, PIL RGB, bs 1, fp32",
                   parity={i: json.load(open(OUT / f"parity_{i}.json")) for i in (0, 1)}), open(OUT / "PROVENANCE.json", "w"), indent=1)
    with zipfile.ZipFile("/kaggle/working/hbpef.zip", "w", zipfile.ZIP_STORED) as z:
        for p in sorted(OUT.iterdir()): z.write(p, p.name)
    shutil.rmtree(OUT); log("JOB DONE")
except Exception as ex:
    log(f"VMERR {ex!r}"); traceback.print_exc(); raise
'''

if __name__ == "__main__":
    assert '"""' not in WORKER and '"""' not in SYNI; compile(WORKER, "worker", "exec")
    src_ = open(K / "build_t3_sub26_kernel.py").read(); ns = {}; exec(src_[src_.index("def links():"):src_.index("CODE = r\'\'\'")], ns)
    urls = {k: v for k, v in ns["links"]().items() if k in ("vox2_faces.zip", "vb_media.zip")}; assert len(urls) == 2
    refs = mac_refs(); print("mac refs", [round(sum(x * x for x in r) ** .5, 4) for r in refs])
    rep = {"__URLS__": json.dumps(urls), "__PEREV__": PEREV, "__PESHA__": PESHA, "__SYNI__": SYNI, "__REFS__": json.dumps(refs), "__WORKER__": WORKER}
    src = MAIN
    for a_, b_ in rep.items(): assert a_ in src, a_; src = src.replace(a_, b_)
    left = set(re.findall(r"__[A-Z0-9]+__", src)); assert not left, left
    cells = [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s_} for s_ in (COMMON, src)]
    for c in cells: compile(c["source"], "cell", "exec")
    d = K / "hbpef_extract"; d.mkdir(exist_ok=True)
    json.dump({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}, open(d / "hbpef_extract.ipynb", "w"))
    json.dump({"id": f"{USER}/flag-t3-hbpef-extract", "title": "FLAG T3 hbpef extract", "code_file": "hbpef_extract.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
    print(f"built {d} | notebook {(d / 'hbpef_extract.ipynb').stat().st_size / 1e6:.2f} MB")
