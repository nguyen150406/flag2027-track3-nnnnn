"""HB + ImageBind VOICE blocks (17/11 PM, user "oke 1 2 3 luôn đi", incl. using the HF token to sign VoxCeleb2 links): Kaggle notebook
ncamtu06/flag-t3-hbibv-extract. Voice embeddings of every HB pool clip with (a) FROZEN ImageBind-huge audio and (b) IB-LoRA (merged; lora sha
60d816e4..., trained on VoxCeleb1 + MAV = ids disjoint from HB's VoxCeleb2 + VoxBlink) -> the voice-side counterparts of caches/ext/*_face_{ibv,ibl}.
Audio: VoxCeleb2 clips of rows_vox2 (ALL clips, no exclusion; HB applies its own LID/exclusion) by range request from the gated HF mirror
ProgramComputer/voxceleb through SIGNED URLs made at build time (the token never leaves the Mac; URLs expire ~1 h -> push at once), ffmpeg ->
16 kHz mono (= the 13/11 audeering / whisper kernels); VoxBlink wav from vb_media.zip (trncnguyn kernel output link).
Extractor = scripts/t3_imagebind_extract.py (3 fixed 2-s crops, kaldi fbank, L2), same code as the FLAG / eval caches.
GATE (fail fast in each worker before the long run): two synthetic wavs (deterministic numpy, written with soundfile PCM_16) must reproduce the Mac
embeddings of the same model (frozen / merged IB-LoRA), min cos > .999.
Inputs: dataset ncamtu06/flag-t3-hbib-files (lora_best.pt, vox2_manifest.csv, rows_vox2.csv, rows_vb.csv).
Output hbibv.zip: ib_voice.f16.npy, ibl_voice.f16.npy, hbibv_voices.csv (src, clip_index, good), parity json, PROVENANCE.
usage: python build_hbibv_extract.py [--data]   (--data: (re)write kaggle/hbib_files/ for `kaggle datasets create`)"""
import os, json, hashlib, re, sys, shutil
from pathlib import Path
import numpy as np
K = Path(__file__).resolve().parent; sys.path.insert(0, str(K)); T3 = K.parent; CX = T3 / "caches/ext"
import build_xj1ib_extract as B
USER = "ncamtu06"; LP = T3 / "xj/iblora/lora_best.pt"; LSHA = hashlib.sha256(LP.read_bytes()).hexdigest(); assert LSHA.startswith("60d816e4")
SRC_IL = (B.SC / "t3_iblora.py").read_bytes()
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
    """frozen and merged IB-LoRA voice embeddings of the synthetic wavs on the Mac CPU (same code as the FLAG / eval caches)"""
    import tempfile, torch
    sys.path.insert(0, str(B.SC)); import t3_imagebind_extract as T, t3_iblora as IL
    ns = {}; exec(SYN, ns); d = tempfile.mkdtemp(); ps = ns["synth_wavs"](d); dev = torch.device("cpu")
    m0 = T.model(dev); e0 = T.encode(m0, ps, "voice", dev); del m0
    m1, _ = IL.load_merged(str(LP), dev); e1 = T.encode(m1, ps, "voice", dev)
    return {"frozen": e0.astype(np.float32).tolist(), "lora": e1.astype(np.float32).tolist()}


WORKER = r'''
import os, sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
MODE = sys.argv[1]; T0 = time.time()
def log(m): print(time.strftime("%H:%M:%S"), f"[{(time.time() - T0) / 60:6.1f}m]", MODE, m, flush=True)
W = Path("/kaggle/temp/hbibv"); OUT = Path("/kaggle/working/hbibv_out"); sys.path.insert(0, str(W / "root/Track 3/scripts"))
import t3_imagebind_extract as T, t3_iblora as IL
dev = torch.device("cuda"); u = lambda E: E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
if MODE == "lora":   # RAM: load after the frozen worker has its model
    t = time.time()
    while not (W / "frozen_ready").exists() and time.time() - t < 1200: time.sleep(10)
m = T.model(dev) if MODE == "frozen" else IL.load_merged(str(W / "lora_best.pt"), dev)[0]
if MODE == "frozen": (W / "frozen_ready").touch()
torch.cuda.empty_cache(); ref = np.array(json.load(open(W / "refs.json"))[MODE], np.float32)
e = T.encode(m, [str(W / "synth0.wav"), str(W / "synth1.wav")], "voice", dev); c = (u(e) * u(ref)).sum(1); log(f"PARITY vs Mac on synthetic wavs: {c.round(6).tolist()}")
json.dump(dict(min=float(c.min())), open(OUT / f"parity_{MODE}.json", "w")); assert c.min() > .999, f"parity failed {c}"
L = pd.read_csv(W / "voices.csv"); t = time.time(); E = T.encode(m, L.path.tolist(), "voice", dev, bs=32).astype(np.float32)
log(f"encoded {len(E)} in {(time.time() - t) / 60:.1f} min"); assert np.isfinite(E).all()
np.save(OUT / f"{'ib' if MODE == 'frozen' else 'ibl'}_voice.f16.npy", E.astype(np.float16)); log("SAVED")
'''

BODY = r'''try:
    import numpy as np
    import torch; ng = torch.cuda.device_count(); log(f"GPUs {ng}; cpus {os.cpu_count()}"); assert ng >= 1, "no GPU"
    for k_, u_ in list(URLS.items()) + [(f"hf{i}", h_) for i, h_ in enumerate(__HFURLS__)]:   # fail fast: temporary links still alive?
        r_ = requests.get(u_, headers={"Range": "bytes=0-3"}, timeout=120); log(f"link {k_}: HTTP {r_.status_code}"); assert r_.status_code == 206 and r_.content[:2] == b"PK", f"link {k_} dead"
    EXT = T3 / "caches/ext"; EXT.mkdir(parents=True, exist_ok=True); WB = Path("/kaggle/temp/hbibv"); WB.mkdir(parents=True, exist_ok=True)
    OB = Path("/kaggle/working/hbibv_out"); OB.mkdir(parents=True, exist_ok=True)
    for name, dst in (("vox2_manifest.csv", EXT), ("rows_vox2.csv", EXT), ("rows_vb.csv", EXT), ("lora_best.pt", WB)):
        hit = [h for h in find(name) if "flag-t3-hbib-files" in str(h)]; assert hit, f"{name} missing"; shutil.copyfile(hit[0], dst / name)
    import hashlib as hl_
    lh = hl_.sha256(open(WB / "lora_best.pt", "rb").read()).hexdigest(); log(f"lora sha {lh[:16]}"); assert lh == "__LSHA__"
    open(EXT / "exclude_bengali.csv", "w").write("identity,src\n")   # fetch ALL rows_vox2 clips (HB applies LID + exclusion itself)
    # ---- ImageBind code + checkpoint (pins = the Mac / xj1ib extraction)
    sh("pip install -q ftfy iopath einops 2>&1 | tail -1", "extract.log")
    import gzip as gz_, base64 as b64_
    SB = WB / "root/Track 3/scripts"; SB.mkdir(parents=True, exist_ok=True); IBD = WB / "root/Track 3/caches/imagebind"; (IBD / "pylib").mkdir(parents=True, exist_ok=True)
    for n, (s_, h_) in json.loads(r"""__SRCS__""").items():
        b_ = gz_.decompress(b64_.b64decode(s_)); assert hl_.sha256(b_).hexdigest() == h_, n; (SB / n).write_bytes(b_)
    rp = IBD / "ImageBind-__COMMIT__"; sh(f"git clone -q https://github.com/facebookresearch/ImageBind.git '{rp}' && git -C '{rp}' checkout -q __COMMIT__", "extract.log")
    for f, h in json.loads(r"""__MODF__""").items(): assert hl_.sha256(open(rp / f"imagebind/models/{f}.py", "rb").read()).hexdigest() == h, f"ImageBind {f}.py differs"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    from huggingface_hub import hf_hub_download
    ck = hf_hub_download("seung275/imagebind_huge.pth", "imagebind_huge.pth", cache_dir=str(WB / "hf"))
    h = hl_.sha256(open(ck, "rb").read()).hexdigest(); log(f"IB checkpoint sha {h[:16]}"); assert h == "__CKSHA__"; os.symlink(os.path.realpath(ck), IBD / "imagebind_huge.pth")
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
    V.assign(path=np.where(V.good, V.path, sub)).to_csv(WB / "voices.csv", index=False); V[["src", "clip_index", "good"]].to_csv(OB / "hbibv_voices.csv", index=False)
    open(WB / "worker.py", "w").write(r"""__WORKER__""")
    procs = []
    for g, mode in enumerate(("frozen", "lora")):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g if ng >= 2 else 0), PYTHONUNBUFFERED="1")
        if ng < 2 and mode == "lora":   # one GPU: run sequentially
            for p_ in procs: p_.wait()
        procs.append(subprocess.Popen([sys.executable, str(WB / "worker.py"), mode], stdout=open(LG / f"w_{mode}.log", "w"), stderr=subprocess.STDOUT, env=env, cwd=str(SB)))
    while any(p.poll() is None for p in procs):
        time.sleep(120)
        if any(p.poll() not in (None, 0) for p in procs):
            for p in procs:
                if p.poll() is None: p.kill()
        for mode in ("frozen", "lora"):
            f = LG / f"w_{mode}.log"
            if f.exists(): log(f"[{mode}] " + ((f.read_text().strip().splitlines() or ["-"])[-1])[-160:])
    for mode in ("frozen", "lora"): print(f"===== {mode}\n" + "\n".join((LG / f"w_{mode}.log").read_text().splitlines()[-25:]), flush=True)
    assert all(p.returncode == 0 for p in procs), [p.returncode for p in procs]
    json.dump(dict(pair_local=True, labels_used=False, test_statistics=False, lora_sha256="__LSHA__", n_voices=len(V),
                   parity={m: json.load(open(OB / f"parity_{m}.json")) for m in ("frozen", "lora")},
                   imagebind="facebookresearch/ImageBind@__COMMIT__, seung275/imagebind_huge.pth sha256 __CKSHA__ (+ IB-LoRA merged for ibl)",
                   audio="vox2: HF mirror ProgramComputer/voxceleb mp4 -> ffmpeg 16 kHz mono; vb: vb_media.zip wav"), open(OB / "PROVENANCE.json", "w"), indent=1)
    with zipfile.ZipFile(OUT / "hbibv.zip", "w", zipfile.ZIP_STORED) as zz:
        for p in sorted(OB.iterdir()): zz.write(p, p.name)
    shutil.rmtree(OB); log(f"JOB DONE {(OUT / 'hbibv.zip').stat().st_size / 1e6:.0f} MB")
'''

if __name__ == "__main__":
    if "--data" in sys.argv:
        fd = K / "hbib_files"; fd.mkdir(exist_ok=True)
        for f in (LP, CX / "vox2_manifest.csv", CX / "rows_vox2.csv", CX / "rows_vb.csv"): shutil.copyfile(f, fd / f.name)
        json.dump({"title": "flag-t3-hbib-files", "id": f"{USER}/flag-t3-hbib-files", "licenses": [{"name": "other"}]}, open(fd / "dataset-metadata.json", "w"))
        print("data dir", fd); sys.exit()
    import types
    _w = (K / "build_t3_whs_kernel.py").read_text(); ns = {"__file__": str(K / "build_t3_whs_kernel.py")}
    exec(_w[:_w.index("BODY = r\'\'\'")], ns); exec(_w[_w.index("def links():"):_w.index("code = (HEAD")], ns)
    HEAD, TAIL, VOX2 = ns["HEAD"], ns["TAIL"], ns["VOX2"]
    assert compile(WORKER, "w", "exec") and '"""' not in WORKER and '"""' not in SYN
    refs = mac_refs(); print("mac refs done")
    srcs = {"t3_imagebind_extract.py": B.SRC, "t3_iblora.py": SRC_IL}
    body = BODY.replace("__VOX2__", VOX2)
    rep = {"__SRCS__": json.dumps({n: [B.b64(b), hashlib.sha256(b).hexdigest()] for n, b in srcs.items()}), "__COMMIT__": B.COMMIT, "__MODF__": json.dumps(B.MODF),
           "__CKSHA__": B.CKPT_SHA, "__LSHA__": LSHA, "__SYN__": SYN, "__REFS__": json.dumps(refs), "__WORKER__": WORKER}
    for a, b in rep.items(): assert a in body, a; body = body.replace(a, b)
    urls = {k: v for k, v in ns["links"]().items() if k == "vb_media.zip"}
    code = (HEAD + body + TAIL).replace("__URLS__", json.dumps(urls)).replace("__HFURLS__", json.dumps(ns["hf_urls"]()))
    code = code.replace("import os, sys, json, time", "import pandas as pd\nimport os, sys, json, time", 1)
    left = set(re.findall(r"__[A-Z0-9]+__", code)); assert not left, left
    compile(code, "hbibv", "exec")
    d = K / "hbibv_extract"; d.mkdir(exist_ok=True)
    json.dump({"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
               "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}, open(d / "hbibv_extract.ipynb", "w"))
    json.dump({"id": f"{USER}/flag-t3-hbibv-extract", "title": "FLAG T3 hbibv extract", "code_file": "hbibv_extract.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
               "dataset_sources": [f"{USER}/flag-t3-hbib-files"], "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
    print(f"built {d} | notebook {(d / 'hbibv_extract.ipynb').stat().st_size / 1e6:.2f} MB")
