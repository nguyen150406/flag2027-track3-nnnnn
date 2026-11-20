"""Audeering age/gender wav2vec2 embeddings for the R2 voice pool (user 13/11 "hãy thử nghiệm hết": Simicch-style attribute embeddings
as extra voice input). Extraction only, no training, no FLAG data.
  vox2 : VoxCeleb2 clips of rows_vox2 (Bengali-excluded speakers skipped; EN/XL held-out speakers KEPT so the gate can score them),
         range reads from the HF mirror through signed URLs made at build time (token stays on the Mac; URLs expire ~1 h -> push at once)
  vb   : wav from the trncnguyn/flag-t3-voxblink-prep output (vb_media.zip, signed link made at build time)
  model: scripts/extract_voice_age.py (same code as FLAG/eval/MAV), audeering 24-layer and 6-layer, T3_DEVICE=cuda, one file at a time
         (<= 60 s chunks), two shards on the two T4s; smoke on 20 files first.
Output: aud_embeddings.zip = {vox2,vb}_{a24,a6}.npz (clip_index, age, gender, emb float16).
  python build_t3_aud_kernel.py <user> <token dir>      then: kaggle datasets create -p aud_files_<user> ; kaggle kernels push -p t3_aud"""
import json, re, shutil, sys
from pathlib import Path
import pandas as pd
K = Path(__file__).parent; T3 = K.parent; EXT = T3 / "caches/ext"; USER = sys.argv[1]
import types
_w = (K / "build_t3_whs_kernel.py").read_text()          # reuse HEAD / TAIL / VOX2 and the signed-link helpers without running that builder
WHS = types.SimpleNamespace(); _ns = {"__file__": str(K / "build_t3_whs_kernel.py")}
exec(_w[:_w.index("BODY = r\'\'\'")], _ns); exec(_w[_w.index("def links():"):_w.index("code = (HEAD")], _ns)
WHS.HEAD, WHS.TAIL, WHS.VOX2, WHS.links, WHS.hf_urls = _ns["HEAD"], _ns["TAIL"], _ns["VOX2"], _ns["links"], _ns["hf_urls"]
fd = K / f"aud_files_{USER}"; fd.mkdir(exist_ok=True)
for f in ("scripts/extract_voice_age.py", "caches/ext/vox2_manifest.csv", "caches/ext/rows_vox2.csv", "caches/ext/rows_vb.csv"):
    shutil.copyfile(T3 / f, fd / Path(f).name)
e = pd.read_csv(EXT / "exclude_ultra.csv"); e[~e.reason.astype(str).str.startswith("ultra held-out")].to_csv(fd / "exclude_bengali.csv", index=False)
json.dump({"title": "flag-t3-aud-files", "id": f"{USER}/flag-t3-aud-files", "licenses": [{"name": "other"}]}, open(fd / "dataset-metadata.json", "w"))
BODY = r'''try:
    import torch; ng = torch.cuda.device_count(); log(f"GPUs {ng}; cpus {os.cpu_count()}"); assert ng >= 1, "no GPU"
    EXT = T3 / "caches/ext"; EXT.mkdir(parents=True, exist_ok=True); SC.mkdir(parents=True, exist_ok=True)
    for name, dst in (("extract_voice_age.py", SC), ("vox2_manifest.csv", EXT), ("rows_vox2.csv", EXT), ("rows_vb.csv", EXT), ("exclude_bengali.csv", EXT)):
        hit = [h for h in find(name) if "flag-t3-aud-files" in str(h)]; assert hit, f"{name} missing"; shutil.copyfile(hit[0], dst / name)
    # v2 (13/11): Kaggle's transformers 5.x breaks the custom audeering class -> pin the Mac version (same numerics as FLAG / eval / MAV caches)
    sh("pip install -q 'transformers==4.57.3' 2>&1 | tail -2; python -c 'import transformers; print(transformers.__version__)'", "extract.log")
__VOX2__    fetch(URLS["vb_media.zip"], DL / "vb_media.zip")
    with zipfile.ZipFile(DL / "vb_media.zip") as z: z.extractall(TMP / "vbm", [m for m in z.namelist() if m.startswith("wav/")])
    shutil.move(str(TMP / "vbm/wav"), str(AUD / "vb")); shutil.rmtree(DL); shutil.rmtree(TMP / "vbm"); DL.mkdir()
    exb = set(pd.read_csv(EXT / "exclude_bengali.csv").query("src == 'vb'").identity.astype(str)); rvb = pd.read_csv(EXT / "rows_vb.csv")
    keep_vb = set(rvb.clip_index[~rvb.identity.astype(str).isin(exb)].astype(int))
    L = []
    for s in ("vox2", "vb"):
        for p in sorted((AUD / s).glob("*.wav"), key=lambda q: int(q.stem)):
            if s == "vox2" or int(p.stem) in keep_vb: L.append((s, int(p.stem), str(p)))
    log(f"STAGE audio done: {len(L)} wav ({sum(x[0] == 'vox2' for x in L)} vox2)")
    import pandas as pd_
    for g in range(min(ng, 2)):
        pd_.DataFrame([dict(src=s, clip_index=c, path=p) for k, (s, c, p) in enumerate(L) if k % min(ng, 2) == g]).to_csv(TMP / f"list{g}.csv", index=False)
    pd_.read_csv(TMP / "list0.csv").head(20).to_csv(TMP / "smoke.csv", index=False)
    R24, R6 = "audeering/wav2vec2-large-robust-24-ft-age-gender", "audeering/wav2vec2-large-robust-6-ft-age-gender"
    t = time.time(); sh(f"cd '{SC}' && T3_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 python -u extract_voice_age.py {R24} {TMP}/smoke.csv {TMP}/smoke24.npz && "
                        f"T3_DEVICE=cuda CUDA_VISIBLE_DEVICES=0 python -u extract_voice_age.py {R6} {TMP}/smoke.csv {TMP}/smoke6.npz", "smoke.log")
    import numpy as np
    z = np.load(TMP / "smoke24.npz"); log(f"STAGE smoke done {(time.time() - t) / 60:.1f} min; emb {z['emb'].shape} finite {np.isfinite(z['emb'].astype(np.float32)).all()} age {z['age'][:5]}")
    procs = [subprocess.Popen(["bash", "-c", f"cd '{SC}' && export T3_DEVICE=cuda CUDA_VISIBLE_DEVICES={g} && python -u extract_voice_age.py {R24} {TMP}/list{g}.csv {TMP}/a24_{g}.npz > {LG}/a24_{g}.log 2>&1 && "
                              f"python -u extract_voice_age.py {R6} {TMP}/list{g}.csv {TMP}/a6_{g}.npz > {LG}/a6_{g}.log 2>&1"]) for g in range(min(ng, 2))]
    while any(p.poll() is None for p in procs):
        time.sleep(300)
        for g in range(min(ng, 2)):
            for m in ("a24", "a6"):
                f = LG / f"{m}_{g}.log"
                if f.exists(): log(f"{m} gpu{g}: " + (f.read_text().strip().splitlines() or ["-"])[-1][-120:])
    assert all(p.returncode == 0 for p in procs), [p.returncode for p in procs]
    with zipfile.ZipFile(OUT / "aud_embeddings.zip", "w", zipfile.ZIP_DEFLATED) as zz:
        for m in ("a24", "a6"):
            parts = [(pd_.read_csv(TMP / f"list{g}.csv"), np.load(TMP / f"{m}_{g}.npz")) for g in range(min(ng, 2))]
            lst = pd_.concat([p[0] for p in parts]); arr = {k: np.concatenate([p[1][k] for p in parts]) for k in ("age", "gender", "emb")}
            for s in ("vox2", "vb"):
                sel = (lst.src == s).to_numpy(); o = TMP / f"{s}_{m}.npz"
                np.savez(o, clip_index=lst.clip_index.to_numpy()[sel], age=arr["age"][sel], gender=arr["gender"][sel], emb=arr["emb"][sel].astype(np.float16)); zz.write(o, o.name)
                log(f"{s} {m}: {sel.sum()} clips, finite {np.isfinite(arr['emb'][sel].astype(np.float32)).all()}")
    log(f"JOB DONE {(OUT / 'aud_embeddings.zip').stat().st_size / 1e6:.0f} MB")
'''
code = (WHS.HEAD + BODY.replace("__VOX2__", WHS.VOX2) + WHS.TAIL).replace("__URLS__", json.dumps(WHS.links())).replace("__HFURLS__", json.dumps(WHS.hf_urls()))
code = code.replace("import os, sys, json, time", "import pandas as pd\nimport os, sys, json, time", 1)
compile(code, "aud", "exec")
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = K / "t3_aud"; d.mkdir(exist_ok=True); json.dump(nb, open(d / "t3_aud.ipynb", "w"), indent=1)
json.dump({"id": f"{USER}/flag-t3-aud", "title": "FLAG T3 aud", "code_file": "t3_aud.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": [f"{USER}/flag-t3-aud-files"], "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d, fd)
