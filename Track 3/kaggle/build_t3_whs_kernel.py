"""Whisper voice branch, step 2 on Kaggle (user 2026-11-03 "cứ chạy hướng whisper đi"; account chosen after the quota review:
trncnguyn ~11 h left this week, nguyntrnduck ~10 h, trnlhvy ~2 h). Extraction only, no training:
  vox2 audio first : VoxCeleb2 clips of rows_vox2 (Bengali-excluded speakers skipped) by range request from the HF mirror through
                     SIGNED URLs made at build time (the token never leaves the Mac; URLs expire ~1 h -> push right away)
  VoxBlink         : wav from the trncnguyn/flag-t3-voxblink-prep output (vb_media.zip)
  smoke            : 40 files per set on GPU 0, prints s/file (fails fast before the long run)
  whs              : t3_extract_wavlmsv.py --name whs (frozen whisper-large-v3-turbo, layer 2 mean|std, one file at a time) on
                     FLAG train+dev + vox2 + vb, two shards on the two T4s (halves the session hours), 4 h timeout each
Output: whs_embeddings.zip with float16 arrays (local install converts to float32). Inputs: trncnguyn/flag-t3-cv-bundle, -cv-extra,
flag-2027-data, flag-t3-overlay56 (code tree) + flag-t3-whs-files (current extractor/t3data/t3lib, row tables, exclusion list).
Usage: python build_t3_whs_kernel.py   (then kaggle kernels push -p t3_whs at once)"""
import json, re
from pathlib import Path
import requests
K = Path(__file__).parent
src = (K / "build_t3_sub26_kernel.py").read_text()
CODE = re.search(r"CODE = r'''(.*?)'''", src, re.S).group(1)
HEAD = CODE[:CODE.index("\ntry:") + 1]; TAIL = CODE[CODE.index("\nexcept Exception as e:") + 1:]  # top-level try (fetch() has its own)
vox2 = (K / "build_t3_sub35_kernel.py").read_text(); VOX2 = vox2[vox2.index("VOX2 = r'''") + 11:vox2.index("'''\nhead = head[:dl0]")]
BODY = r'''try:
    import torch; ng = torch.cuda.device_count(); log(f"GPUs {ng}: {[torch.cuda.get_device_name(i) for i in range(ng)]}; cpus {os.cpu_count()}; inputs {sorted(p.name for p in IN.iterdir())}")
    assert ng >= 1, "no GPU"
    (W / "FLAG_2027").mkdir(exist_ok=True)
    tree("flag-t3-cv-bundle", "flag_cv16.tar"); tree("flag-t3-cv-extra", "flag_cv_extra.tar"); tree("flag-t3-overlay56", "overlay56.tar")
    EXT = T3 / "caches/ext"
    for name, dst in (("t3_extract_wavlmsv.py", SC), ("t3data.py", SC), ("t3lib.py", SC), ("vox2_manifest.csv", EXT), ("rows_vox2.csv", EXT),
                      ("rows_vb.csv", EXT), ("exclude_bengali.csv", EXT)):
        hit = [h for h in find(name) if "flag-t3-whs-files" in str(h)]; assert hit, f"{name} missing in flag-t3-whs-files"; shutil.copyfile(hit[0], dst / name)
    sh(f"grep -c whisper-large-v3-turbo {SC}/t3_extract_wavlmsv.py; df -h {TMP} | tail -1", "extract.log"); log("STAGE extract done")
    dev = [p for p in find("dev_set") if (p / "gender/English_test/faces/00000.jpg").exists()]; assert dev, "missing dev set (flag-2027-data)"; FD = dev[0].parent
    log(f"FLAG root {FD}")
__VOX2__    fetch(URLS["vb_media.zip"], DL / "vb_media.zip")
    with zipfile.ZipFile(DL / "vb_media.zip") as z: z.extractall(TMP / "vbm", [m for m in z.namelist() if m.startswith("wav/")])
    shutil.move(str(TMP / "vbm/wav"), str(AUD / "vb")); shutil.rmtree(DL); shutil.rmtree(TMP / "vbm"); DL.mkdir()
    log(f"STAGE audio done: vox2 {len(list((AUD / 'vox2').glob('*.wav')))} wav, vb {len(list((AUD / 'vb').glob('*.wav')))} wav")
    sh(f"cd '{SC}' && python -c \"from transformers import WhisperModel, WhisperFeatureExtractor as F; r='41f01f3fe87f28c78e2fbf8b568835947dd65ed9'; "
       f"F.from_pretrained('openai/whisper-large-v3-turbo', revision=r); WhisperModel.from_pretrained('openai/whisper-large-v3-turbo', revision=r); print('model cached')\"", "model.log")
    X = f"python -X faulthandler -u t3_extract_wavlmsv.py --name whs --flag_data '{FD}' --roots vox2={AUD}/vox2 vb={AUD}/vb --device cuda --workers 4"
    t = time.time(); sh(f"cd '{SC}' && T3_FT_DEBUG_N=40 CUDA_VISIBLE_DEVICES=0 {X} --out {TMP}/smoke", "smoke.log")
    log(f"STAGE smoke done in {(time.time() - t) / 60:.1f} min (3 sets x 40 files incl. model load)")
    parts = [(0, f"--shard 0/{min(ng, 2)}"), (1, f"--skip_flag --shard 1/2")] if ng >= 2 else [(0, "")]
    procs = [subprocess.Popen(["bash", "-c", f"cd '{SC}' && CUDA_VISIBLE_DEVICES={g} timeout -s ABRT 14400 {X} --out {TMP}/whs{g} {extra} > {LG}/whs{g}.log 2>&1"]) for g, extra in parts]
    while any(p.poll() is None for p in procs):
        time.sleep(600)
        for g, _ in parts: log(f"whs{g}: " + ((LG / f"whs{g}.log").read_text().strip().splitlines() or ["(no output yet)"])[-1][-160:])
    assert all(p.returncode == 0 for p in procs), [p.returncode for p in procs]
    import numpy as np
    out = {"flag_voice_whs": np.load(f"{TMP}/whs0/flag_voice_whs.float32.npy")}
    for s in ("vox2", "vb"):
        sfx = [f".s{g}" for g, _ in parts] if len(parts) == 2 else [""]
        out[f"{s}_voice_whs"] = sum(np.load(f"{TMP}/whs{g}/{s}_voice_whs{x}.float32.npy") for (g, _), x in zip(parts, sfx))
        out[f"{s}_voice_whs.ok"] = np.logical_or.reduce([np.load(f"{TMP}/whs{g}/{s}_voice_whs{x}.ok.npy") for (g, _), x in zip(parts, sfx)])
        log(f"{s}: {out[f'{s}_voice_whs.ok'].sum()}/{len(out[f'{s}_voice_whs.ok'])} rows encoded")
    with zipfile.ZipFile(OUT / "whs_embeddings.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in out.items():
            p = TMP / (k + (".npy" if k.endswith(".ok") else ".f16.npy")); np.save(p, v if k.endswith(".ok") else v.astype(np.float16)); z.write(p, p.name)
        for p in Path(f"{TMP}/whs0").glob("diag_*.json"): z.write(p, p.name)
    log(f"JOB DONE -> {OUT / 'whs_embeddings.zip'} {(OUT / 'whs_embeddings.zip').stat().st_size / 1e6:.0f} MB")
'''


def links():  # fresh temporary link of the VoxBlink media zip (trncnguyn = default ~/.kaggle token)
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api = KaggleApi(); api.authenticate()
    with api.build_kaggle_client() as k:
        r = ApiListKernelSessionOutputRequest(); r.user_name = "trncnguyn"; r.kernel_slug = "flag-t3-voxblink-prep"
        out = {f.file_name: f.url for f in k.kernels.kernels_api_client.list_kernel_session_output(r).files if f.file_name == "vb_media.zip"}
    assert len(out) == 1, out; return out


def hf_urls():  # signed CDN URLs of the 6 VoxCeleb2 mp4 zips (the token stays on the Mac; URLs expire ~1 h)
    tok = (Path.home() / ".hf_lrs3_token").read_text().strip(); out = []
    for i in range(1, 7):
        r = requests.head(f"https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_mp4_{i}.zip", allow_redirects=False, timeout=60,
                          headers={"Authorization": f"Bearer {tok}"}); r.raise_for_status(); assert r.status_code in (301, 302, 307), r.status_code; out.append(r.headers["Location"])
    return out


code = (HEAD + BODY.replace("__VOX2__", VOX2) + TAIL).replace("__URLS__", json.dumps(links())).replace("__HFURLS__", json.dumps(hf_urls()))
compile(code, "whs", "exec")
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = K / "t3_whs"; d.mkdir(exist_ok=True); json.dump(nb, open(d / "t3_whs.ipynb", "w"), indent=1)
json.dump({"id": "trncnguyn/flag-t3-whs", "title": "FLAG T3 whs", "code_file": "t3_whs.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["trncnguyn/flag-t3-cv-bundle", "trncnguyn/flag-t3-cv-extra", "trncnguyn/flag-2027-data", "trncnguyn/flag-t3-overlay56",
                               "trncnguyn/flag-t3-whs-files"], "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d)
