"""ULTRA (11/11): no-FLAG CLIP LoRA (clipft recipe + --no_flag), T3_EXCLUDE = exclude_ultra.csv (v3 + 550 held-out external persons). Derived from build_t3_clean_kernel.py.
Clean retrain of the four sub60 CLIP LoRAs on Kaggle (user 2026-11-05: "train lại hết tất cả trên data đó", big run from the
Saturday 07/11 reset). Exact original recipes (Track 3/release/README_SUB60.md 5b), Bengali list v3 (caches/ext/exclude_bengali_v3.csv,
phase A) via T3_EXCLUDE for both the LoRA (t3_ft_clip.py reads external identities through t3data.ext_source) and the projectors:
  lft : clipft   6 blocks                      -> branches of configs/t3_c33a.json (clipft x 5 voices)
  lsa : clipsa   6 blocks + South-Asian batch  -> LoRA + face arrays only (clipsa_wsv is built on the Mac with clipft: t3_c35)
  l57 : clipft12 12 blocks                     -> configs/t3_c57.json
  l58 : clipsa12 12 blocks + South-Asian batch -> configs/t3_c58.json
SUB57 template (SUB26 head: FLAG / VoxCeleb1 / VoxCeleb2 / VoxBlink / LRS3 faces; <user>/flag-t3-overlay56 must exist on the account),
current code + configs + list v3 embedded in the notebook, 20-step smoke with the 12 h speed guard. Output: <job>_out.zip.
Usage: python build_t3_clean_kernel.py lft|lsa|l57|l58 <user>   (default ~/.kaggle token = trncnguyn for the image links; push at once)"""
import json, base64, sys
from pathlib import Path
K = Path(__file__).parent; T3 = K.parent
SPEC = {"lnf": ("clipnf", " --no_flag", ""), "lnf2": ("clipnf2", " --no_flag --seed 7", "")}
JOB, USER = sys.argv[1], sys.argv[2]; TAG, EXTRA, BUILD = SPEC[JOB]
assert (T3 / "caches/ext/exclude_bengali_v3.csv").exists(), "run phase A first (t3_lid_merge.py -> exclude_bengali_v3.csv)"
src = (K / "build_t3_sub56_kernels.py").read_text(); g = {"__file__": str(K / "build_t3_sub56_kernels.py")}
exec(src[:src.index("urls = links()")], g)  # links, head, tail, PRE
files = ["scripts/t3data.py", "scripts/t3_ft_clip.py", "scripts/t3_build.py", "scripts/t3_build_par.py", "scripts/t3train.py", "caches/ext/exclude_ultra.csv"]
files.append(f"configs/{BUILD or 't3_c33a'}.json")  # lsa has no build but its preflight reads t3_c33a.json (failed 07/11 22:41)
EMB = {rel: base64.b64encode((T3 / rel).read_bytes()).decode() for rel in files}
INJECT = f'''    import base64 as _b64
    for _rel, _b in {json.dumps(EMB)}.items(): (T3 / _rel).parent.mkdir(parents=True, exist_ok=True); (T3 / _rel).write_bytes(_b64.b64decode(_b))
    os.environ["T3_EXCLUDE"] = str(T3 / "caches/ext/exclude_ultra.csv")  # inherited by every subprocess (LoRA + projectors)
    log("embedded files written, T3_EXCLUDE = ultra (v3 + held-out)")
'''
a = '    sh(f"df -h {TMP} | tail -1", "extract.log"); log("STAGE extract done")\n'
head = g["head"]; assert head.count(a) == 1; head = head.replace(a, INJECT + a)
OV = {"ncamtu06": ['flag-t3-overlay50', 'flag-t3-overlay66']}.get(USER, ['flag-t3-overlay56'])  # ncamtu06 has no overlay56 (overlay56 = overlay50 - FaRL/LoRA files + code)
if OV != ['flag-t3-overlay56']:
    o = 'tree("flag-t3-overlay56", "overlay56.tar")'; assert head.count(o) == 1
    head = head.replace(o, '; '.join(f'tree("{d}", "{d.split("-")[-1]}.tar")' for d in OV))
PRE = g["PRE"].replace("__CFG__", f"{BUILD or 't3_c33a'}.json").replace("'../caches/ext/exclude_bengali.csv'", "'../caches/ext/exclude_ultra.csv'")
WORK = PRE + '''    FT = f"python -X faulthandler -u t3_ft_clip.py --fold -1 --tag __TAG____EXTRA__ --device cuda --workers 4 --roots {roots}"
    t = time.time(); sh(f"cd '{SC}' && T3_FT_DEBUG_N=256 {FT} --steps 20 --out {TMP}/smoke", "smoke.log")
    import re as _re
    m = {int(k): float(v) for v, k in _re.findall(r"\\[\\s*([0-9.]+)m\\] fold -1: step (\\d+)/20", (LG / "smoke.log").read_text())}
    sps = (m[19] - m[0]) * 60 / 19; log(f"STAGE smoke done in {(time.time() - t) / 60:.1f} min: {sps:.1f} s/step -> 1515 steps ~{1515 * sps / 3600:.1f} h")
    assert 1515 * sps < 8 * 3600, "too slow for the 12 h limit (LoRA + encode + build); stopping early"
    sh(f"cd '{SC}' && {FT} --out {TMP}/ft", "ft.log"); shutil.copy(f"{TMP}/ft/__TAG__.pt", OUT); install("__TAG__", f"{TMP}/ft")
    shutil.copy(f"{TMP}/ft/flag_face___TAG__.float32.npy", OUT)
    with zipfile.ZipFile(OUT / "__TAG___faces.zip", "w", zipfile.ZIP_DEFLATED) as z:  # external face arrays (the Mac builds t3_c35 from them)
        for s in ("vox", "vox2", "vb", "lrs3"):
            for e in ("float32", "ok"): z.write(f"{TMP}/ft/{s}_face___TAG__.{e}.npy", f"{s}_face___TAG__.{e}.npy")
'''
WORK += ('''    build_pack("__BUILD__.json", "__BUILD__", [f"{TMP}/ft/__TAG__.pt", f"{TMP}/ft/flag_face___TAG__.float32.npy"])\n''' if BUILD else
         '''    log("JOB DONE (LoRA + faces only)")\n''')
WORK = WORK.replace("__TAG__", TAG).replace("__EXTRA__", EXTRA).replace("__BUILD__", BUILD)
code = (head + WORK + g["tail"]).replace("__URLS__", json.dumps(g["links"]())); compile(code, JOB, "exec")
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = K / f"t3_ultra_{JOB}_{USER}"; d.mkdir(exist_ok=True); json.dump(nb, open(d / f"t3_ultra_{JOB}.ipynb", "w"), indent=1)
json.dump({"id": f"{USER}/flag-t3-ultra-{JOB}", "title": f"FLAG T3 ultra {JOB}", "code_file": f"t3_ultra_{JOB}.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": ["trncnguyn/flag-t3-cv-bundle", "trncnguyn/flag-t3-cv-extra", "trncnguyn/flag-t3-ft-local", "trncnguyn/flag-2027-data"] + [f"{USER}/{d}" for d in OV],
           "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d)
