"""90c-X (15/11, user "oke chạy ngay đi"): the sub90c no-FLAG CLIP LoRA recipe (clipnf seed 2027 / clipnf2 seed 7, t3_ft_clip.py --no_flag,
T3_EXCLUDE = exclude_ultra.csv) with ONE change: an extra external source "mav" = MAV-Celeb v1+v2 (XJ-U's clip-LID-filtered set,
English + Urdu + Hindi, South-Asian identities; voices = cat4 from the XJ-1 extraction; dataset <user>/flag-t3-mavsrc built by
build_mavsrc.py; images from KAGGLE_USER/external-data-mavceleb). Same-identity voices in another language are SupCon positives.
15/11 option A (user "cứ làm thử hết"): --xlang mav = for MAV identities with voices in >= 2 languages, the per-id voice draws
of a batch come from DIFFERENT languages (forced cross-language positives); other sources and their rng stream unchanged.
Copy of build_t3_ultra_lora.py (kept unchanged); everything else identical (steps 1515, k 6, rank 16, smoke + speed guard).
Usage: python build_t3_lnfm.py lnfm|lnfm2 <user>   (default ~/.kaggle token = trncnguyn only to list the image links)"""
import json, base64, sys
from pathlib import Path
K = Path(__file__).parent; T3 = K.parent
SPEC = {"lnfm": ("clipnfm", " --no_flag --sources vox,vox2,vb,lrs3,mav --nocache mav --xlang mav", ""),
        "lnfm2": ("clipnfm2", " --no_flag --seed 7 --sources vox,vox2,vb,lrs3,mav --nocache mav --xlang mav", "")}
JOB, USER = sys.argv[1], sys.argv[2]; TAG, EXTRA, BUILD = SPEC[JOB]
assert (T3 / "caches/ext/exclude_bengali_v3.csv").exists(), "run phase A first (t3_lid_merge.py -> exclude_bengali_v3.csv)"
src = (K / "build_t3_sub56_kernels.py").read_text(); g = {"__file__": str(K / "build_t3_sub56_kernels.py")}
exec(src[:src.index("urls = links()")], g)  # links, head, tail, PRE
files = ["scripts/t3data.py", "scripts/t3_ft_clip.py", "scripts/t3_build.py", "scripts/t3_build_par.py", "scripts/t3train.py", "caches/ext/exclude_ultra.csv"]
files.append(f"configs/{BUILD or 't3_c33a'}.json")  # lsa has no build but its preflight reads t3_c33a.json (failed 07/11 22:41)
EMB = {rel: base64.b64encode((T3 / rel).read_bytes()).decode() for rel in files}
INJECT = '''    import base64 as _b64
    for _rel, _b in __EMB__.items(): (T3 / _rel).parent.mkdir(parents=True, exist_ok=True); (T3 / _rel).write_bytes(_b64.b64decode(_b))
    os.environ["T3_EXCLUDE"] = str(T3 / "caches/ext/exclude_ultra.csv")  # inherited by every subprocess (LoRA + projectors)
    log("embedded files written, T3_EXCLUDE = ultra (v3 + held-out)")
    _ms = [p.parent for p in find("rows_mav.csv") if "mavsrc" in str(p)]; assert _ms, "missing <user>/flag-t3-mavsrc"; _ms = _ms[0]
    import numpy as _np, pandas as _pd
    _r = _pd.read_csv(_ms / "rows_mav.csv"); _vok = _np.load(_ms / "mavv_ok.npy"); _fok = _np.load(_ms / "mavf_ok.npy"); _nv, _nf = len(_vok), len(_fok)
    assert len(_r) == _nv + _nf, "rows_mav layout"; _E = T3 / "caches/ext"; shutil.copy(_ms / "rows_mav.csv", _E / "rows_mav.csv")
    for _e in ("simam", "sam100ft", "r293", "ecapa"):
        _v = _np.load(_ms / f"mavv_{_e}.float32.npy"); assert len(_v) == _nv and _np.isfinite(_v).all()
        _np.save(_E / f"mav_voice_{_e}.float32.npy", _np.vstack([_v, _np.zeros((_nf, _v.shape[1]), _np.float32)])); _np.save(_E / f"mav_voice_{_e}.ok.npy", _np.concatenate([_vok, _np.zeros(_nf, bool)]))
    _np.save(_E / "mav_face_clip.float32.npy", _np.zeros((_nv + _nf, 768), _np.float32)); _np.save(_E / "mav_face_clip.ok.npy", _np.concatenate([_np.zeros(_nv, bool), _fok]))
    log(f"mav source installed: {_nv} voice rows ({int(_vok.sum())} kept), {_nf} face rows ({int(_fok.sum())} ok); face cache = placeholder (identity check, --nocache mav)")
'''.replace("__EMB__", json.dumps(EMB))
a = '    sh(f"df -h {TMP} | tail -1", "extract.log"); log("STAGE extract done")\n'
head = g["head"]; assert head.count(a) == 1; head = head.replace(a, INJECT + a)
OV = {"ncamtu06": ['flag-t3-overlay50', 'flag-t3-overlay66']}.get(USER, ['flag-t3-overlay56'])  # ncamtu06 has no overlay56 (overlay56 = overlay50 - FaRL/LoRA files + code)
if OV != ['flag-t3-overlay56']:
    o = 'tree("flag-t3-overlay56", "overlay56.tar")'; assert head.count(o) == 1
    head = head.replace(o, '; '.join(f'tree("{d}", "{d.split("-")[-1]}.tar")' for d in OV))
PRE = g["PRE"].replace("__CFG__", f"{BUILD or 't3_c33a'}.json").replace("'../caches/ext/exclude_bengali.csv'", "'../caches/ext/exclude_ultra.csv'")
WORK = PRE + '''    _mv = sorted({str(p.parent) for t in ("mavceleb_v1_complete-001", "mavceleb_v2_complete-002") for p in find(t) if p.is_dir()}); assert _mv, "MAV images not mounted"
    roots += " mav=" + ",".join(_mv); log(f"mav image roots {_mv}")
    FT = f"python -X faulthandler -u t3_ft_clip.py --fold -1 --tag __TAG____EXTRA__ --device cuda --workers 4 --roots {roots}"
    t = time.time(); sh(f"cd '{SC}' && T3_FT_DEBUG_N=256 {FT} --steps 20 --out {TMP}/smoke", "smoke.log")
    import re as _re
    m = {int(k): float(v) for v, k in _re.findall(r"\\[\\s*([0-9.]+)m\\] fold -1: step (\\d+)/20", (LG / "smoke.log").read_text())}
    sps = (m[19] - m[0]) * 60 / 19; log(f"STAGE smoke done in {(time.time() - t) / 60:.1f} min: {sps:.1f} s/step -> 1515 steps ~{1515 * sps / 3600:.1f} h")
    assert 1515 * sps < 8 * 3600, "too slow for the 12 h limit (LoRA + encode + build); stopping early"
    _m = _re.search(r"source mav: (\\d+) identities, (\\d+)/(\\d+) face rows", (LG / "smoke.log").read_text())
    assert _m and int(_m.group(1)) >= 100 and int(_m.group(2)) >= .9 * int(_m.group(3)), f"mav source check failed: {_m and _m.groups()}"
    log(f"STAGE mav check ok: {_m.groups()}")
    _x = _re.search(r"source mav: forced cross-language voice pairs for (\\d+)/(\\d+) identities", (LG / "smoke.log").read_text())
    assert _x and int(_x.group(1)) >= 50, f"xlang check failed: {_x and _x.groups()}"; log(f"STAGE xlang check ok: {_x.groups()}")
    sh(f"cd '{SC}' && {FT} --out {TMP}/ft", "ft.log"); shutil.copy(f"{TMP}/ft/__TAG__.pt", OUT); install("__TAG__", f"{TMP}/ft")
    shutil.copy(f"{TMP}/ft/flag_face___TAG__.float32.npy", OUT)
    with zipfile.ZipFile(OUT / "__TAG___faces.zip", "w", zipfile.ZIP_DEFLATED) as z:  # external face arrays (the Mac builds t3_c35 from them)
        for s in ("vox", "vox2", "vb", "lrs3", "mav"):
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
           "dataset_sources": ["trncnguyn/flag-t3-cv-bundle", "trncnguyn/flag-t3-cv-extra", "trncnguyn/flag-t3-ft-local", "trncnguyn/flag-2027-data"] + [f"{USER}/{d}" for d in OV]
                              + [f"{USER}/flag-t3-mavsrc", "KAGGLE_USER/external-data-mavceleb"],
           "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d)
