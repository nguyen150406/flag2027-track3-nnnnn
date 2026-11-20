"""SUB56 = SigLIP2 LoRA (user 2026-10-20 "oke chạy đi, seed 2 chạy nguyntrnduck"): a second fine-tuned face BACKBONE next to the CLIP
LoRAs (all earlier 3rd-LoRA tries were CLIP again). google/siglip2-so400m-patch14-224 with the CLIP LoRA v1 recipe (r16, last 6 blocks,
1515 steps, FLAG + vox/vox2/vb/lrs3, filtered data), then the 5 clipft-style branches (face dropout .55). Two LoRA seeds because LoRA
seed noise on dev is ~0.6-0.7; one kernel per account, SUB26 template + <user>/flag-t3-overlay56 (overlay50 minus FaRL/LoRA files +
current code + configs t3_sub56s1/s2):
  trnlhvy/flag-t3-sub56s1      : sigft1, LoRA seed 2027
  nguyntrnduck/flag-t3-sub56s2 : sigft2, LoRA seed 7
After the 20-step smoke the kernel stops early if the measured speed would not fit Kaggle's 12 h limit.
Usage: python build_t3_sub56_kernels.py  (default ~/.kaggle token = trncnguyn for the image links; push right away)"""
import json, re
from pathlib import Path
K = Path(__file__).parent
src = (K / "build_t3_sub26_kernel.py").read_text()
links = {}; exec(src[src.index("def links():"):src.index("CODE = r'''")], links); links = links["links"]
CODE = re.search(r"CODE = r'''(.*?)'''", src, re.S).group(1)
a = 'tree("flag-t3-overlay26", "overlay26.tar")'; assert CODE.count(a) == 1; CODE = CODE.replace(a, 'tree("flag-t3-overlay56", "overlay56.tar")')
head, tail = CODE[:CODE.index("    pre = (")], CODE[CODE.index("except Exception as e:"):]
PRE = '''    pre = ("import json, numpy as np, pandas as pd, t3_build as TB, t3data as D; cfg = json.load(open('../configs/__CFG__')); A = TB.AGE; "
           "TB.age_offset(np.load(A / 'train_face_age.npy'), np.load(A / 'train_voice_age.npy')); "
           "[np.load(A / f'{p}_{l}_{m}_age.npy') for (p, l) in TB.EXPECTED for m in ('face', 'voice')]; "
           "fa, va = TB.age_v2(cfg); pf, pv = TB.gender_v2(); assert np.isfinite(fa).all() and len(pf) == 11349; "
           "assert all(b['face'] in D.FACE for b in cfg['branches']); "
           "e = pd.read_csv('../caches/ext/exclude_bengali.csv'); print('preflight ok', len(e), 'excluded')")
    sh(f'cd "{SC}" && python -c "{pre}"', "preflight.log"); log("STAGE preflight done")
    roots = f"flag={flag_root} flagdev={dev_root} lrs3={lrs3_root} vox={IMG / 'vox1_faces'} vox2={IMG / 'vox2_faces'} vb={vb_root}"
    S = T3 / "submissions"; EXT = T3 / "caches/ext"
    def install(tag, d):
        (T3 / f"caches/{tag}").mkdir(exist_ok=True); shutil.copyfile(f"{d}/flag_face_{tag}.float32.npy", T3 / f"caches/{tag}/face_embeddings.float32.npy")
        for s in ("vox", "vox2", "vb", "lrs3"):
            for e in ("float32", "ok"): shutil.copyfile(f"{d}/{s}_face_{tag}.{e}.npy", EXT / f"{s}_face_{tag}.{e}.npy")
        log(f"STAGE install {tag} done")
    def build_pack(cfg, out, extra):
        sh(f"cd '{SC}' && T3_TRAIN_DEVICE=cuda T3_AUX_DEVICE=0 T3_NGPU={ng} OMP_NUM_THREADS=1 python -u t3_build_par.py ../configs/{cfg} ../submissions/{out} 4", "build.log")
        log("STAGE build done")
        for f in extra: shutil.copy(f, S / out)
        with zipfile.ZipFile(OUT / f"{out[3:]}_out.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for d in (S / out, S / f"{out}.models"):
                for p in d.rglob("*"):
                    if p.is_file(): z.write(p, p.relative_to(T3))
        log(f"JOB DONE -> {OUT / (out[3:] + '_out.zip')} {(OUT / (out[3:] + '_out.zip')).stat().st_size / 1e6:.0f} MB")
'''
def work(tag, seed, cfg):
    return PRE.replace("__CFG__", cfg) + f'''    FT = f"python -X faulthandler -u t3_ft_clip.py --encoder siglip --fold -1 --seed {seed} --tag {tag} --device cuda --workers 4 --roots {{roots}}"
    t = time.time(); sh(f"cd '{{SC}}' && T3_FT_DEBUG_N=256 {{FT}} --steps 20 --out {{TMP}}/smoke", "smoke.log")
    import re as _re
    m = {{int(k): float(v) for v, k in _re.findall(r"\\[\\s*([0-9.]+)m\\] fold -1: step (\\d+)/20", (LG / "smoke.log").read_text())}}
    sps = (m[19] - m[0]) * 60 / 19; log(f"STAGE smoke done in {{(time.time() - t) / 60:.1f}} min: {{sps:.1f}} s/step -> 1515 steps ~{{1515 * sps / 3600:.1f}} h")
    assert 1515 * sps < 7.5 * 3600, "too slow for the 12 h limit (LoRA + encode + build); stopping early"
    sh(f"cd '{{SC}}' && {{FT}} --out {{TMP}}/ft", "ft.log"); shutil.copy(f"{{TMP}}/ft/{tag}.pt", OUT); install("{tag}", f"{{TMP}}/ft")
    shutil.copy(f"{{TMP}}/ft/flag_face_{tag}.float32.npy", OUT)
    build_pack("{cfg}", "{cfg[:-5]}", [f"{{TMP}}/ft/{tag}.pt", f"{{TMP}}/ft/flag_face_{tag}.float32.npy"])
'''
urls = links()
BASE_DS = ["trncnguyn/flag-t3-cv-bundle", "trncnguyn/flag-t3-cv-extra", "trncnguyn/flag-t3-ft-local", "trncnguyn/flag-2027-data"]
for user, slug, w in (("trnlhvy", "sub56s1", work("sigft1", 2027, "t3_sub56s1.json")), ("nguyntrnduck", "sub56s2", work("sigft2", 7, "t3_sub56s2.json"))):
    code = (head + w + tail).replace("__URLS__", json.dumps(urls)); compile(code, slug, "exec")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    d = K / f"t3_{slug}"; d.mkdir(exist_ok=True); json.dump(nb, open(d / f"t3_{slug}.ipynb", "w"), indent=1)
    json.dump({"id": f"{user}/flag-t3-{slug}", "title": f"FLAG T3 {slug}", "code_file": f"t3_{slug}.ipynb", "language": "python", "kernel_type": "notebook",
               "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": BASE_DS + [f"{user}/flag-t3-overlay56"],
               "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
    print("built", d)
