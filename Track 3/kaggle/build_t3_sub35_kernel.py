"""SUB35 on Kaggle (account ncamtu06, user-approved 2026-10-15): WavLM-SV voice branch ('wsv', microsoft/wavlm-base-plus-sv) paired with
the LoRA faces (v1 clipft, clipsa), as a NEW ensemble member next to sub33a. Kernel stages:
  vox2 audio first : VoxCeleb2 clips of rows_vox2 (Bengali-excluded speakers skipped) by range request from the HF mirror (gated since
                     2026-10-13) through SIGNED URLs made at build time with the user's token (the token never leaves the Mac; the URLs
                     expire ~1 h after the build, so this runs before anything else); ffmpeg 16 kHz mono s16
  VoxBlink         : wav + img from the trncnguyn kernel output (vb_media.zip), as SUB26/SUB33 but the wavs are kept
  wsv              : t3_extract_wavlmsv.py -> wsv voices of FLAG (flag-2027-data) + vox2 + vb (pair-local, one file at a time)
  lora_ft / lora_sa: t3_ft_clip.py --load (encode only) -> LoRA faces of FLAG + vox2/vb
  build            : configs/t3_sub35.json (clipft_wsv, clipsa_wsv; aux vox2/vb; face dropout .55; 10 projectors)
Inputs: trncnguyn/flag-t3-cv-bundle, -cv-extra, -ft-local, flag-2027-data, ncamtu06/flag-t3-overlay33 (LoRA checkpoints, code) and
ncamtu06/flag-t3-overlay35 (current code, configs/t3_sub35.json, caches/ext/vox2_manifest.csv), extracted in that order.
Usage: python build_t3_sub35_kernel.py   (makes fresh signed HF URLs + Kaggle links; push right away)"""
import json, re
from pathlib import Path
import requests
K = Path(__file__).parent
src = (K / "build_t3_sub26_kernel.py").read_text()
links = {}; exec(src[src.index("def links():"):src.index("CODE = r'''")], links); links = links["links"]
CODE = re.search(r"CODE = r'''(.*?)'''", src, re.S).group(1)
for a, b in (('tree("flag-t3-overlay26", "overlay26.tar")', 'tree("flag-t3-overlay33", "overlay33.tar"); tree("flag-t3-overlay35", "overlay35.tar")'),
             ('z.extractall(TMP / "vbm", [m for m in z.namelist() if m.startswith("img/")])', 'z.extractall(TMP / "vbm", [m for m in z.namelist() if m.startswith(("img/", "wav/"))])'),
             ('shutil.rmtree(DL); shutil.rmtree(TMP / "vbm"); log("STAGE download done")',
              '(TMP / "aud").mkdir(exist_ok=True); shutil.move(str(TMP / "vbm/wav"), str(TMP / "aud/vb")); shutil.rmtree(DL); shutil.rmtree(TMP / "vbm"); DL.mkdir(); log("STAGE download done")')):
    assert CODE.count(a) == 1, a[:50]; CODE = CODE.replace(a, b)
head, tail = CODE[:CODE.index("    pre = (")], CODE[CODE.index("except Exception as e:"):]
# the VoxCeleb2 fetch must start before the signed URLs expire -> move it in front of the Kaggle-output downloads
dl0 = head.index("    def local(zipname, dirname):")
VOX2 = r'''    import pandas as pd, struct, zlib, threading
    from concurrent.futures import as_completed
    sh("pip install -q imageio-ffmpeg 2>&1 | tail -1", "extract.log"); import imageio_ffmpeg; FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    HFU = __HFURLS__; EXT = T3 / "caches/ext"; AUD = TMP / "aud"; (AUD / "vox2").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(EXT / "vox2_manifest.csv"); rows = pd.read_csv(EXT / "rows_vox2.csv")
    exb = set(pd.read_csv(EXT / "exclude_bengali.csv").query("src == 'vox2'").identity.astype(str))
    need = sorted(set(rows.clip_index[~rows.identity.isin(exb)].astype(int))); REC = list(man.itertuples(index=False))  # namedtuples (.size!)
    assert all(REC[i].identity == r for i, r in rows.drop_duplicates("clip_index").set_index("clip_index").identity.loc[need].items()), "manifest/rows mismatch"
    tls = threading.local()
    def clip(r):
        if not hasattr(tls, "s"): tls.s = requests.Session()
        url = HFU[int(r.zip)]; start = int(r.off); end = start + 30 + len(r.fname.encode()) + 1024 + int(r.csize); err = None
        for k in range(4):
            try:
                b = tls.s.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120); b.raise_for_status(); b = b.content; assert b[:4] == b"PK\x03\x04"
                n, m = struct.unpack("<HH", b[26:30]); o = 30 + n + m; d = b[o:o + int(r.csize)]
                if len(d) < int(r.csize): d += tls.s.get(url, headers={"Range": f"bytes={start + o + len(d)}-{start + o + int(r.csize) - 1}"}, timeout=120).content
                d = zlib.decompressobj(-15).decompress(d) if int(r.ctype) == 8 else d; assert len(d) == int(r.size); return d
            except Exception as e:
                err = e; time.sleep(2 ** k)
        raise err
    def work(i):
        wav = AUD / "vox2" / f"{i}.wav"
        try:
            tmp = DL / f"v2_{i}.mp4"; tmp.write_bytes(clip(REC[i]))
            p = subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(tmp), "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(wav)], capture_output=True, text=True)
            tmp.unlink(); return p.returncode == 0 and wav.exists(), p.stderr[-120:]
        except Exception as e:
            return False, repr(e)[:160]
    okc, errs, t = 0, [], time.time()
    with ThreadPoolExecutor(32) as ex_:
        futs = [ex_.submit(work, i) for i in need]
        for k, fu in enumerate(as_completed(futs)):
            g, msg = fu.result(); okc += g
            if not g and len(errs) < 5: errs.append(msg); log(f"vox2 clip error: {msg}")
            if k % 3000 == 0: log(f"vox2 audio {k}/{len(need)} ok {okc} ({(time.time() - t) / 60:.1f} min)")
            if k == 300 and okc < 150:
                ex_.shutdown(wait=False, cancel_futures=True); raise RuntimeError(f"VoxCeleb2 fetch failing: {errs[:1]}")
    log(f"STAGE vox2 audio: {okc}/{len(need)} clips in {(time.time() - t) / 60:.1f} min"); assert okc >= .9 * len(need), "too many VoxCeleb2 clips failed"
'''
head = head[:dl0] + VOX2 + head[dl0:]
WORK = r'''    pre = ("import json, numpy as np, t3_build as TB; cfg = json.load(open('../configs/t3_sub35.json')); A = TB.AGE; "
           "TB.age_offset(np.load(A / 'train_face_age.npy'), np.load(A / 'train_voice_age.npy')); "
           "[np.load(A / f'{p}_{l}_{m}_age.npy') for (p, l) in TB.EXPECTED for m in ('face', 'voice')]; "
           "fa, va = TB.age_v2(cfg); pf, pv = TB.gender_v2(); assert np.isfinite(fa).all() and len(pf) == 11349; import t3data as D; assert 'wsv' in D.FLAG_EXTRA_VOICE; print('preflight ok')")
    sh(f'cd "{SC}" && python -c "{pre}"', "preflight.log"); log("STAGE preflight done")
    S = T3 / "submissions"; OUTD = OUT
    # WavLM-SV voices first (the new artifact; saved to /kaggle/working at once)
    sh(f"cd '{SC}' && timeout -s ABRT 5400 python -X faulthandler -u t3_extract_wavlmsv.py --out {TMP}/wsv --flag_data {dev_root.parent} --roots vox2={AUD}/vox2 vb={AUD}/vb --device cuda --workers 4", "wsv.log")
    shutil.copyfile(f"{TMP}/wsv/flag_voice_wsv.float32.npy", EXT / "flag_voice_wsv.float32.npy")
    for s in ("vox2", "vb"):
        for e in ("float32", "ok"): shutil.copyfile(f"{TMP}/wsv/{s}_voice_wsv.{e}.npy", EXT / f"{s}_voice_wsv.{e}.npy")
    with zipfile.ZipFile(OUT / "wsv_embeddings.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in Path(f"{TMP}/wsv").iterdir(): z.write(p, p.name)
    log("STAGE wsv done")
    roots = f"flag={flag_root} flagdev={dev_root} vox2={IMG / 'vox2_faces'} vb={vb_root}"
    for tag, ck in (("clipft", "t3_sub22/clipft.pt"), ("clipsa", "t3_sub26/clipsa.pt")):  # LoRA faces, encode only
        sh(f"cd '{SC}' && timeout -s ABRT 4800 python -X faulthandler -u t3_ft_clip.py --fold -1 --load '{S}/{ck}' --tag {tag} --sources vox2,vb --device cuda --workers 4 --roots {roots} --out {TMP}/enc_{tag}", f"lora_{tag}.log")
        (T3 / f"caches/{tag}").mkdir(exist_ok=True); shutil.copyfile(f"{TMP}/enc_{tag}/flag_face_{tag}.float32.npy", T3 / f"caches/{tag}/face_embeddings.float32.npy")
        for s in ("vox2", "vb"):
            for e in ("float32", "ok"): shutil.copyfile(f"{TMP}/enc_{tag}/{s}_face_{tag}.{e}.npy", EXT / f"{s}_face_{tag}.{e}.npy")
        log(f"STAGE lora_{tag} done")
    sh(f"cd '{SC}' && T3_TRAIN_DEVICE=cuda T3_AUX_DEVICE=0 T3_NGPU={ng} OMP_NUM_THREADS=1 timeout -s ABRT 5400 python -X faulthandler -u t3_build_par.py ../configs/t3_sub35.json ../submissions/t3_sub35 4", "build.log")
    for tag in ("clipft", "clipsa"): shutil.copyfile(f"{TMP}/enc_{tag}/flag_face_{tag}.float32.npy", S / "t3_sub35" / f"flag_face_{tag}.float32.npy")
    shutil.copyfile(f"{TMP}/wsv/flag_voice_wsv.float32.npy", S / "t3_sub35/flag_voice_wsv.float32.npy"); shutil.copyfile(f"{TMP}/wsv/diag_wsv.json", S / "t3_sub35/diag_wsv.json")
    with zipfile.ZipFile(OUT / "sub35_out.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for d in (S / "t3_sub35", S / "t3_sub35.models"):
            for p in d.rglob("*"):
                if p.is_file(): z.write(p, p.relative_to(T3))
    log(f"JOB DONE -> {OUT / 'sub35_out.zip'} {(OUT / 'sub35_out.zip').stat().st_size / 1e6:.0f} MB")
'''
def hf_urls():  # signed CDN URLs of the 6 VoxCeleb2 mp4 zips (the token stays on the Mac; URLs expire ~1 h)
    tok = (Path.home() / ".hf_lrs3_token").read_text().strip(); out = []
    for i in range(1, 7):
        r = requests.head(f"https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_mp4_{i}.zip", allow_redirects=False, timeout=60,
                          headers={"Authorization": f"Bearer {tok}"}); r.raise_for_status(); assert r.status_code in (301, 302, 307), r.status_code; out.append(r.headers["Location"])
    return out
urls = links(); hfu = hf_urls()
code = (head + WORK + tail).replace("__URLS__", json.dumps(urls)).replace("__HFURLS__", json.dumps(hfu)); compile(code, "sub35", "exec")
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = K / "t3_sub35"; d.mkdir(exist_ok=True); json.dump(nb, open(d / "t3_sub35.ipynb", "w"), indent=1)
json.dump({"id": "ncamtu06/flag-t3-sub35", "title": "FLAG T3 sub35", "code_file": "t3_sub35.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": ["trncnguyn/flag-t3-cv-bundle", "trncnguyn/flag-t3-cv-extra", "trncnguyn/flag-t3-ft-local", "trncnguyn/flag-2027-data",
                               "ncamtu06/flag-t3-overlay33", "ncamtu06/flag-t3-overlay35"], "competition_sources": [], "kernel_sources": []}, open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d, sorted(urls), len(hfu), "signed HF urls")
