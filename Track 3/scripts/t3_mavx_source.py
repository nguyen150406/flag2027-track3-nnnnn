"""MAV-Celeb v1 English as a TRAINING source "mavx" (user 13/11 "làm thử đi": fix the English cells at the representation level).
People: the mavx subset (caches/ext/mavx_subset.csv) minus the 16 MV hold-out ids (caches/ext/mav_holdout.csv; validation only), minus the
Bengali-LID-flagged ids, minus the 3 VoxCeleb2-dev duplicates (t3_mav_english.VOX2_DUP) -> ~39 TRAIN people, English voices only.
Features: per-file arrays already extracted for outputs/mav_en with the eval extractor (same encoders as every other cache).
Writes caches/ext/rows_mavx.csv (identity, gender, language, video, voice, face) + mavx_face_<n>.{float32,ok}.npy + mavx_voice_<n>.{float32,ok}.npy
in the layout of t3data.ext_source / t3_r2.ext_full: row j of an identity holds its j-th face and j-th voice (the shorter list is padded
with ok = False). Video ids in the paths are alias-merged (t3_mav_english.video_groups), so cross-video pairs are real.
  python t3_mavx_source.py <mav_en dir> <lid_mavx.csv>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import t3data as D, t3_mav_english as ME
import os
ALL = os.environ.get("T3_MAV_ALL") == "1"; NAME = "mavy" if ALL else "mavx"   # 13/11 final-refit option: "mavy" = mavx + the MV hold-out ids (no MAV validation left)

FACES = {"clipnf": "feat/face_clipnf.npy", "clipnf2": "feat/face_clipnf2.npy", "farl": "feat/face_farl.npy", "clip": "feat/face_clip.npy", "ibv": "ultra/ib_face.npy"}
VOICES = {"simam": "feat/voice_sim.npy", "sam100ft": "feat/voice_sam100ft.npy", "r293": "feat/voice_r293.npy", "ecapa": "feat/voice_ecapa.npy"}


def main(mdir, lid_csv):
    mdir = Path(mdir); T = pd.read_csv(mdir / "rows.csv"); ME.D = D
    S, bad = ME.subset(lid_csv); ho = set(pd.read_csv(D.EXT / "mav_holdout.csv").identity.astype(str)); S = S[~S.identity.isin(ho)] if not ALL else S
    if ALL: ho = set()
    media = Path(T.face.iloc[0].split("/faces/")[0]); vg = ME.video_groups(S, media)
    S = S.assign(path=[str(media / ("faces" if k == "face" else "voices") / r) for k, r in zip(S.kind, S.relpath)], gvid=[vg[(i, v)] for i, v in zip(S.identity, S.video)])
    fo = {p: i for i, p in enumerate(T.face)}; vo = {p: i for i, p in enumerate(T.voice)}
    S = S[[p in (fo if k == "face" else vo) for p, k in zip(S.path, S.kind)]]          # only files with extracted features
    g = D.gender_map(); rows, fsrc, vsrc = [], [], []
    for idn, x in S.groupby("identity"):
        f, v = x[x.kind == "face"], x[x.kind == "voice"]; n = max(len(f), len(v))
        for j in range(n):
            fr, vr = f.iloc[j % len(f)], v.iloc[j % len(v)]
            rows.append(dict(identity=idn, gender="m" if g[idn] else "f", language="English", video=vr.gvid,
                             voice=f"{idn}/English/{vr.gvid}/{Path(vr.relpath).name}", face=f"{idn}/English/{fr.gvid}/{Path(fr.relpath).name}"))
            fsrc.append((fo[fr.path], j < len(f))); vsrc.append((vo[vr.path], j < len(v)))
    R = pd.DataFrame(rows); fi, fok = map(np.array, zip(*fsrc)); vi, vok = map(np.array, zip(*vsrc))
    assert not set(R.identity) & (ho | ME.VOX2_DUP | set(bad))
    R.to_csv(D.EXT / f"rows_{NAME}.csv", index=False)
    for n, rel in FACES.items():
        a = np.load(mdir / rel)[fi].astype(np.float32); ok = fok & np.isfinite(a).all(1); np.save(D.EXT / f"{NAME}_face_{n}.float32.npy", a); np.save(D.EXT / f"{NAME}_face_{n}.ok.npy", ok)
    for n, rel in VOICES.items():
        a = np.load(mdir / rel)[vi].astype(np.float32); ok = vok & np.isfinite(a).all(1); np.save(D.EXT / f"{NAME}_voice_{n}.float32.npy", a); np.save(D.EXT / f"{NAME}_voice_{n}.ok.npy", ok)
    print(f"{NAME}: {R.identity.nunique()} ids ({(R.drop_duplicates('identity').gender == 'm').sum()} m), {len(R)} rows, faces ok {fok.sum()}, voices ok {vok.sum()}, "
          f"videos {R.video.nunique()}")


if __name__ == "__main__":
    main(*sys.argv[1:])
