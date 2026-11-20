"""English proxy from MAV-Celeb v1 (user 2026-11-12: fix the English cells; our own design).
MAV v1 = Pakistani people speaking English / Urdu, recorded like FLAG; none of our models trained on it (LoRA: vox/vox2/vb/lrs3 faces,
heads: vox/vox2/vb, R2: vox2/vb), so scores on MAV pairs come from FINAL models on unseen people, exactly as on the eval set.
Trials (fixed seed, decided before any result): every ENGLISH voice clip of the subset gets one positive (a face of the same person from a
DIFFERENT video) and one negative (a face of another person; gender protocol: same gender), per protocol. Persons: 16 held out = MV
(caches/ext/mav_holdout.csv, validation only), the others = TR (may fit the English fuser). Identities flagged by the Bengali LID are dropped.
  python t3_mav_english.py trials <media dir> <lid_mavx.csv> <out dir>   -> <out>/rows.csv (+ protocol, language, pair_id, y, split)
Then, with the SAME code as the eval set:
  t3_eval_extract.py run <out>/rows.csv <out>/feat farl clipnf clipnf2 sim sam100ft r293 ecapa vitage mivolo a24 a6
  t3_imagebind_extract.py <out>/feat/_work/faces.csv face ... ; voices.csv voice ...   (mapped to rows by `python t3_mav_english.py ib <out>`)
13/11 review fixes (outputs/mav_en was built before them): (1) t3data.gender_map had MAV v3 genders overwriting 25/64 v1 ids -> 45% of the
"gender" negatives were cross-gender (fixed in t3data); (2) 3 MAV people are VoxCeleb2-dev people (VOX2_DUP) -> dropped; (3) some voice clips
are byte-identical under two video ids -> video ids sharing a clip are merged before choosing cross-video positives.
  python t3_mav_english.py retrial <old out dir> <lid_mavx.csv> <new out dir>
re-draws the trials with the fixed rules using ONLY files already extracted in <old out dir>, and copies their features by file path (features
are per file, so no re-extraction)."""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; T3 = HERE.parent
sys.path.insert(0, str(HERE))


def lid_flagged(lid_csv):
    """t3_lid_merge rule chosen by the user on 07/11: a clip is flagged when ANY model gives P(bn) + P(as) > .5; one flagged clip drops the identity."""
    L = pd.read_csv(lid_csv); cols = [c for c in L.columns if c.endswith("_pbnas")] or [c for c in L.columns if "bnas" in c]
    assert cols, list(L.columns); flag = (L[cols].astype(float) > .5).any(axis=1)
    ids = L.key.astype(str).str.split("/").str[0]
    return sorted(set(ids[flag])), cols


VOX2_DUP = {"id0025", "id0028", "id0041"}   # = vox2 dev id00844 / id01038 / id05434 (face cos .87-.93, voice cos .83, voice rank 1-2 of 5,994)


def video_groups(S, media):
    """(identity, video) -> group: videos that share a byte-identical voice clip are one video (alias ids)"""
    import hashlib
    par = {}
    def find(x):
        while par.setdefault(x, x) != x: x = par[x]
        return x
    first = {}
    for idn, vid, rel in S[S.kind == "voice"][["identity", "video", "relpath"]].itertuples(index=False):
        h = hashlib.md5((media / "voices" / rel).read_bytes()).hexdigest(); a = find((idn, vid))
        if h in first: par[a] = find(first[h])
        else: first[h] = (idn, vid)
    out = {k: find(k)[1] for k in set(map(tuple, S[["identity", "video"]].values))}
    print("alias video merges:", sum(k[1] != v for k, v in out.items()))
    return out


def make_rows(S, media, bad, out):
    g = D.gender_map(); ho = set(pd.read_csv(T3 / "caches/ext/mav_holdout.csv").identity.astype(str))
    vg = video_groups(S, media); S = S.assign(video=[vg[(i, v)] for i, v in zip(S.identity, S.video)])
    rng = np.random.default_rng(20261112); rows = []
    F = {i: x for i, x in S[S.kind == "face"].groupby("identity")}; V = S[S.kind == "voice"]
    for split, ids in (("MV", sorted(ho & set(S.identity))), ("TR", sorted(set(S.identity) - ho))):
        vv = V[V.identity.isin(ids)]
        for pr in ("no_gender", "gender"):
            for _, v in vv.iterrows():
                f = F[v.identity]; c = f[f.video != v.video]
                if len(c): rows.append(dict(face=str(media / "faces" / c.sample(1, random_state=int(rng.integers(1 << 30))).relpath.iloc[0]),
                                            voice=str(media / "voices" / v.relpath), y=1, protocol=pr, split=split, identity_v=v.identity))
                others = [j for j in ids if j != v.identity and (pr == "no_gender" or bool(g[j]) == bool(g[v.identity]))]
                b = others[rng.integers(len(others))]
                rows.append(dict(face=str(media / "faces" / F[b].sample(1, random_state=int(rng.integers(1 << 30))).relpath.iloc[0]),
                                 voice=str(media / "voices" / v.relpath), y=0, protocol=pr, split=split, identity_v=v.identity))
    R = pd.DataFrame(rows); R["language"] = "English"; R["pair_id"] = [f"mav{i:06d}" for i in range(len(R))]
    R["identity_f"] = R.face.str.extract(r"/faces/(id\d+)/")[0]
    assert (R[R.y == 1].identity_f == R[R.y == 1].identity_v).all()
    gn = R[(R.y == 0) & R.protocol.eq("gender")]; assert all(bool(g[a]) == bool(g[b]) for a, b in zip(gn.identity_v, gn.identity_f)), "gender negatives"
    for p in list(R.face.unique()) + list(R.voice.unique()): assert Path(p).exists(), p
    out.mkdir(parents=True, exist_ok=True); R.to_csv(out / "rows.csv", index=False)
    print(R.groupby(["split", "protocol", "y"]).size().to_string())
    return R


def subset(lid_csv):
    S = pd.read_csv(T3 / "caches/ext/mavx_subset.csv"); bad, cols = lid_flagged(lid_csv); print("LID columns", cols, "flagged ids", bad, "vox2 dup", sorted(VOX2_DUP))
    m1 = pd.read_csv(T3.parent / "external_mavceleb/v1_complete/meta_v1.csv"); g = D.gender_map()
    assert all(bool(g[i]) == (x.lower()[0] == "m") for i, x in zip(m1.ids, m1.gender)), "gender_map must use MAV v1 metadata"
    return S[~S.identity.isin(set(bad) | VOX2_DUP)], bad


def trials(media, lid_csv, out):
    import t3data as D_; globals()["D"] = D_
    S, bad = subset(lid_csv); make_rows(S, Path(media), bad, Path(out))


def retrial(old, lid_csv, new):
    import t3data as D_; globals()["D"] = D_
    old, new = Path(old), Path(new); assert not new.exists(), new; T = pd.read_csv(old / "rows.csv")
    media = Path(T.face.iloc[0].split("/faces/")[0]); S, bad = subset(lid_csv)
    have = set(T.face) | set(T.voice); S = S[[str(media / ("faces" if k == "face" else "voices") / r) in have for k, r in zip(S.kind, S.relpath)]]
    R = make_rows(S, media, bad, new)
    fo = {p: i for i, p in enumerate(T.face)}; vo = {p: i for i, p in enumerate(T.voice)}; fi = R.face.map(fo).to_numpy(); vi = R.voice.map(vo).to_numpy()
    FACE = ("face_", "vit_age", "mivolo_", "ib_face"); VOICE = ("voice_", "a24_", "a6_", "ib_voice")
    for sub in ("feat", "ultra"):
        (new / sub).mkdir()
        for f in sorted((old / sub).glob("*.npy")):
            kind = "f" if f.name.startswith(FACE) else "v" if f.name.startswith(VOICE) else None; assert kind, f.name
            a = np.load(f); assert len(a) == len(T), f; np.save(new / sub / f.name, a[fi if kind == "f" else vi])
    print("copied features for", len(R), "rows")


def ib(out):
    """map the unique-file ImageBind arrays back to rows (same mapping as t3_eval_extract.run)"""
    out = Path(out); t = pd.read_csv(out / "rows.csv"); W = out / "feat/_work"
    uf, fi = np.unique(t.face.astype(str), return_inverse=True); uv, vi = np.unique(t.voice.astype(str), return_inverse=True)
    assert (pd.read_csv(W / "faces.csv").path.astype(str).to_numpy() == uf).all() and (pd.read_csv(W / "voices.csv").path.astype(str).to_numpy() == uv).all()
    (out / "ultra").mkdir(exist_ok=True)
    np.save(out / "ultra/ib_face.npy", np.load(out / "_ib_face_u.npy")[fi]); np.save(out / "ultra/ib_voice.npy", np.load(out / "_ib_voice_u.npy")[vi])
    for n in ("clipnf", "clipnf2"): np.save(out / f"ultra/face_{n}.npy", np.load(out / f"feat/face_{n}.npy"))
    print("mapped", len(t), "rows")


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    {"trials": trials, "ib": ib, "retrial": retrial}[cmd](*a)
