"""Eval-phase scorer for the frozen sub60 system (user 2026-10-29 "bắt đầu viết code đi"): no training, no test statistics.
The 125 saved projectors (25 branches x 5 seeds, submissions/<build>.models) score every pair independently; age/gender
terms as t3_build v2 (ridge/standardisation/offset fitted on the 6,485 FLAG train rows only); weights = t3_sub60.
  python t3_eval.py table <data_root> <rows.csv>     pair list of <root>/<proto>/<Lang>_test.txt (+ META dev row if any)
  python t3_eval.py slice <rows.csv> <feat_dir>      dev rehearsal: features taken from the existing dev caches
  python t3_eval.py score <rows.csv> <feat_dir> <out_dir>
feat_dir holds one array per feature, rows in rows.csv order (see FEATS); extraction: t3_eval_extract.py."""
import sys, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd
import t3_build as TB, t3data as D
from t3lib import META, T3, ROOT
from t3train import pair_cos

S = T3 / "submissions"; SYSTEM = "t3_sub60"
BUILDS = ["t3_sub38", "t3_sub53fl", "t3_sub33a_ft", "t3_sub35", "t3_sub57", "t3_sub58"]  # raw builds behind every sub60 branch
FACES = ["farl", "clip", "clipft", "clipsa", "clipft12", "clipsa12"]
VOICES = ["sim", "ecapa", "sam100ft", "r293", "rdn", "wsv"]
FEATS = [f"face_{n}" for n in FACES] + [f"voice_{n}" for n in VOICES] + \
        ["vit_age", "mivolo_age", "mivolo_pmale", "a24_age", "a24_gender", "a24_emb", "a6_emb"]
LORA = {"clipft": S / "t3_sub33a_ft/flag_face_clipft.float32.npy", "clipsa": S / "t3_sub35/flag_face_clipsa.float32.npy",  # the arrays the
        "clipft12": S / "t3_sub57/flag_face_clipft12.float32.npy", "clipsa12": S / "t3_sub58/flag_face_clipsa12.float32.npy"}  # projectors saw
import os
if os.environ.get("T3_SYSSET"):  # e.g. configs/sysset_clean.json: the same sub60 layout/weights on the clean-retrained builds and LoRAs
    _j = json.load(open(os.environ["T3_SYSSET"])); BUILDS = _j["builds"]; LORA = {k: S / v for k, v in _j["lora_arrays"].items()}
CELLS = list(TB.EXPECTED)  # (protocol, language) -> file name


def table(root, out):
    dev = META[META.partition.eq("dev")].set_index(["protocol", "language", "pair_id"]).row_id
    rows = []
    for proto, lang in CELLS:
        for l in (Path(root) / proto / f"{lang}_test.txt").read_text().splitlines():
            pid, v, f = l.split()
            rows.append(dict(protocol=proto, language=lang, pair_id=pid, voice=str(Path(root) / proto / v), face=str(Path(root) / proto / f),
                             meta_row=int(dev.get((proto, lang, pid), -1))))
    t = pd.DataFrame(rows); assert not t.duplicated(["protocol", "language", "pair_id"]).any()
    t.to_csv(out, index=False); print(f"{len(t)} pairs", t.groupby(["protocol", "language"]).size().to_dict(), "| META dev rows", int((t.meta_row >= 0).sum()))


def slice_dev(rows_csv, fd):
    r = pd.read_csv(rows_csv).meta_row.to_numpy(); assert (r >= 6485).all(), "slice needs dev rows of META"
    fd = Path(fd); fd.mkdir(parents=True, exist_ok=True)
    a24, a6 = np.load(T3 / "caches/flag_voiceage24.npz"), np.load(T3 / "caches/flag_voiceage6.npz")
    src = {**{f"face_{n}": (lambda n=n: np.load(LORA[n], mmap_mode="r") if n in LORA else D.face(n)) for n in FACES},
           **{f"voice_{n}": (lambda n=n: D.voice_full(n)) for n in VOICES},
           "vit_age": TB.all_rows_face_vit, "mivolo_age": lambda: np.load(T3 / "caches/mivolo_face/face_age.npy"),
           "mivolo_pmale": lambda: np.load(T3 / "caches/mivolo_face/face_pmale.npy"),
           "a24_age": lambda: a24["age"], "a24_gender": lambda: a24["gender"], "a24_emb": lambda: a24["emb"], "a6_emb": lambda: a6["emb"]}
    for k in FEATS:
        x = np.asarray(src[k]()); assert len(x) == len(META), (k, x.shape); np.save(fd / f"{k}.npy", x[r])
    print("wrote", len(FEATS), "features for", len(r), "rows to", fd)


def branches():
    """sub60 branch name -> (build, branch index, face, voice, seeds), from the raw builds' own configs."""
    out = {}
    for b in BUILDS:
        c = json.load(open(S / b / "provenance.json"))["config"]
        for i, br in enumerate(c["branches"]): out.setdefault(br["name"], (b, i, br["face"], br["voice"], c["seeds"]))
    return out


def score(rows_csv, fd, out_dir):
    t, fd, out = pd.read_csv(rows_csv), Path(fd), Path(out_dir)
    if out.exists(): raise FileExistsError(out)
    X = {k: np.load(fd / f"{k}.npy") for k in FEATS}
    for k, x in X.items(): assert len(x) == len(t) and np.isfinite(x).all(), k
    W = json.load(open(S / SYSTEM / "provenance.json"))["weights"]; BR = branches(); names = [k for k in W if k not in ("age", "gender")]
    assert set(names) <= set(BR), set(names) - set(BR)
    def voice(n): return np.hstack([D._l2(voice(m)) for m in D.CAT[n]]) if n in D.CAT else X[f"voice_{n}"]
    import torch
    comp_all = {}
    for n in names:
        b, i, fn, vn, seeds = BR[n]; F, V = X[f"face_{fn}"], voice(vn)
        ms = [torch.load(S / f"{b}.models/b{i}_s{s}.pt", weights_only=False) for s in seeds]
        comp_all[n] = np.mean([pair_cos(m, F, V, np.arange(len(t)), np.arange(len(t))) for m in ms], 0)
        k = np.arange(len(t))[::-1][:64]  # pair-locality: scoring a reversed subset alone gives the same values
        assert np.abs(pair_cos(ms[0], F, V, k, k) - pair_cos(ms[0], F, V, np.arange(len(t)), np.arange(len(t)))[k]).max() < 1e-5, n
    # age/gender (t3_build v2): every fitted quantity comes from the FLAG train rows of the existing caches
    a24, a6 = np.load(T3 / "caches/flag_voiceage24.npz"), np.load(T3 / "caches/flag_voiceage6.npz")
    fa_tr = (np.load(TB.AGE / "train_face_age.npy") + np.load(T3 / "caches/mivolo_face/face_age.npy")[:6485] / 100) / 2
    fa = (X["vit_age"] + X["mivolo_age"] / 100) / 2
    va = TB.voice_age_v2(fa_tr, TB.age_hidden(a24["emb"][:6485], a6["emb"][:6485]), a24["age"][:6485], TB.age_hidden(X["a24_emb"], X["a6_emb"]), X["a24_age"])
    comp_all["age"] = -np.abs(fa - va)
    pf, g = X["mivolo_pmale"], X["a24_gender"]; pv = g[:, 1] / (g[:, 0] + g[:, 1])
    gender = np.log(np.clip(pf * pv + (1 - pf) * (1 - pv), 1e-4, 1))
    out.mkdir(parents=True); files = {}
    for (proto, lang), name in TB.EXPECTED.items():
        r = np.flatnonzero(t.protocol.eq(proto) & t.language.eq(lang)); c = {k: v[r] for k, v in comp_all.items()}
        if proto == "no_gender" and W.get("gender", 0): c["gender"] = gender[r]
        s = sum(W[b] * c[b] for b in names) + W["age"] * c["age"] + (W.get("gender", 0) * c["gender"] if "gender" in c else 0)
        ids = t.pair_id.to_numpy()[r]; np.savez(out / f"components_{proto}_{lang}.npz", ids=ids, **c)
        p = out / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("".join(f"{i} {-v:.8f}\n" for i, v in zip(ids, s)))
        files[name] = dict(rows=len(r), sha256=TB.sha(p))
    with zipfile.ZipFile(out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in TB.EXPECTED.values(): z.write(out / name, name)
    json.dump(dict(system=SYSTEM, sysset=os.environ.get("T3_SYSSET", ""), builds=BUILDS, weights=W, rows_csv=str(rows_csv), features={k: TB.sha(fd / f"{k}.npy") for k in FEATS},
                   files=files, zip_sha256=TB.sha(out / "submission.zip"), pair_local=True, test_statistics_used=False),
              open(out / "provenance.json", "w"), indent=1)
    TB.named_copy(out); print("wrote", out / "submission.zip", TB.sha(out / "submission.zip"))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"table": table, "slice": slice_dev, "score": score}[cmd](*args)
