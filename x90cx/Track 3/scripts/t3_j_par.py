"""Parallel T3-J: the canonical 6-branch x 3-seed OOF check of t3_j_full.py, one single-threaded worker per (branch, seed).
Small-matrix training is overhead-bound, so N single-thread processes beat 3 multi-thread ones (~3x throughput on M4).
Usage: T3_RECIPE=combo T3_WSG=.5 python t3_j_par.py <tag> <vox2 source(s), e.g. vox2 or vox2ab or none> [workers=9]
Writes outputs/oof_j_<tag>_other.pkl (all 6 branches; readable by t3_fuse_proj.py / t3_fuse_vs07.py)."""
import os, sys, time, pickle
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import multiprocessing as mp

SPEC = {"v1": dict(src="v1", cap=None), "v3": dict(src="v3", cap=20), "vox": dict(src="vox", cap=None),
        "v1en": dict(src="v1", langs=["English"]), "v3en": dict(src="v3", langs=["English"], cap=20)}
BRANCHES = ["farl:sim", "clip:sim", "clip:r293", "clip:ecapa", "clip:sam100ft", "farl:ecapa"]
if os.environ.get("T3_BRANCHES"): BRANCHES = os.environ["T3_BRANCHES"].split(",")  # e.g. clip:cat4,farl:cat4
SEEDS = (2027, 42, 3407)


def kw_from_env():  # identical to t3_j_full.py
    KW = dict(dim=256, w_proto=.6, w_con=.5)
    if os.environ.get("T3_RECIPE") == "combo":
        KW = dict(KW, margin=.1, voice_dropout=.55, temp=.2, face_dropout=.85, dim=512)
    if os.environ.get("T3_WSG"):
        KW["w_sg"] = float(os.environ["T3_WSG"])
    if os.environ.get("T3_PROJ"):
        KW["proj"] = os.environ["T3_PROJ"]; KW["proj_rank"] = int(os.environ.get("T3_RANK", "64"))
    for _k, _f in (("T3_MARGIN", "margin"), ("T3_TEMP", "temp"), ("T3_FD", "face_dropout"), ("T3_VD", "voice_dropout"),
                   ("T3_WCON", "w_con"), ("T3_WPROTO", "w_proto"), ("T3_PERID", "per_id"), ("T3_LR", "lr"), ("T3_WD", "wd")):
        if os.environ.get(_k):
            KW[_f] = (int if _f == "per_id" else float)(os.environ[_k])
    if os.environ.get("T3_MERGE"):
        KW["merge_aux"] = os.environ["T3_MERGE"] == "1"
    if os.environ.get("T3_WSGAUX"):
        KW["w_sg_aux"] = float(os.environ["T3_WSGAUX"])
    if os.environ.get("T3_DECOUPLE"):  # FLAG positives cross-row (face and voice of different rows of the same ID), like the dev/test trials
        KW["decouple"] = os.environ["T3_DECOUPLE"] == "1"
    for _k, _f in (("T3_WUNI", "w_uni"), ("T3_UNIT", "uni_t"), ("T3_WHUB", "w_hub"),  # training-time hubness reduction
                   ("T3_WSGA", "w_sga"), ("T3_AGETHR", "age_thr")):  # 2026-10-20 review direction A: same gender + close age negatives
        if os.environ.get(_k):
            KW[_f] = float(os.environ[_k])
    for _k, _f in (("T3_AUXIDS", "aux_ids"), ("T3_EPOCHS", "epochs"), ("T3_IDS", "ids_per_batch"), ("T3_HUBK", "hub_k"), ("T3_SGTOPK", "sg_topk")):
        if os.environ.get(_k):
            KW[_f] = int(os.environ[_k])
    if os.environ.get("T3_SNAPS"):  # e.g. "15,40,60" with T3_EPOCHS=80: OOF of the shorter runs too -> oof_j_<tag>_e<E>_*.pkl
        KW["snaps"] = os.environ["T3_SNAPS"]
    if os.environ.get("T3_GENDER"):  # "m"/"f": gender-specific projector (FLAG + external identities of that gender only)
        KW["gender"] = os.environ["T3_GENDER"]
    return KW


def job(args):
    branch, extra, seed = args
    ng = int(os.environ.get("T3_NGPU", "0"))
    if ng and "CUDA_VISIBLE_DEVICES" not in os.environ:  # spread pool workers over the GPUs (set before torch is imported)
        os.environ["CUDA_VISIBLE_DEVICES"] = str((mp.current_process()._identity[0] - 1) % ng)
    import torch; torch.set_num_threads(1)
    from t3lib import SPK, fold_ids, cell
    from t3train import train_projector, pair_cos
    import t3data as D
    fn, vn = branch.split(":")
    base = os.environ.get("T3_BASESRC")  # override the small base sources (default: MAV v1 + MAV v3 + VoxCeleb1)
    base = ["v1a" if vn == "sim" else "v1en", "v3", "vox"] if base is None else [x for x in base.split(",") if x and x != "none"]
    srcs = base + [s for s in extra.split(",") if s != "none"]
    F, V = np.asarray(D.face(fn)[:6485]), np.asarray(D.voice_full(vn)[:6485])
    w2 = float(os.environ.get("T3_VOX2W", "1.0"))
    aux = [D.mav(fn, vn)[0] if s == "v1a" else
           D.ext_source(SPEC[s]["src"], fn, vn, langs=SPEC[s].get("langs"), max_faces_per_video=SPEC[s].get("cap"), w=w2)
           if s in SPEC else D.ext_source(s, fn, vn, w=w2) for s in srcs]  # T3_VOX2W scales every external source (default 1.0)
    KW = kw_from_env()
    if KW.get("gender"):
        aux = D.gender_split(aux, KW["gender"])
    if KW.get("w_sg_aux"):
        aux = [D.attach_gender(a) for a in aux]
    out = {}
    for fold in range(4):  # same model per (fold, seed) as t3train.oof_branch; the seed-mean is taken in main
        fit, _ = fold_ids(fold)
        def score(m, key=()):
            for p in ("standard", "same_gender"):
                fr, _ = cell(fold, p)
                out[(fold, p) + key] = pair_cos(m, F, V, fr.face_row.to_numpy(), fr.voice_row.to_numpy())
        m = train_projector(F, V, np.flatnonzero(np.isin(SPK, fit)), SPK, seed=seed, aux=aux,
                            on_snap=(lambda e, mm: score(mm, (e,))) if KW.get("snaps") else None, **KW)
        score(m)
    return branch, seed, out


if __name__ == "__main__":
    tag, extra = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from t3lib import T3, evaluate
    path = T3 / f"outputs/oof_j_{tag}_{os.environ.get('T3_OUTSUFFIX', 'other')}.pkl"
    res = pickle.load(open(path, "rb")) if path.exists() else {}
    todo = [(b, extra, s) for b in BRANCHES if b.replace(":", "_") not in res for s in SEEDS]
    per = {}; snaps = {}; t0 = time.time()
    print(f"{tag}: {len(todo)} jobs on {workers} workers, sources +{extra}, KW {kw_from_env()}", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for branch, seed, out in pool.imap_unordered(job, todo):
            per.setdefault(branch, {})[seed] = out
            print(f"  [{(time.time() - t0) / 60:5.1f}m] {branch} seed {seed}", flush=True)
            if len(per[branch]) == len(SEEDS):
                name = branch.replace(":", "_")
                mean = {k: np.mean([per[branch][s][k] for s in SEEDS], 0) for k in per[branch][SEEDS[0]]}
                res[name] = {k: v for k, v in mean.items() if len(k) == 2}
                for k, v in mean.items():  # snapshot epochs (T3_SNAPS) -> one OOF pickle per epoch, same layout as the main one
                    if len(k) == 3: snaps.setdefault(k[2], {}).setdefault(name, {})[k[:2]] = v
                pickle.dump(res, open(path, "wb"))
                for e, r in snaps.items(): pickle.dump(r, open(str(path).replace(f"oof_j_{tag}_", f"oof_j_{tag}_e{e}_"), "wb"))
                for e in sorted(snaps):
                    print(f"{tag} {name:14s} @e{e:<3d}", end=" ", flush=True); evaluate(lambda f, p, fr, v: snaps[e][name][f, p], verbose=True)
                print(f"{tag} {name:14s}", end=" ", flush=True); evaluate(lambda f, p, fr, v: res[name][f, p], verbose=True)
    print(f"{tag}: done in {(time.time() - t0) / 60:.1f} min", flush=True)
