"""Own frozen ImageBind stream "ibo" for the R2 system (12/11; inline code of that day saved as a script on 13/11 for the release).
FLAG-train ImageBind-huge face/voice embeddings from the Track 2 XP65 extraction (same extractor as scripts/t3_imagebind_extract.py),
re-ordered to META train row order; score = unit(face - mu_f) . unit(voice - mu_v), means from each fold's FIT persons only.
  python t3_ib_own.py [out_dir (default outputs/ultra)]  -> ib_own_flag_face.npy, ib_own_flag_voice.npy, ib_own.npz (t3_ultra_eval trial keys)"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import t3_ultra_eval as E
from t3lib import META, SPK


def main(out=None):
    out = Path(out) if out else E.O; out.mkdir(parents=True, exist_ok=True)
    X = E.T3.parent / "Track 2/outputs/t2_xp65_imagebind_train/extraction"
    ids = open(X / "pair_ids.txt").read().split(); pos = {s: i for i, s in enumerate(ids)}
    tr = META[META.partition.eq("train")].sort_values("row_id"); order = np.array([pos[s] for s in tr.pair_id.astype(str)])
    Fi = np.load(X / "face_embeddings.float32.npy")[order]; Vi = np.load(X / "voice_embeddings.float32.npy")[order]
    np.save(out / "ib_own_flag_face.npy", Fi); np.save(out / "ib_own_flag_voice.npy", Vi)
    u = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True); T = E.load_trials(); SP = E.SP; res = {}
    for k in range(5):
        fit = np.isin(SPK, [p for j, f in enumerate(SP["folds"]) if j != k for p in f]); mf, mv = Fi[fit].mean(0), Vi[fit].mean(0)
        for pr in E.PR:
            t = T[f"f{k}_{pr}"]; res[f"f{k}_{pr}"] = (u(Fi[t["fi"]] - mf) * u(Vi[t["vi"]] - mv)).sum(1)
            t = T[f"res_{pr}"]; res.setdefault(f"res_{pr}", []).append((u(Fi[t["fi"]] - mf) * u(Vi[t["vi"]] - mv)).sum(1))
    for pr in E.PR: res[f"res_{pr}"] = np.mean(res[f"res_{pr}"], 0)
    np.savez(out / "ib_own.npz", **res)
    for pr in E.PR:
        y = np.concatenate([T[f"f{k}_{pr}"]["y"] for k in range(5)])
        print(pr, "ib_own CV %.2f res %.2f" % (E.eer(y, np.concatenate([res[f"f{k}_{pr}"] for k in range(5)])), E.eer(T[f"res_{pr}"]["y"], res[f"res_{pr}"])))


if __name__ == "__main__":
    main(*sys.argv[1:])
