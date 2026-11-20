"""Two-branch ticket: English files = XJ-family model dir (own re-implementation of the teammate's XJ_ens5, external data Bengali-filtered),
Bangla files = sub90c (bit-exact). usage: python build_ticket.py <xj model dir> <name>   or   <ng model dir>,<g model dir> <name>"""
import sys, json, zipfile, hashlib, shutil
import numpy as np, pandas as pd
from pathlib import Path
import xj_lib as X
T3 = Path(__file__).resolve().parents[1]; SUB = T3 / "submissions"
srcs = [Path(x) for x in sys.argv[1].split(",")]; src_of = {"no_gender": srcs[0], "gender": srcs[-1]}; src = srcs[0]
name = sys.argv[2]; out = SUB / f"t3_{name}"; assert not out.exists(), out; out.mkdir()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest(); files = {}
for tr in ("no_gender", "gender"):
    for lang, tag in (("English", "heard"), ("Bangla", "unheard")):
        n = f"{tr}/sub_score_v4_{lang}_{tag}.txt"; (out / tr).mkdir(exist_ok=True); L, _, _ = X.load_eval(tr, lang)
        if lang == "Bangla": shutil.copyfile(SUB / "t3_sub90c_eval" / n, out / n); assert sha(out / n) == sha(SUB / "t3_sub90c_eval" / n); how = "sub90c bit-exact"
        else: shutil.copyfile(src_of[tr] / n, out / n); how = f"XJ-family ({src_of[tr].name})"
        d = pd.read_csv(out / n, sep=" ", header=None, names=["id", "s"], dtype={"id": str})
        assert len(d) == len(L) and (d.id.values == L.pid.values).all() and d.id.is_unique and np.isfinite(d.s).all(), n
        files[n] = dict(rows=len(d), sha256=sha(out / n), source=how); print(f"  {n}: {how}, {len(d)} rows")
zp = out / f"{name}.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for n in files: z.write(out / n, n)
cfgs = {tr: (json.load(open(d / "config.json")) if (d / "config.json").exists() else None) for tr, d in src_of.items()}; cfg = cfgs["no_gender"]
assert all(c and c.get("langs") == cfg.get("langs") and c.get("flag") == cfg.get("flag") and c.get("mav_lid") == cfg.get("mav_lid") and c.get("bengali")
           for c in cfgs.values()), "same Bengali filters in both protocols"
code = {q: sha(Path(__file__).parent / q) for q in ("xj_lib.py", "xj_ext.py", "final_xj.py", "build_ticket.py", "make_ens_dir.py", "vox2_faces_btc.py", "xj_val.py", "mavx_check.py")}
code.update({f"kaggle/{q}": sha(T3 / "kaggle" / q) for q in ("xjx_vb/xjx_vb.py", "xjx_vox2a/xjx_vox2a.py", "xjx_lid/xjx_lid.py", "xjx_lid2/xjx_lid2.py", "xjx_mavx0/xjx_mavx.py", "xjx_mavx1/xjx_mavx.py", "xjx_mavxds/t3_lid_audit.py") if (T3 / "kaggle" / q).exists()})
XD = Path(__file__).parent
data_files = [XD / "ext_feats/faces.npy", XD / "ext_feats/voices.npy", XD / "ext_feats/faces_meta.csv", XD / "ext_feats/voices_meta.csv",
              XD / "vb_btc/vb_btc_face.f16.npy", XD / "vb_btc/vb_btc_voice.f32.npy", XD / "vb_btc/vb_btc_face_rows.csv", XD / "vb_btc/vb_btc_voice_rows.csv",
              XD / "vox2_btc/vox2_btc_face.f16.npy", XD / "vox2_btc/vox2_btc_voice.f32.npy", XD / "vox2_btc/vox2_btc_face_rows.csv", XD / "vox2_btc/vox2_btc_voice_rows.csv",
              XD / "lid_xjx_urdu_hindi.csv", XD / "lid_xjx_mav_english.csv", XD / "vb_leak_drop.json", XD / "vox2_leak_drop.json",
              T3 / "kaggle/enx2r_data/meta/exclude.csv", T3 / "outputs/enx2r/v2_lid_exclude.json", T3 / "caches/ext/exclude_bengali_v3.csv",
              T3.parent / "external_mavceleb/mavx_dl/lid_mavx.csv"] + (sorted((XD / "mavx").glob("mavx*")) if any(c.get("mavx") or any(m.get("mavx") for m in c.get("members", {}).values()) for c in cfgs.values())
                                                                    else sorted((XD / "mavx").glob("mavx*_lid.csv")))   # MAVX LID results feed the identity filter even when MAVX clips are not used
data_files += [T3 / f"outputs/lid_v3/{w}/lid/lid_{i}of2.csv" for w in ("vox", "web") for i in range(2)] + [T3 / "kaggle/xjx_lidlist/xjx_lid_clips.csv",   # review 14/11:
               X.DATA / "train_set/train_set/train_English.txt", X.DATA / "train_set/features/faces/train_English_faces.csv",           # FLAG mask, VB/VoxCeleb1/2
               X.DATA / "train_set/features/voices/train_English_voices.csv", X.DATA / "meta_file_train_set.csv"]                       # filters, Urdu/Hindi map, FLAG train
missing = [str(f) for f in data_files if not f.exists()]; assert not missing, f"provenance inputs missing (fail closed): {missing}"
data_sha = {str(f.relative_to(T3.parent)): sha(f) for f in data_files}
filt = ("external MAV: identity dropped if any LID-checked clip is flagged (LID v3: any model P(bn)+P(as) > .5 in any 30-s window; v1 lid_mavx.csv, "
        "v2 v2_lid_exclude.json, 14/11 clip-level runs lid_xjx_urdu_hindi.csv + lid_xjx_mav_english.csv + mavx0/1_lid.csv when present; the MAVX run checked 4,761 extra MAV clips of the kept ids and its flags drop those identities even when the extra clips are not used) or it has no LID coverage; "
        "VoxCeleb1: India-nationality ids dropped, all kept ids LID v3-audited at identity level (subset of clips), 0 flags"
        + (("; MAV Hindi/Urdu clips kept only if cleared by LID v3 (3 models, every 30-s window); a flagged clip drops its whole identity" if cfg.get("mav_lid")
            else "; MAV Hindi/Urdu clips excluded (never LID-checked)") if cfg and cfg.get("langs") == "english,unk" else "; MAV Hindi/Urdu clips of kept ids included, NOT LID-checked")
        + (f"; FLAG train: clips flagged by LID ({cfg['flag']}) dropped" if cfg and cfg.get("flag", "none") != "none" else "; FLAG train: all English clips used"))
def desc(c):
    if "members" in c: return "mean centred cosine of [" + "; ".join(f"{n}: {desc(m)}" for n, m in c["members"].items()) + "]"
    ext = ["MAV-Celeb v1/v2", "VoxCeleb1"] + (["VoxCeleb2"] if c.get("vox2") else []) + (["VoxBlink"] if c.get("vb") else []) + (["extra MAV-Celeb clips (MAVX, LID-cleared)"] if c.get("mavx") else [])
    return f"{len(c.get('seeds', [0]*5))} seeds, towers {c.get('hidden', 256)}->{c.get('outd', 128)}, {c.get('epochs', 3)} epochs, ext ratio {c.get('ratio', 4)}, ext = {' + '.join(ext)}"
json.dump(dict(system=f"two-branch: English = re-implementation of teammate XJ_ens5 (BTC VGGFace fc7 + ECAPA features, 2 MLP towers, symmetric InfoNCE, joint FLAG train + external, "
               "FLAG-train centring) -- " + " | ".join(f"{t}: {d.name} = {desc(cfgs[t])}" for t, d in src_of.items()) + "; Bangla = sub90c bit-exact",
               bengali_filter=filt + ("; VoxBlink: Bengali speakers excluded at prep, ids kept only if LID v3 covered them and no clip flagged, minus ENX2 exclusion list" if any(c.get("vb") for c in cfgs.values()) else "")
               + ("; VoxCeleb2: ids kept only if LID v3 covered them and no clip flagged, minus ENX2 exclusion list" if any(c.get("vox2") for c in cfgs.values()) else "")
               + ("; MAVX (extra MAV-Celeb clips of kept ids): every selected clip LID v3-checked (3 models, every 30-s window), only cleared clips used, a flagged clip drops its whole identity" if any(c.get("mavx") for c in cfgs.values()) else "")
,
               model_config=cfgs, code_sha256={**code, "btc_feat.py": sha(Path(__file__).parent / "btc_feat.py")}, input_data_sha256=data_sha, files=files, zip_sha256=sha(zp), pair_local=True, test_statistics_used=False,
               checkpoints={f"{t}/{p.name}": sha(p) for t, d in src_of.items() for p in sorted(d.glob("*xj_m*.pt"))}), open(out / "provenance.json", "w"), indent=1)
print("wrote", zp)
