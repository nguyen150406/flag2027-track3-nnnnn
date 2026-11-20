"""90c-X (15/11, user "oke chạy ngay đi"): MAV-Celeb v1+v2 as an extra external source "mav" for the no-FLAG CLIP LoRA (t3_ft_clip.py).
Same MAV clips as XJ-U's pool (teammate xjc set: 9,520 faces / 8,779 voices, English + Urdu + Hindi), CLIP-level Bengali rule
(xj_ext.clip_lid: a clip any LID v3 model flags, or an unchecked/unreadable clip, is dropped; identities kept). Identities are
'v1_<id>' / 'v2_<id>' (MAV v1 and v2 reuse the names id0001... for DIFFERENT people). Voices = cat4 parts (simam, sam100ft, r293, ecapa)
from the XJ-1 Kaggle extraction (ncamtu06/flag-t3-xj1-mav, sub90c recipes, parity-gated vs our local v1 caches) -> the same voice
space as every other external source of the LoRA. Same-identity voices in another language are positives in the LoRA's SupCon /
prototype losses (cross-language supervision).
Layout for t3data.ext_source (one table, independent face / voice selection through the ok masks):
  rows 0..nv-1          voice rows (voice = clip, face = a placeholder face of the same identity, face ok False)
  rows nv..nv+nf-1      face rows  (face = image rel path under the MAV mount, voice placeholder, voice ok False)
The dataset stores only the nv voice rows of each voice array; the kernel pads the face rows with zeros (ok False) when installing.
usage: python build_mavsrc.py <user> [<user> ...] -> kaggle/mavsrc_<user>/ (private dataset <user>/flag-t3-mavsrc)"""
import sys, json, hashlib, shutil
from pathlib import Path
import numpy as np, pandas as pd
K = Path(__file__).resolve().parent; T3 = K.parent; sys.path.insert(0, str(T3 / "xj"))
import xj_ext as XE
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
XO = T3 / "xj/xj1"; ENC = ("simam", "sam100ft", "r293", "ecapa")

R = pd.read_csv(T3 / "kaggle/xjc_rows/xjc_rows.csv"); vr, fr = R[R.kind.eq("voice")].reset_index(drop=True), R[R.kind.eq("face")].reset_index(drop=True)
assert (pd.read_csv(XO / "xj1_voice_rows.csv")[["src", "rel"]].values == vr[["src", "rel"]].values).all(), "XJ-1 voice rows == xjc voice rows"
assert (pd.read_csv(XO / "xj1_face_rows.csv")[["src", "rel"]].values == fr[["src", "rel"]].values).all(), "XJ-1 face rows == xjc face rows"
V = {e: np.load(XO / f"xj1_voice_{e}.float32.npy") for e in ENC}; vok = np.logical_and.reduce([np.load(XO / f"xj1_voice_{e}.ok.npy") for e in ENC])
fok = np.logical_and.reduce([np.load(XO / f"xj1_face_{e}.ok.npy") for e in ("clipnf", "clipnf2", "farl")])
assert all(len(v) == len(vr) and np.isfinite(v).all() for v in V.values()) and len(fok) == len(fr)
assert not vr.lang.astype(str).str.lower().isin(["bengali", "bangla", "bn"]).any() and not fr.lang.astype(str).str.lower().isin(["bengali", "bangla", "bn"]).any()
chk, fl = XE.clip_lid(); lk = vr.src + ":" + vr.rel.str.split("/voices/").str[1]
lid_ok = np.array([k in chk and k not in fl for k in lk]); vkeep = vok & lid_ok
ident_v, ident_f = (vr.src + "_" + vr.id).to_numpy(), (fr.src + "_" + fr.id).to_numpy()
lang = lambda s: s.astype(str).str.capitalize()
first_face = dict(zip(ident_f[::-1], fr.rel.values[::-1])); first_voice = dict(zip(ident_v[::-1], vr.rel.values[::-1]))
rows_v = pd.DataFrame(dict(identity=ident_v, language=lang(vr.lang), video=vr.video, voice=vr.rel, face=[first_face.get(i, "none/none/none.jpg") for i in ident_v]))
rows_f = pd.DataFrame(dict(identity=ident_f, language=lang(fr.lang), video=fr.video, voice=[first_voice.get(i, "none/none/none.wav") for i in ident_f], face=fr.rel))
rows = pd.concat([rows_v, rows_f], ignore_index=True); nv, nf = len(rows_v), len(rows_f)
ids_both = sorted(set(ident_v[vkeep]) & set(ident_f[fok]))
summ = dict(voices_total=int(nv), voices_kept=int(vkeep.sum()), voices_lid_dropped=int((vok & ~lid_ok).sum()), voices_unusable=int((~vok).sum()),
            faces_total=int(nf), faces_ok=int(fok.sum()), identities_with_face_and_voice=len(ids_both),
            identities_by_version={v: sum(i.startswith(v + "_") for i in ids_both) for v in ("v1", "v2")},
            kept_voice_languages=rows_v.language[vkeep].value_counts().to_dict(),
            ids_with_2plus_voice_languages=int(rows_v[vkeep].groupby("identity").language.nunique().ge(2).sum()))
print(json.dumps(summ))
for user in sys.argv[1:]:
    d = K / f"mavsrc_{user}"; d.mkdir(exist_ok=True)
    rows.to_csv(d / "rows_mav.csv", index=False)
    for e in ENC:
        np.save(d / f"mavv_{e}.float32.npy", V[e].astype(np.float32))
    np.save(d / "mavv_ok.npy", vkeep); np.save(d / "mavf_ok.npy", fok)
    json.dump(dict(summary=summ, nv=nv, nf=nf, source_rows=str(T3 / "kaggle/xjc_rows/xjc_rows.csv"), xj1_provenance=json.load(open(XO / "PROVENANCE.json")) if (XO / "PROVENANCE.json").exists() else None,
                   code_sha256={"build_mavsrc.py": sha(__file__), "xj/xj_ext.py": sha(T3 / "xj/xj_ext.py")}),
              open(d / "MAVSRC_PROVENANCE.json", "w"), indent=1)
    json.dump({"title": "flag-t3-mavsrc", "id": f"{user}/flag-t3-mavsrc", "licenses": [{"name": "other"}]}, open(d / "dataset-metadata.json", "w"), indent=1)
    print("built", d, sorted(p.name for p in d.iterdir()))
