"""Bengali-speaker exclusion list for the external VoxCeleb sources (FLAG 2027 FAQ: Bengali speech must not train the audio side).
VoxCeleb2: Kaggle kernel trncnguyn/flag-t3-voxceleb-language-filter (speechbrain VoxLingua107 ECAPA on every VoxCeleb2 clip we
train on) -> kaggle/outlid/lid_vox2.csv; a PERSON is dropped if any of their clips has Bengali/Assamese as top language or
P(bn) >= 0.2. VoxCeleb1 (no audio LID: its datasets hung the kernel at start-up): every person
whose nationality is India or Bangladesh is dropped (conservative; 26 of 1,251, none Bangladeshi). VoxBlink is already filtered by the official spk2lan list, AVSpeech by Whisper LID.
Output: caches/ext/exclude_bengali.csv (src, identity, reason), read by t3data.ext_source.  Usage: python t3_lid_filter.py"""
import zipfile
import pandas as pd
from t3lib import T3

L = T3 / "kaggle/outlid"
if (L / "t3_lid.zip").exists():
    zipfile.ZipFile(L / "t3_lid.zip").extractall(L)
v2 = pd.read_csv(L / "lid_vox2.csv")
v1 = pd.read_csv(T3 / "caches/ext/rows_vox.csv").drop_duplicates("identity"); v1 = v1[v1.nationality.isin(["India", "Bangladesh"])]
rows = [dict(src="vox", identity=r.identity, reason=f"VoxCeleb1 nationality {r.nationality} (no audio LID)") for r in v1.itertuples()]
print(f"vox: excluded {len(v1)} people by nationality {v1.nationality.value_counts().to_dict()}")
for src, d in (("vox2", v2),):
    d = d[d.top.fillna("").ne("")]  # clips whose audio could not be read carry no language
    lang = d.top.str.split(":").str[0].str.strip()
    hit = d[lang.isin(["bn", "as"]) | d.p_bn.ge(0.2)]
    for ident, g in hit.groupby("identity"):
        rows.append(dict(src=src, identity=ident, reason=f"{len(g)}/{(d.identity == ident).sum()} clips; top {sorted(set(lang[g.index]))}; max p_bn {g.p_bn.max():.2f}"))
    print(f"{src}: {d.identity.nunique()} people / {len(d)} clips with LID; top languages {lang.value_counts().head(6).to_dict()}; "
          f"excluded {hit.identity.nunique()} people ({len(hit)} clips)")
out = pd.DataFrame(rows, columns=["src", "identity", "reason"]); out.to_csv(T3 / "caches/ext/exclude_bengali.csv", index=False)
print("wrote caches/ext/exclude_bengali.csv", len(out), "people"); print(out.head(10).to_string())
