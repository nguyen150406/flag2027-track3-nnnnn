"""Package fetched VoxBlink2 media for the Kaggle encode kernel: caches/vb2/kaggle_ds/{vb2_media.zip, vb2_manifest.csv}."""
import pickle, zipfile, json
from pathlib import Path
import pandas as pd
D = Path(__file__).resolve().parents[1] / "caches/vb2"; M = D / "media"; OUT = D / "kaggle_ds"; OUT.mkdir(exist_ok=True)
f = pd.read_csv(D / "fetch_manifest.csv", dtype=str); f = f[f.ok == "1"].drop_duplicates(["identity", "video"])
lang = pickle.load(open(D.parent / "voxblink_official/vb2_spk_lang_whisper.pkl", "rb"))
excl = set(pd.read_csv(D / "exclude_bengali.csv").identity); assert not (set(f.identity) & excl), "Bengali-flagged speaker in media"
man = pd.DataFrame(dict(identity="vb2_" + f.identity, gender="u", language=[lang[s].most_common(1)[0][0] if s in lang and lang[s] else "?" for s in f.identity],
                        video=f.video, utt=f.utt, key=f.identity + "_" + f.video, n_frames=f.n_frames.astype(int), ok=True, dur=f.dur))
man = man[[(M / "wav" / f"{k}.wav").exists() for k in man.key]].reset_index(drop=True)
man.to_csv(OUT / "vb2_manifest.csv", index=False)
with zipfile.ZipFile(OUT / "vb2_media.zip", "w", zipfile.ZIP_STORED) as z:
    z.write(OUT / "vb2_manifest.csv", "vb2_manifest.csv")
    for r in man.itertuples(index=False):
        z.write(M / "wav" / f"{r.key}.wav", f"wav/{r.key}.wav")
        for k in (1, 2):
            p = M / "img" / f"{r.key}_{k}.jpg"
            if p.exists(): z.write(p, f"img/{r.key}_{k}.jpg")
json.dump({"title": "FLAG T3 VB2 Media", "id": "KAGGLE_USER/flag-t3-vb2-media", "licenses": [{"name": "CC-BY-NC-SA-4.0"}]}, open(OUT / "dataset-metadata.json", "w"))
print(f"packed {len(man)} clips from {man.identity.nunique()} speakers; languages {man.drop_duplicates('identity').language.value_counts().head(8).to_dict()}")
