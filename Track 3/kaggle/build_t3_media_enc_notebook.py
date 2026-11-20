"""Generic media encode kernel (usage: python build_t3_media_enc_notebook.py <name>, e.g. lrs3, vb2). Originally the LRS3 encode kernel (team account GPU): mounts the private dataset KAGGLE_USER/flag-t3-lrs3-media (built locally by
scripts/prep_lrs3.py) and runs all six frozen Track 3 encoders -> lrs3_* arrays + rows_lrs3.csv (VoxCeleb2 batch format)."""
import json, sys
from pathlib import Path
NAME = sys.argv[1] if len(sys.argv) > 1 else "lrs3"

HERE = Path(__file__).resolve().parent
B = json.load(open(HERE / "t3_vox2b/t3_vox2b.ipynb"))
setup = [c["source"] for c in B["cells"] if c["cell_type"] == "code"][0]
setup = setup.replace('W = Path("/kaggle/temp/vox2")', f'W = Path("/kaggle/temp/{NAME}")').replace('OUT = Path("/kaggle/working/vox2_out")', f'OUT = Path("/kaggle/working/{NAME}_out")')
assert f"/kaggle/temp/{NAME}" in setup and f"{NAME}_out" in setup

body = (r'''
# ---------------------------------------------------------------- media (zip or auto-extracted), rows (one per face frame), encoders
SRC = find("{NAME}_manifest.csv")[0].parent
if (SRC / "{NAME}_media.zip").exists():
    with zipfile.ZipFile(SRC / "{NAME}_media.zip") as z: z.extractall(W)
    MED = W
else:
    MED = [d for d in [SRC, SRC / "{NAME}_media"] if (d / "wav").exists()][0]
man = pd.read_csv(SRC / "{NAME}_manifest.csv", dtype={"video": str, "utt": str, "key": str})
man = man[man.ok].reset_index(drop=True)
rows = []
for i, r in man.iterrows():
    for k in range(1, 3):
        fp = MED / "img" / f"{r.key}_{k}.jpg"
        if fp.exists():
            rows.append(dict(identity=r.identity, gender=r.gender, language=r.language, video=r.video, clip_index=int(i),
                             voice=f"{r.identity}/{r.video}/{r.utt}.wav", face=f"{r.identity}/{r.video}/{r.utt}_{k}.jpg", face_path=str(fp)))
rows = pd.DataFrame(rows); clips = np.array(sorted(rows.clip_index.unique()))
cpos = {c: j for j, c in enumerate(clips)}; ridx = rows.clip_index.map(cpos).values
log(f"rows {len(rows)} clips {len(clips)} identities {rows.identity.nunique()} media {MED}")
import torch
def save(tag, arr):
    np.save(OUT / f"{NAME}_{tag}.float32.npy", arr.astype(np.float32)); np.save(OUT / f"{NAME}_{tag}.ok.npy", np.ones(len(arr), bool))
    log(f"saved {tag} {arr.shape}")
for enc in ("farl", "clip"):
    save(f"face_{enc}", X.face_encoder(enc)(list(rows.face_path), log=log)); torch.cuda.empty_cache()
wavs = [str(MED / "wav" / f"{man.key[c]}.wav") for c in clips]
for enc in ("simam", "sam100ft", "ecapa", "r293"):
    e = X.voice_encoder(enc)(wavs, log=log); save(f"voice_{enc}", e[ridx]); torch.cuda.empty_cache()
rows.drop(columns=["face_path"]).to_csv(OUT / "rows_{NAME}.csv", index=False)
with zipfile.ZipFile("/kaggle/working/t3_{NAME}.zip", "w", zipfile.ZIP_STORED) as z:
    for p in sorted(OUT.iterdir()): z.write(p, p.name)
shutil.rmtree(OUT); log("done")
''').replace("{NAME}", NAME)
cells = [{"cell_type": "markdown", "metadata": {}, "source": f"# FLAG Track 3 — {NAME} encode"}]
cells += [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s} for s in (setup, body)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
d = HERE / f"t3_{NAME}enc_team"; d.mkdir(exist_ok=True)
json.dump(nb, open(d / f"t3_{NAME}enc_team.ipynb", "w"), indent=1)
json.dump({"id": f"KAGGLE_USER/flag-t3-{NAME}-enc", "title": f"FLAG T3 {NAME.upper()} Enc", "code_file": f"t3_{NAME}enc_team.ipynb", "language": "python",
           "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
           "dataset_sources": ["KAGGLE_USER/flag-t3-tools", f"KAGGLE_USER/flag-t3-{NAME}-media"], "competition_sources": [], "kernel_sources": []},
          open(d / "kernel-metadata.json", "w"), indent=1)
print("built", d)
