"""MAV-Celeb v3 (EN+DE train) voices with all Track 3 voice encoders (local CPU)."""
import glob, os, sys, numpy as np, pandas as pd
os.environ.setdefault("T3_DEVICE", "cpu")
import t3extract as X
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "external_mavceleb/v3_full"
OUT = ROOT / "Track 3/caches/mav_v3_voices"; OUT.mkdir(parents=True, exist_ok=True)
paths = sorted(glob.glob(str(V3 / "voices/*/*/*/*.wav")))
meta = pd.DataFrame(dict(path=paths)); p = meta.path.str.split("/")
meta["identity"], meta["language"], meta["video"] = p.str[-4], p.str[-3], p.str[-2]
meta.to_csv(OUT / "metadata.csv", index=False)
for enc in sys.argv[1:]:
    if not (OUT / f"{enc}.float32.npy").exists():
        np.save(OUT / f"{enc}.float32.npy", X.voice_encoder(enc)(paths, log=print))
