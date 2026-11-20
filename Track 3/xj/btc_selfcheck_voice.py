"""self-check: speechbrain ECAPA (yangwang825/ecapa-tdnn-vox2), batch 1, vs the organizer BTC voice CSV on FLAG train rows. Run in ~/.venvs/xjsb."""
import sys, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); import btc_feat as B
torch.set_num_threads(6); DATA = Path(__file__).resolve().parents[2] / "data"
L = pd.read_csv(DATA / "train_set/train_set/train_English.txt", sep=" ", header=None, names=["pid", "y", "voice", "face", "id", "age"])
V = pd.read_csv(DATA / "train_set/features/voices/train_English_voices.csv", header=None).to_numpy(np.float32)[:, :-1]
enc = B.ecapa(str(Path.home() / ".cache/xj_ecapa")); n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
ix = np.random.default_rng(0).choice(len(L), n, replace=False); out = {}
for ms in (30.0, 0.0):
    e = np.stack([B.voice_emb(enc, B.load_wav(DATA / "train_set/train_set" / L.voice.iat[i], ms)) for i in ix]); b = V[ix]
    cos = (e * b).sum(1) / np.linalg.norm(e, axis=1) / np.linalg.norm(b, axis=1)
    print(f"max_seconds {ms}: voice cos min {cos.min():.5f} mean {cos.mean():.6f} | norm ours {np.linalg.norm(e,axis=1).mean():.3f} btc {np.linalg.norm(b,axis=1).mean():.3f}", flush=True)
