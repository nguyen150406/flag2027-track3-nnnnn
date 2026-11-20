"""self-check: our torch VGGFace port vs the organizer BTC face CSV on FLAG train rows (target cos >= .9999)."""
import sys, numpy as np, torch, xj_lib as X, btc_feat as B
torch.set_num_threads(6); L, F, _ = X.load_flag(); m = B.VGGFace.from_h5("models/vgg_face_weights.h5")
rng = np.random.default_rng(0); ix = rng.choice(len(L), int(sys.argv[1]) if len(sys.argv) > 1 else 48, replace=False)
x = torch.from_numpy(np.stack([B.load_img(X.DATA / "train_set/train_set" / L.face.iat[i]) for i in ix]))
with torch.no_grad(): e = torch.cat([m(x[j:j + 8]) for j in range(0, len(x), 8)]).numpy()
b = F[ix]; cos = (e * b).sum(1) / np.linalg.norm(e, axis=1) / np.linalg.norm(b, axis=1)
print(f"face cos min {cos.min():.7f} mean {cos.mean():.7f} | norm ours {np.linalg.norm(e,axis=1).mean():.3f} btc {np.linalg.norm(b,axis=1).mean():.3f} | "
      f"max abs diff {np.abs(e-b).max():.4g} | nonzero ours {(e>0).mean():.3f} btc {(b>0).mean():.3f}")
