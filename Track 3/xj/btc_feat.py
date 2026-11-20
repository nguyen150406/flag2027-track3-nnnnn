"""BTC (organizer baseline) feature extractors, re-implemented from the public starter kit
(github.com/SwapnilKhandoker101/FLAG_2027, feature_extraction/extract_face_features.py, extract_voice_features_ecapa.py):
 face  = VGGFace (deepface vgg_face_weights.h5) fc7 after ReLU, 4096-d; input = PIL RGB resize 224x224 (PIL default filter, bicubic; same call as the starter kit), RGB->BGR,
         minus channel means (93.5940, 104.7624, 129.1863), no /255.  Ported to torch (Keras conv kernels HWIO -> OIHW, NHWC -> NCHW).
 voice = speechbrain ECAPA (yangwang825/ecapa-tdnn-vox2): librosa.load(sr=16000, mono=True), centre-crop to max_seconds (teammate: 30),
         compute_features -> mean_var_norm(feats, rel_lens) -> embedding_model, normalize=False, 192-d.  Batch size 1 (no padding effects)."""
import numpy as np, torch, torch.nn as nn
from PIL import Image
MEANS = np.array([93.5940, 104.7624, 129.1863], np.float32)   # applied to B, G, R channels after the RGB->BGR flip


def load_img(path):
    a = np.asarray(Image.open(path).convert("RGB").resize((224, 224)), dtype=np.float32)[..., ::-1].copy(); a -= MEANS; return a


class VGGFace(nn.Module):
    CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]

    def __init__(self):
        super().__init__(); L, c = [], 3
        for v in self.CFG:
            if v == "M": L.append(nn.MaxPool2d(2, 2))
            else: L += [nn.Conv2d(c, v, 3, padding=1), nn.ReLU(inplace=True)]; c = v
        self.features = nn.Sequential(*L); self.fc6 = nn.Conv2d(512, 4096, 7); self.fc7 = nn.Conv2d(4096, 4096, 1)

    def forward(self, x_nhwc):   # x: (B, 224, 224, 3) BGR mean-subtracted
        x = self.features(x_nhwc.permute(0, 3, 1, 2)); x = torch.relu(self.fc6(x)); x = torch.relu(self.fc7(x)); return x.flatten(1)

    @classmethod
    def from_h5(cls, path):
        import h5py
        m = cls(); convs = [l for l in m.features if isinstance(l, nn.Conv2d)] + [m.fc6, m.fc7]
        with h5py.File(path, "r") as f:
            g = f["model_weights"] if "model_weights" in f else f
            names = [n.decode() if isinstance(n, bytes) else n for n in g.attrs["layer_names"]]
            kw = []
            for n in names:
                wn = [w.decode() if isinstance(w, bytes) else w for w in g[n].attrs["weight_names"]]
                if wn: kw.append((n, [np.asarray(g[n][w]) for w in wn]))
        assert len(kw) == len(convs) + 1, (len(kw), len(convs))   # + the final 2622 classifier conv (unused)
        for (n, (W, b)), conv in zip(kw, convs):
            W = torch.from_numpy(W).permute(3, 2, 0, 1).contiguous(); assert W.shape == conv.weight.shape, (n, W.shape, conv.weight.shape)
            conv.weight.data.copy_(W); conv.bias.data.copy_(torch.from_numpy(b))
        return m.eval()


def ecapa(savedir, device="cpu"):
    from speechbrain.inference.speaker import EncoderClassifier
    import sys
    enc = EncoderClassifier.from_hparams(source="yangwang825/ecapa-tdnn-vox2", savedir=savedir, run_opts={"device": device})
    from speechbrain.utils.importutils import LazyModule
    for k in [k for k, v in list(sys.modules.items()) if isinstance(v, LazyModule)]: del sys.modules[k]   # inspect.getmodule() would trigger k2 etc.
    return enc


def load_wav(path, max_seconds=30.0):
    import librosa
    y, _ = librosa.load(path, sr=16000, mono=True)
    if max_seconds and y.size > int(max_seconds * 16000):
        k = int(max_seconds * 16000); s = (y.size - k) // 2; y = y[s:s + k]
    return y.astype(np.float32)


@torch.no_grad()
def voice_emb(enc, y):
    w = torch.from_numpy(y)[None]; lens = torch.ones(1)
    f = enc.mods.compute_features(w); f = enc.mods.mean_var_norm(f, lens); return enc.mods.embedding_model(f, lens).reshape(-1).numpy()
