"""Frozen audeering wav2vec2 age/gender models; every file processed alone (no padding).
Usage: python extract_voice_age.py <repo> <list.csv with column path> <out.npz> [seg seconds]
Writes age (0..1) and P(female, male, child) per file.
seg S (user 2026-11-04, research direction A1; FAQ 20/10 allows fixed sub-segments encoded independently and averaged): every file is cut
into fixed non-overlapping S-second windows (a remainder shorter than S/2 joins the previous window; files shorter than S stay whole),
each window is encoded alone and age / gender probabilities / hidden state are averaged. Default (no S): whole file, <=60 s chunks."""
import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.io import wavfile
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel


class Head(nn.Module):
    def __init__(self, c, n):
        super().__init__(); self.dense = nn.Linear(c.hidden_size, c.hidden_size)
        self.dropout = nn.Dropout(c.final_dropout); self.out_proj = nn.Linear(c.hidden_size, n)
    def forward(self, x):
        return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))


class AgeGender(Wav2Vec2PreTrainedModel):
    def __init__(self, c):
        super().__init__(c); self.wav2vec2 = Wav2Vec2Model(c); self.age = Head(c, 1); self.gender = Head(c, 3); self.init_weights()
    def forward(self, x):
        h = self.wav2vec2(x).last_hidden_state.mean(1)
        return self.age(h), torch.softmax(self.gender(h), 1), h


def read(p):
    sr, x = wavfile.read(p); assert sr == 16000, p
    x = x.mean(1) if x.ndim == 2 else x
    return (x.astype(np.float32) / 32768.0) if x.dtype == np.int16 else x.astype(np.float32)


if __name__ == "__main__":
    repo, lst, out = sys.argv[1:4]; SEG = float(sys.argv[4]) if len(sys.argv) > 4 else 0
    dev = os.environ.get("T3_DEVICE") or ("mps" if torch.backends.mps.is_available() else "cpu")  # 12/11: MPS hung twice on MAV clips -> CPU via T3_DEVICE
    proc = Wav2Vec2Processor.from_pretrained(repo); model = AgeGender.from_pretrained(repo).to(dev).eval()
    paths = pd.read_csv(lst).path.tolist(); ages, gen, emb = [], [], []
    with torch.inference_mode():
        for i, p in enumerate(paths):
            w = read(p); L = 60 * 16000  # files longer than 60 s: average over balanced <=60 s chunks
            k = max(1, int(np.ceil(len(w) / L))); bounds = np.linspace(0, len(w), k + 1).astype(int)
            if SEG:
                S = int(SEG * 16000); bounds = list(range(0, len(w), S)) + [len(w)]
                if len(bounds) > 2 and bounds[-1] - bounds[-2] < S // 2: bounds.pop(-2)
                bounds = np.array(bounds)
            outs = []
            for s0, s1 in zip(bounds[:-1], bounds[1:]):
                x = proc(w[s0:s1], sampling_rate=16000, return_tensors="pt").input_values.to(dev)
                a, g, h = model(x); outs.append((float(a.clamp(0, 1)), g[0].cpu().numpy(), h[0].cpu().numpy()))
            ages.append(np.mean([o[0] for o in outs])); gen.append(np.mean([o[1] for o in outs], 0)); emb.append(np.mean([o[2] for o in outs], 0))
            if i % 1000 == 0: print(i, flush=True)
    np.savez(out, age=np.array(ages, np.float32), gender=np.stack(gen).astype(np.float32), emb=np.stack(emb).astype(np.float16))
