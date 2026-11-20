"""Re-implementation (own code) of the teammate's XJ / C3 face-voice model from XJ_ens5_submission1.md (13/11/2026).
Inputs = organizer (BTC) features: VGGFace fc7 4096-d faces, ECAPA 192-d voices. Two MLP towers, symmetric InfoNCE, centering by
the FLAG-train mean embedding, score = cosine (file stores -cosine). CPU only (never MPS)."""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data"; EV = DATA / "eval_set/evaluation_phase"
torch.set_num_threads(8)


def load_flag():
    L = pd.read_csv(DATA / "train_set/train_set/train_English.txt", sep=" ", header=None, names=["pid", "y", "voice", "face", "id", "age"])
    F = pd.read_csv(DATA / "train_set/features/faces/train_English_faces.csv", header=None).to_numpy(np.float32)
    V = pd.read_csv(DATA / "train_set/features/voices/train_English_voices.csv", header=None).to_numpy(np.float32)
    assert (F[:, -1] == L.age.values).all() and (V[:, -1] == L.age.values).all(), "train CSV last column = age"   # train CSVs carry age as an extra column
    F, V = F[:, :-1], V[:, :-1]
    g = pd.read_csv(DATA / "meta_file_train_set.csv", header=None, names=["id", "g"]).set_index("id").g
    assert len(L) == len(F) == len(V) == 6485 and F.shape[1] == 4096 and V.shape[1] == 192 and (L.y == 1).all()
    L["g"] = L.id.map(g); assert L.g.notna().all()
    return L, F, V


def flag_bengali_mask(L, rule="all3"):
    """True = FLAG-train clip flagged Bengali by our LID v3 run (outputs/lid_v3/vox/lid, src flag): rule all3 = all 3 models P(bn)+P(as) > .5
    (711 clips), any = any model (1492). Used to drop clips from TRAINING only (user 14/11 "oke lọc xong")."""
    l = pd.concat([pd.read_csv(ROOT / f"Track 3/outputs/lid_v3/vox/lid/lid_{i}of2.csv") for i in range(2)]); l = l[l.src.eq("flag")]
    P = l[["vl_bnas", "wh_bnas", "mms_bnas"]].to_numpy() > .5; fl = P.all(1) if rule == "all3" else P.any(1)
    key = l.key.str.replace("flag:", "", regex=False).str.split("/")
    m = dict(zip("voices/English/" + key.str[0] + "/" + key.str[1], fl)); out = L.voice.map(m)
    assert out.notna().all(), "LID missing for some FLAG clips"; return out.to_numpy(bool)


def load_eval(track, lang):
    rows = [l.split() for l in (EV / track / f"{lang}_valid.txt").read_text().splitlines() if l.strip()]
    L = pd.DataFrame([(r[0], r[1], "/".join(r[2:])) for r in rows], columns=["pid", "voice", "face"])   # gender/English_valid.txt has "faces 00000.jpg" (space) on every line
    F = pd.read_csv(EV / track / f"features/{lang}_valid_faces.csv", header=None).to_numpy(np.float32)
    V = pd.read_csv(EV / track / f"features/{lang}_valid_voices.csv", header=None).to_numpy(np.float32)
    assert len(L) == len(F) == len(V)
    for i, (v, f) in enumerate(zip(L.voice, L.face)): assert int(Path(v).stem) == i and int(Path(f).stem) == i, (track, lang, i)
    return L, F, V


def prep_face(F): F = np.sqrt(np.maximum(F, 0)); return F / np.linalg.norm(F, axis=1, keepdims=True).clip(1e-12)
def prep_voice(V): return V / np.linalg.norm(V, axis=1, keepdims=True).clip(1e-12)


def tower(d, h=256, o=128, p=0.7): return nn.Sequential(nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, o))


class XJ(nn.Module):
    def __init__(self, df=4096, dv=192, h=256, o=128, p=0.7):
        super().__init__(); self.f, self.v = tower(df, h, o, p), tower(dv, h, o, p); self.t = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
        self.register_buffer("cf", torch.zeros(o)); self.register_buffer("cv", torch.zeros(o))
    def emb(self, F, V): return Fn.normalize(self.f(F), dim=-1), Fn.normalize(self.v(V), dim=-1)


def train(Ff, Vv, seed, epochs=3, bs=128, lr=1e-3, wd=1e-4, ext=None, ext_ratio=4, center_rows=None, ids=None, mask_same=False,
          hidden=256, outd=128, genders=None, ext_gender=None, gpure=False, drop=0.7):
    """Ff, Vv: prepared FLAG pairs (row i = a matched pair). ext: optional callable(rng) -> (Fe, Ve) new external pairs for one epoch.
    center_rows: (F, V) prepared FLAG-train rows whose mean embedding becomes the fixed centre (defaults to the training FLAG rows)."""
    """gpure: gender-pure batches (genders = FLAG row genders m/f, ext_gender = dict external key -> m/f; unknown -> mixed batches)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = XJ(Ff.shape[1], Vv.shape[1], hidden, outd, drop)
    if gpure: assert genders is not None and len(genders) == len(Ff) and (ext is None or ext_gender is not None)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        F_, V_ = Ff, Vv; K_ = np.asarray(ids, object) if ids is not None else None
        if ext is not None:
            r = ext(rng, ext_ratio * len(Ff)); Fe, Ve = r[0], r[1]; F_, V_ = np.concatenate([Ff, Fe]), np.concatenate([Vv, Ve])
            if K_ is not None: K_ = np.concatenate([K_, np.asarray(r[2], object)])
        if mask_same: codes = pd.factorize(K_)[0]
        perm = rng.permutation(len(F_)); m.train()
        if gpure:
            G_ = np.asarray(genders, object)
            if ext is not None: G_ = np.concatenate([G_, np.array([ext_gender.get(k, "u") for k in r[2]], object)])
            G_ = np.where(np.isin(G_, ["m", "f"]), G_, "u"); bl = []
            for gg in ("m", "f", "u"):
                q = perm[G_[perm] == gg]; bl += [q[s:s + bs] for s in range(0, len(q), bs)]
            batches = [bl[j] for j in rng.permutation(len(bl))]
        else: batches = [perm[s:s + bs] for s in range(0, len(perm) - 1, bs)]
        for ix in batches:
            if len(ix) < 2: continue
            ef, ev = m.emb(torch.from_numpy(F_[ix]), torch.from_numpy(V_[ix])); sc = m.t.exp().clamp(max=100)
            lg = sc * ef @ ev.T; tgt = torch.arange(len(ix))
            if mask_same:   # same identity off the diagonal = false negative -> excluded from the softmax
                c = torch.from_numpy(codes[ix]); same = (c[:, None] == c[None, :]) & ~torch.eye(len(ix), dtype=torch.bool); lg = lg.masked_fill(same, -1e4)
            loss = (Fn.cross_entropy(lg, tgt) + Fn.cross_entropy(lg.T, tgt)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); cF, cV = center_rows if center_rows is not None else (Ff, Vv)
    with torch.no_grad():
        ef, ev = m.emb(torch.from_numpy(cF), torch.from_numpy(cV)); m.cf.copy_(ef.mean(0)); m.cv.copy_(ev.mean(0))
    return m



def train_nf(Ff, Vv, seed, epochs=3, bs=128, lr=1e-3, wd=1e-4, ext=None, ext_ratio=4, center_rows=None, hidden=256, outd=128, drop=0.7, **unused):
    """14/11 no-FLAG variant of train(): identical loop/optimiser/batching, but every epoch uses ONLY external pairs ((ext_ratio + 1) x len(Ff) draws,
    capped by availability); FLAG rows (Ff, Vv) are used only for the fixed centre (unsupervised mean embedding)."""
    assert ext is not None and not unused.get("mask_same") and not unused.get("gpure"), "no-FLAG training needs external data; mask/gpure not supported"
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = XJ(Ff.shape[1], Vv.shape[1], hidden, outd, drop)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        r = ext(rng, (ext_ratio + 1) * len(Ff)); F_, V_ = r[0], r[1]
        perm = rng.permutation(len(F_)); m.train()
        for ix in [perm[s:s + bs] for s in range(0, len(perm) - 1, bs)]:
            if len(ix) < 2: continue
            ef, ev = m.emb(torch.from_numpy(F_[ix]), torch.from_numpy(V_[ix])); sc = m.t.exp().clamp(max=100)
            lg = sc * ef @ ev.T; tgt = torch.arange(len(ix)); loss = (Fn.cross_entropy(lg, tgt) + Fn.cross_entropy(lg.T, tgt)) / 2
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); cF, cV = center_rows if center_rows is not None else (Ff, Vv)
    with torch.no_grad():
        ef, ev = m.emb(torch.from_numpy(cF), torch.from_numpy(cV)); m.cf.copy_(ef.mean(0)); m.cv.copy_(ev.mean(0))
    return m

@torch.no_grad()
def score(m, F, V):
    """cosine after fixed centering (higher = same person); F, V prepared, row i vs row i"""
    m.eval(); ef, ev = m.emb(torch.from_numpy(F), torch.from_numpy(V))
    ef = Fn.normalize(ef - m.cf, dim=-1); ev = Fn.normalize(ev - m.cv, dim=-1); return (ef * ev).sum(1).numpy()


def eer(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr; i = np.nanargmin(np.abs(fnr - fpr)); return 100 * float(fpr[i] + fnr[i]) / 2


def folds7(L, rep):
    """7 gender-balanced folds of FLAG ids, repetition rep"""
    rng = np.random.default_rng(1000 + rep); ids = L.drop_duplicates("id")[["id", "g"]]; out = [[] for _ in range(7)]
    for g in ("m", "f"):
        u = rng.permutation(ids[ids.g == g].id.values)
        for i, x in enumerate(u): out[i % 7].append(x)
    return out


def trials(L, test_ids, seed):
    """per test voice row: 1 positive (another face row of the same id) + 1 negative (face of another test id); gender track: same gender"""
    rng = np.random.default_rng(seed); T = L[L.id.isin(test_ids)]; rows = {k: v.index.values for k, v in T.groupby("id")}
    out = {"no_gender": [], "gender": []}
    for r, idv, g in zip(T.index, T.id, T.g):
        pos = rng.choice([x for x in rows[idv] if x != r])
        for tr in out:
            cand = [k for k in rows if k != idv and (tr == "no_gender" or L.g.at[rows[k][0]] == g)]
            neg = rng.choice(rows[rng.choice(cand)]); out[tr] += [(r, pos, 1), (r, neg, 0)]
    return {k: np.array(v) for k, v in out.items()}
