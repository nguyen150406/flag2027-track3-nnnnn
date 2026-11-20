"""Embedding loaders for FLAG (train+dev rows, canonical row order) and MAV auxiliaries."""
import os
from pathlib import Path
import numpy as np
from t3lib import ROOT, T2, T3
from run_t2_b01 import load_v1_english_auxiliary, load_v3_english_auxiliary, replace_mav_face_encoder

FACE = {"farl": T2 / "img01_b01_cache/direct", "ada": ROOT / "flag_raw_s01_v3_adaface_simam100",
        "clip": T2 / "clip01_cache/direct",
        "clipft": T3 / "caches/clipft",  # LoRA fine-tuned CLIP (t3_ft_clip.py --fold -1); external rows: caches/ext/<src>_face_clipft.*
        "farlft": T3 / "caches/farlft",  # LoRA fine-tuned FaRL (t3_ft_clip.py --encoder farl --fold -1); caches/ext/<src>_face_farlft.*
        "clipnf": T3 / "caches/clipnf",  # CLIP LoRA trained WITHOUT FLAG identities (t3_ft_clip.py --no_flag --tag clipnf)
        "clipnf2": T3 / "caches/clipnf2",  # ULTRA 2026-11-11: same recipe, LoRA seed 7 (seed ensemble with clipnf)
        "clipsa": T3 / "caches/clipsa",
        "clipmv": T3 / "caches/clipmv",
        "clipsa2": T3 / "caches/clipsa2",
        "clipmv2": T3 / "caches/clipmv2", "clipmv3": T3 / "caches/clipmv3",  # SUB48: clipmv recipe, LoRA seeds 3407 / 99
        "fcat2": T3 / "caches/fcat2",  # face concat: L2(farl) | L2(clip), 1280-d (early fusion of the two frozen face encoders, like voice cat4)
        "farlsa": T3 / "caches/farlsa",  # SUB50: FaRL LoRA with the South-Asian domain batch (clipsa recipe on FaRL), filtered data
        "sigft1": T3 / "caches/sigft1", "sigft2": T3 / "caches/sigft2", "clipft12": T3 / "caches/clipft12", "clipsa12": T3 / "caches/clipsa12", "clipft12r": T3 / "caches/clipft12r", "clipft12a": T3 / "caches/clipft12a", "clipft12b": T3 / "caches/clipft12b", "clipft12c": T3 / "caches/clipft12c", **{n: T3 / f"caches/{n}" for n in ["clipcf"] + [f"clipcf_d{h}" for h in range(5)]},  # SUB64 cross-fit LoRA (OOF train rows; _d<h> = LoRA h test rows)  # SUB56: SigLIP2 so400m LoRA, CLIP-v1 recipe, LoRA seeds 2027 / 7
        "vit": T3 / "caches/attr_vit",
        **{n: T3 / f"caches/{n}" for n in ("c336", "pel", "mc2h", "cl16", "clipmouth", "clipnose", "cliplow", "clipft12w", "clipsa12w", "ibv", "rff")}}  # A2 region crops; A3 WiSE-FT LoRA faces (2026-11-04)  # 2026-10-27 frozen face-encoder screen (t3_face_enc.py)  # R2 attribute branch: ViT-age face features (CLS + logits, 777-d); vox2 ext rows  # clipsa recipe, LoRA seed 7 (clipsa: 2027) -> seed variance + 3rd ensemble member  # clipsa recipe + MAV-Celeb v1 (Pakistani EN/Urdu, FLAG's sister corpus) as a LoRA source, seed 42  # CLIP LoRA v1 recipe + South-Asian domain batch (t3_ft_clip.py --domain caches/ext/domain_sa.csv --tag clipsa)
VOICE = {"sim": ROOT / "flag_raw_s01_v3_adaface_simam100", "wav": T2 / "wavlm_base_plus_caches/l4_mean"}
DIMS = {"farl": 512, "ada": 512, "clip": 768, "clipft": 768, "farlft": 512, "farlsa": 512, "sigft1": 1152, "sigft2": 1152, "clipft12": 768, "clipsa12": 768, "clipft12r": 768, "clipft12a": 768, "clipft12b": 768, "clipft12c": 768, **{n: 768 for n in ["clipcf"] + [f"clipcf_d{h}" for h in range(5)]}, "fcat2": 1280, "clipnf": 768, "clipnf2": 768,"clipsa": 768, "clipmv": 768, "clipsa2": 768, "clipmv2": 768, "clipmv3": 768, "sim": 256, "simfta": 256, "simftb": 256, "wsv": 512, "whs": 2560, "whc": 2560, "wsc": 512, "wll": 256, "rdn": 192, "vit": 777, "aud24": 1024, "wav": 768, "c336": 768, "pel": 1024, "mc2h": 1024, "cl16": 1024, "clipmouth": 768, "clipnose": 768, "cliplow": 768, "clipft12w": 768, "clipsa12w": 768,
        "r221": 256, "sbe": 192, "wl3": 1280, "iba": 1024, "ibl": 1024, "ibv": 1024, "rff": 1792}  # RF reproduction (rff = L2(clip) | L2(ibv))
MAV_FACE = {"farl": "farl", "ada": "adaface", "clip": "clip"}


def face(name):
    return np.load(FACE[name] / "face_embeddings.float32.npy", mmap_mode="r")


_LOCAL_VOICE = {"ecapa": "ecapa1024_lm", "campplus": "campplus_lm"}


def voice(name):
    p = VOICE.get(name, T3 / "caches/voice" / _LOCAL_VOICE.get(name, name))
    return np.load(p / "voice_embeddings.float32.npy", mmap_mode="r")


def to_aux(d, w):
    fl = np.empty(len(d["face"]), int); vl = np.empty(len(d["voice"]), int)
    for i, s in enumerate(d["speaker_ids"]):
        fl[list(d["face_rows"][s])] = i; vl[list(d["voice_rows"][s])] = i
    out = dict(faces=np.asarray(d["face"], np.float32), face_labels=fl,
               voices=np.asarray(d["voice"], np.float32), labels=vl, weight=w,
               face_names=np.empty(len(fl), object), voice_names=np.empty(len(vl), object))
    for i, s in enumerate(d["speaker_ids"]):
        out["face_names"][fl == i] = s; out["voice_names"][vl == i] = s
    if "face_video" in d and "voice_video" in d:
        out["face_video"] = np.asarray(d["face_video"]).astype(str); out["voice_video"] = np.asarray(d["voice_video"]).astype(str)
    return out


def drop_ids(a, ids):
    """Remove identities (by name) from an aux dict; relabel compactly."""
    fk = ~np.isin(a["face_names"], list(ids)); vk = ~np.isin(a["voice_names"], list(ids))
    names = sorted(set(a["face_names"][fk]) & set(a["voice_names"][vk])); lab = {n: i for i, n in enumerate(names)}
    fk &= np.isin(a["face_names"], names); vk &= np.isin(a["voice_names"], names)
    out = dict(a, faces=a["faces"][fk], voices=a["voices"][vk], face_names=a["face_names"][fk], voice_names=a["voice_names"][vk],
               face_labels=np.array([lab[n] for n in a["face_names"][fk]]), labels=np.array([lab[n] for n in a["voice_names"][vk]]))
    for k in ("face_video", "face_lang"):
        if k in a: out[k] = a[k][fk]
    for k in ("voice_video", "voice_lang"):
        if k in a: out[k] = a[k][vk]
    return out


_mav = {}
def mav(face_name, voice_name, w=1.0):
    voice_name = voice_name.split("@")[0]
    key = (face_name, voice_name)
    if key not in _mav:
        out = []
        for v, fn in (("v1", load_v1_english_auxiliary), ("v3", load_v3_english_auxiliary)):
            d = fn(512, DIMS[voice_name])
            if face_name != "farl":
                d = replace_mav_face_encoder(d, v, MAV_FACE[face_name], DIMS[face_name])
            out.append(to_aux(d, 1.0))
        _mav[key] = out
    return [dict(a, weight=w) for a in _mav[key]]


# ------------------------------------------------------------ full external sources (Track 3 extraction)
EXT = T3 / "caches/ext"          # Kaggle output: v1_*/vox_*/flag_voice_* arrays + rows_*.csv
V3F = T3 / "caches/mav_v3_faces"  # local: farl/clip for every v3 train frame (EN+DE)
V3V = T3 / "caches/mav_v3_voices"  # local: v3 train voices (EN+DE)
FLAG_EXTRA_VOICE = {"r293", "sam100ft", "wavlm4", "simfta", "simftb", "wsv", "wll", "rdn", "aud24", "whs", "whc", "wsc", "r221", "sbe", "wl3", "iba", "ibl"}  # simft*: t3_ft_voice.py (fine-tuned sim, seeds 2027 / 7); wsv/whs: t3_extract_wavlmsv.py
_VOICE_KEY = {"sim": "simam"}


FE = T3 / "caches/fe"   # front-end variants: flag_voice_<enc>_<variant>.float32.npy (Kaggle t3_frontend)


CAT = {"cat4": ["sim", "sam100ft", "r293", "ecapa"],  # concatenated voice input: per-encoder L2-normalised, stacked
       "cat6": ["sim", "sam100ft", "r293", "ecapa", "wsc", "whc"],  # cat4 + WavLM-SV + Whisper, both standardised on train rows (t3_center_voice.py)
       "rfv": ["sam100ft", "r221", "sbe", "wl3"],  # TP voice blocks: SimAM-R100 VB2-ft | ResNet221 | SpeechBrain ECAPA | Whisper-large-v3
       "cat4fa": ["simfta", "sam100ft", "r293", "ecapa"], "cat4fb": ["simftb", "sam100ft", "r293", "ecapa"]}  # sim part fine-tuned


def _l2(x):
    x = np.asarray(x, np.float32); return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def voice_full(name):
    """FLAG voice array (11349 rows) for any encoder name; 'enc@variant' selects a front-end variant."""
    if name in CAT:
        return np.hstack([_l2(voice_full(n)) for n in CAT[name]])
    if "@" in name:
        enc, var = name.split("@")
        return np.load(FE / f"flag_voice_{_VOICE_KEY.get(enc, enc)}_{var}.float32.npy", mmap_mode="r")
    if name in FLAG_EXTRA_VOICE:
        return np.load(EXT / f"flag_voice_{name}.float32.npy", mmap_mode="r")
    return voice(name)


def _labels(ids):
    u = {s: i for i, s in enumerate(sorted(set(ids)))}
    return np.array([u[s] for s in ids])


EXCLUDE = EXT / "exclude_bengali.csv"  # columns src, identity (scripts/t3_lid_filter.py from the VoxLingua107 LID kernel)
if os.environ.get("T3_EXCLUDE"): EXCLUDE = Path(os.environ["T3_EXCLUDE"])  # e.g. caches/ext/exclude_bengali_strict.csv (t3_strict_exclude.py)


def excluded(src):
    import pandas as pd
    if not EXCLUDE.exists(): raise FileNotFoundError(f"{EXCLUDE} missing: training without it would let Bengali speakers in (was a silent no-op)")
    e = pd.read_csv(EXCLUDE); return e.identity[e.src.eq(src)].astype(str).to_numpy()


def ext_source(src, face_name, voice_name, langs=None, max_faces_per_video=None, seed=0, w=1.0):
    """Aux dict for one external source with independent face/voice row tables."""
    import pandas as pd
    voice_name = voice_name.split("@")[0]  # external sources use the raw front-end
    venc = _VOICE_KEY.get(voice_name, voice_name)
    if src == "v3":
        fm = pd.read_csv(V3F / "faces.csv") if (V3F / "faces.csv").exists() else pd.read_csv(ROOT / "external_mavceleb/v3_full/faces.csv")
        F = np.load(V3F / f"{face_name}.float32.npy", mmap_mode="r")
        vm = pd.read_csv(V3V / "metadata.csv")
        V = (np.hstack([_l2(np.load(V3V / f"{_VOICE_KEY.get(n, n)}.float32.npy", mmap_mode="r")) for n in CAT[voice_name]])
             if voice_name in CAT else np.load(V3V / f"{venc}.float32.npy", mmap_mode="r"))
        fid, flang, fvid = fm.identity.values, fm.language.values, fm.video.values
        vid, vlang, vvid = vm.identity.values, vm.language.values, vm.video.values
    else:
        rows = pd.read_csv(EXT / f"rows_{src}.csv")
        F = np.load(EXT / f"{src}_face_{face_name}.float32.npy", mmap_mode="r")
        V = (np.hstack([_l2(np.load(EXT / f"{src}_voice_{_VOICE_KEY.get(n, n)}.float32.npy", mmap_mode="r")) for n in CAT[voice_name]])
             if voice_name in CAT else np.load(EXT / f"{src}_voice_{venc}.float32.npy", mmap_mode="r"))
        fid = vid = rows.identity.values
        vlang = rows.language.values if "language" in rows else np.array(["English"] * len(rows))
        flang = rows.face.str.split("/").str[1].values if src == "v1" else vlang
        fvid = rows.face.str.split("/").str[-2].values
        vvid = rows.voice.str.split("/").str[-2].values
        fok = np.load(EXT / f"{src}_face_{face_name}.ok.npy")
        vok = (np.logical_and.reduce([np.load(EXT / f"{src}_voice_{_VOICE_KEY.get(n, n)}.ok.npy") for n in CAT[voice_name]])
               if voice_name in CAT else np.load(EXT / f"{src}_voice_{venc}.ok.npy"))
    fsel = np.ones(len(fid), bool); vsel = np.ones(len(vid), bool)
    ex = excluded(src)  # Bengali speakers (FAQ: no Bengali speech may train the audio side) -> drop their faces and voices
    fsel &= ~np.isin(fid, ex); vsel &= ~np.isin(vid, ex)
    if src != "v3":
        fsel &= fok; vsel &= vok
    if langs is not None:
        fsel &= np.isin(flang, langs); vsel &= np.isin(vlang, langs)
    if max_faces_per_video:
        rng = np.random.default_rng(seed); keep = np.zeros(len(fid), bool)
        import collections
        groups = collections.defaultdict(list)
        for i in np.flatnonzero(fsel): groups[(fid[i], fvid[i])].append(i)
        for g in groups.values():
            keep[rng.choice(g, min(len(g), max_faces_per_video), replace=False)] = True
        fsel &= keep
    ids = sorted(set(fid[fsel]) & set(vid[vsel]))
    fsel &= np.isin(fid, ids); vsel &= np.isin(vid, ids)
    lab = {s: i for i, s in enumerate(ids)}
    return dict(faces=np.asarray(F[fsel], np.float32), face_labels=np.array([lab[s] for s in fid[fsel]]),
                voices=np.asarray(V[vsel], np.float32), labels=np.array([lab[s] for s in vid[vsel]]), weight=w,
                face_names=np.asarray(fid[fsel]).astype(object), voice_names=np.asarray(vid[vsel]).astype(object),
                face_video=np.asarray(fvid[fsel]).astype(str), voice_video=np.asarray(vvid[vsel]).astype(str),
                face_lang=np.asarray(flang[fsel]).astype(str), voice_lang=np.asarray(vlang[vsel]).astype(str),
                face_rows=np.flatnonzero(fsel), voice_rows=np.flatnonzero(vsel))


def ext_source_langs(src, voice_name):
    """Language per voice row in the same order/selection as ext_source(src, ..., langs=None)."""
    import pandas as pd
    venc = _VOICE_KEY.get(voice_name, voice_name)
    if src == "v3":
        return pd.read_csv(V3V / "metadata.csv").language.values
    rows = pd.read_csv(EXT / f"rows_{src}.csv")
    ok = np.load(EXT / f"{src}_voice_{venc}.ok.npy")
    lang = rows.language.values if "language" in rows else np.array(["English"] * len(rows))
    return lang[ok]


_GENDERS = None
def gender_map():
    """Identity name -> is_male for external sources (VoxCeleb1 metadata, MAV v1 metadata, MAV v3 FaRL-head majority)."""
    global _GENDERS
    if _GENDERS is None:
        import pandas as pd
        g = {}
        vox = pd.read_csv(EXT / "rows_vox.csv"); g.update(dict(zip(vox.identity, vox.gender.str.lower().str[0] == "m")))
        for extra in ("rows_vox2.csv", "rows_vox2b.csv", "rows_vox2c.csv", "rows_vb.csv"):
            if (EXT / extra).exists():
                v2 = pd.read_csv(EXT / extra); g.update(dict(zip(v2.identity, v2["gender"].astype(str).str.lower().str[0] == "m")))
        try:  # MAV v1/v3 metadata (no longer used for training; absent from the Kaggle/Colab bundles)
            m1 = pd.read_csv(ROOT / "external_mavceleb/v1_complete/meta_v1.csv"); g.update(dict(zip(m1.ids, m1.gender.str.lower().str[0] == "m")))
            # 13/11 review: MAV v3 ids reuse the v1 names (id0001..id0050, different people); the old v3 update overwrote 25 of the 64
            # v1 genders used by t3_mav_english / t3_r2.mv_trials -> v3 removed here (t3eval3 keeps its own v3 map)
        except (FileNotFoundError, ImportError):
            pass
        _GENDERS = g
    return _GENDERS


def attach_gender(a):
    """Add label_male (per label index) to an aux dict that carries face/voice names. Sources without real gender labels
    (LRS3, AVSpeech, Seamless: identities absent from gender_map) are returned unchanged, so the same-gender aux loss skips them."""
    gm = gender_map(); n = int(max(a["labels"].max(), a["face_labels"].max())) + 1
    names = np.unique(np.asarray(a["voice_names"]).astype(str))
    if np.mean([x in gm for x in names]) < 0.5:
        return a
    lm = np.zeros(n, bool)
    for lab, name in zip(a["labels"], a["voice_names"]):
        lm[lab] = bool(gm.get(name, False))
    return dict(a, label_male=lm)


def gender_split(aux, g):
    """Gender-specific projector data (2026-10-27): keep only the identities of gender g ('m'/'f') in every aux dict. Sources
    without gender labels (LRS3, AVSpeech) get an identity gender predicted from their faces: logistic regression on the
    labelled sources' faces (same face encoder, <= 10k rows each), identity-mean P(male) > .5."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    gm = gender_map(); names = [np.asarray(a["face_names"]).astype(str) for a in aux]
    has = [np.mean([n in gm for n in np.unique(fn)]) >= .5 for fn in names]
    male = {n: gm[n] for fn, h in zip(names, has) if h for n in np.unique(fn) if n in gm}
    if not all(has):
        rng = np.random.default_rng(0); X, y = [], []
        for a, fn, h in zip(aux, names, has):
            if h:
                ok = np.flatnonzero(np.isin(fn, list(male))); r = rng.choice(ok, min(10000, len(ok)), replace=False)
                X.append(a["faces"][r]); y += [male[n] for n in fn[r]]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)).fit(np.vstack(X), y)
        for a, fn, h in zip(aux, names, has):
            if not h: male.update((pd.Series(clf.predict_proba(a["faces"])[:, 1]).groupby(fn).mean() > .5).to_dict())
    want = g == "m"
    return [drop_ids(a, [n for n in np.unique(fn) if male.get(n, not want) != want]) for a, fn in zip(aux, names)]
