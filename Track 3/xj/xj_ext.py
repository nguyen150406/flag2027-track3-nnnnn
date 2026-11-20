"""External data for XJ (teammate's external_feats: MAV-Celeb v1/v2 all languages + VoxCeleb1, VGGFace fc7 + speechbrain ECAPA, BTC pipeline).
Joint sampler as in the XJ report: per epoch ext_ratio*N voice clips without replacement, each paired with a random face of the same id
(preferring the same video), re-drawn every epoch."""
import json, numpy as np, pandas as pd
from pathlib import Path
import xj_lib as X
D = Path(__file__).resolve().parent / "ext_feats"
OVERLAP = {"mavceleb_v2/id0073": "id039", "mavceleb_v2/id0036": "id047"}
T3 = Path(__file__).resolve().parents[1]


def bengali_keys():
    """Our Bengali rule on XJ's external ids (user 14/11: "vẫn phải lọc mấy cái bangla"): LID v3 any model P(bn)+P(as) > .5 in any window
    drops the identity; an identity without LID coverage is dropped too. MAV v1 = lid_mavx.csv (64 ids, English clips), MAV v2 = v2_lid_exclude.json
    'lid' (28 ids), VoxCeleb1 = exclude_bengali_v3.csv (26 India-nationality ids)."""
    import json, re
    l1 = pd.read_csv(T3.parent / "external_mavceleb/mavx_dl/lid_mavx.csv"); l1["id"] = l1.key.astype(str).str.extract(r"(id\d{4})")[0]
    bad1 = set(l1[(l1[["vl_bnas", "wh_bnas", "mms_bnas"]] > .5).any(axis=1)].id); cov1 = set(l1.id)
    v2 = json.load(open(T3 / "outputs/enx2r/v2_lid_exclude.json")); bad2 = {i.replace("v2_", "") for i in v2["lid"]}
    ex = pd.read_csv(T3 / "caches/ext/exclude_bengali_v3.csv"); badv = set(ex[ex.src.eq("vox")].identity.astype(str))
    here = Path(__file__).resolve().parent
    for lu in [here / "lid_xjx_urdu_hindi.csv", here / "lid_xjx_mav_english.csv", here / "mavx/mavx0_lid.csv", here / "mavx/mavx1_lid.csv"]:
      # 14/11: clip-level LID v3 on the Urdu/Hindi clips, on all kept MAV English clips and on the 4,761 MAVX clips -> a flagged (or unreadable)
      # clip drops its identity. Review 14/11: fail CLOSED -- a missing LID file must stop the run, never silently let flagged ids back in.
      assert lu.exists(), f"required clip-level LID file missing: {lu}"
      if True:
        u = pd.read_csv(lu); fl = (u[["vl_bnas", "wh_bnas", "mms_bnas"]] > .5).any(axis=1) | (u.dur <= 0); ui = u.key.str.extract(r"^(v[12]):(id\d{4})/")
        assert ui[1].notna().all(), f"unparsed LID keys in {lu.name}"
        bad1 |= set(ui[1][ui[0].eq("v1") & fl.values]); bad2 |= set(ui[1][ui[0].eq("v2") & fl.values])
    return {"v1_bad": bad1, "v1_cov": cov1, "v2_bad": bad2, "vox_bad": badv}


LID3 = Path(__file__).resolve().parent / "lid_xjx3.csv"


def clip_lid():
    """CLIP-level Bengali rule (user 14/11 "Có làm đi 2 điều luôn": drop exactly the flagged clips, keep the identity).
    Returns (checked, flagged) sets of LID keys: MAV 'v1:|v2:<id>/<Lang>/<video>/<clip>.wav', VoxCeleb1 'vox1:<id>/<video>/<clip>.wav'.
    flagged = any of the 3 LID v3 models gives P(bn)+P(as) > .5 in any 30-s window, or the clip is unreadable. Fail closed: every file must exist;
    a clip absent from all files counts as unchecked (and is dropped by Ext)."""
    here = Path(__file__).resolve().parent; cols = ["vl_bnas", "wh_bnas", "mms_bnas"]; parts = []
    for f, pre in ((here / "lid_xjx_urdu_hindi.csv", ""), (here / "lid_xjx_mav_english.csv", ""), (T3.parent / "external_mavceleb/mavx_dl/lid_mavx.csv", "v1:"), (LID3, "")):
        assert f.exists(), f"required clip-level LID file missing: {f}"
        u = pd.read_csv(f); k = pre + u.key.astype(str)
        assert k.str.match(r"^(v1|v2|vox1):id\d+/").all(), f"unparsed LID keys in {f.name}"
        fl = ~(u[cols] <= .5).all(axis=1) | ~(u.dur > 0)   # review 14/11: NaN probability or NaN duration counts as FLAGGED (fail closed)
        parts.append(pd.DataFrame({"k": k.values, "fl": fl.to_numpy()}))
    d = pd.concat(parts).groupby("k").fl.max()
    return set(d.index), set(d.index[d.values])


def clip_keys(vm):
    """LID key of every voices_meta row (MAV / VoxCeleb1), None for other sources and for MAVX rows (add_mavx keeps only LID-cleared clips)."""
    p = vm.path.astype(str); src = vm.source.to_numpy(); mx = p.str.startswith("mavx/").to_numpy()
    mav = p.str.extract(r"/(v[12])/voices/")[0] + ":" + p.str.split("/voices/").str[1]
    vox = "vox1:" + p.str.split("/wav/").str[1]
    return np.where((src == "mavceleb") & ~mx, mav, np.where(src == "voxceleb1", vox, None))


def bengali_drop(keys):
    B = bengali_keys(); out = set()
    for k in set(keys):
        src, i = k.split("/")
        if src == "mavceleb_v1" and (i in B["v1_bad"] or i not in B["v1_cov"]): out.add(k)
        elif src == "mavceleb_v2" and i in B["v2_bad"]: out.add(k)
        elif src == "voxceleb1" and i in B["vox_bad"]: out.add(k)
    return out   # XJ report 2.5 (face NN + z>4 on both); our v2 screen found the same 2


def key(m):
    ver = m.path.str.extract(r"/(v1|v2)/")[0]
    return np.where(m.source.eq("mavceleb"), "mavceleb_" + ver.fillna("?") + "/" + m.id.astype(str), m.source + "/" + m.id.astype(str))


class Ext:
    def __init__(self, langs=None, sources=None, bengali_filter=False, exclude_keys=(), vb=False, vb_frac=1.0, mav_lid_ok=False, vox2=False, mavx=False, mav_w=1.0,
                 bengali_clip=False):
        fm, vm = pd.read_csv(D / "faces_meta.csv"), pd.read_csv(D / "voices_meta.csv"); fm["k"], vm["k"] = key(fm), key(vm)
        assert not fm.k.str.contains(r"\?").any() and not vm.k.str.contains(r"\?").any()
        assert not vm.lang.str.lower().isin(["bengali", "bangla", "bn"]).any(), "no Bengali speech in external training data"
        F = np.load(D / "faces.npy", allow_pickle=False, mmap_mode="r"); V = np.load(D / "voices.npy", allow_pickle=False, mmap_mode="r")
        assert len(F) == len(fm) and len(V) == len(vm)
        if vb: fm, vm, F, V = add_vb(fm, vm, F, V, vb_frac)
        if vox2: fm, vm, F, V = add_vox2(fm, vm, F, V)
        if mavx: fm, vm, F, V = add_mavx(fm, vm, F, V)
        vk = np.ones(len(vm), bool)
        if langs is not None:   # language filter applies to MAV clips only (VoxCeleb1 = 'unk'; VoxBlink has its own identity-level LID filter)
            okl = vm.lang.isin(langs).to_numpy() | vm.source.isin(["voxblink", "voxceleb2"]).to_numpy()
            if mav_lid_ok: okl |= vm.path.isin(mav_lid_cleared()).to_numpy()   # Urdu/Hindi clips that passed LID v3 (all 3 models, every window)
            okl |= vm.path.astype(str).str.startswith("mavx/").to_numpy()   # MAVX rows: only LID-cleared clips are ever added
            vk &= okl
        if sources is not None: vk &= vm.source.isin(sources).to_numpy()
        if bengali_filter:
            drop = bengali_drop(vm.k); vk &= ~vm.k.isin(drop).to_numpy()
            print(f"bengali filter: drop {len(drop)} ids (v1 {sum(k.startswith('mavceleb_v1') for k in drop)}, v2 {sum(k.startswith('mavceleb_v2') for k in drop)}, vox1 {sum(k.startswith('voxceleb1') for k in drop)})", flush=True)
        if bengali_clip:   # 14/11: clip-level rule. Every MAV clip (MAVX: already cleared in add_mavx) and every clip of the 26 India-nationality
            # VoxCeleb1 ids must be LID-checked and not flagged; other VoxCeleb1 ids = organizer-permitted, identity-level audited before (0 flags).
            assert not bengali_filter, "choose ONE Bengali rule (identity-level bengali_filter or clip-level bengali_clip)"
            chk, fl = clip_lid(); ck = pd.Series(clip_keys(vm)); src = vm.source.to_numpy()
            ex = pd.read_csv(T3 / "caches/ext/exclude_bengali_v3.csv"); india = set(ex[ex.src.eq("vox")].identity.astype(str))
            mx = vm.path.astype(str).str.startswith("mavx/").to_numpy()   # review 14/11: need decided by source/id only; an unparsable path must stop the run
            need = ((src == "mavceleb") & ~mx) | ((src == "voxceleb1") & vm.id.astype(str).isin(india).to_numpy())
            assert ck[need].notna().all(), "clip-level LID: unparsable path among rows that need a LID check"
            okc = ~need | (ck.isin(chk).to_numpy() & ~ck.isin(fl).to_numpy())
            lost = set(vm.k[vk]) - set(vm.k[vk & okc])
            print(f"bengali clip filter: checked-needed {int((need & vk).sum())}, dropped flagged {int((need & vk & ck.isin(fl).to_numpy()).sum())}, "
                  f"dropped unchecked {int((need & vk & ~ck.isin(chk).to_numpy()).sum())}, ids left with no clip {len(lost)}", flush=True)
            vk &= okc
        vk &= ~vm.k.isin(set(exclude_keys)).to_numpy()   # validation hold-out identities (all their rows, every language)
        self.fm, self.vm = fm, vm[vk].reset_index(drop=True); fk = fm.k.isin(set(self.vm.k)).to_numpy()
        self.F = X.prep_face(np.asarray(F[fk], np.float32)); self.fm = fm[fk].reset_index(drop=True)
        self.V = X.prep_voice(np.asarray(V[np.flatnonzero(vk)], np.float32))
        self.mav_w = float(mav_w)   # 14/11 iteration 9: sampling weight of MAV-Celeb voice rows (1.0 = uniform = old path)
        self.byk = {k: g.index.values for k, g in self.fm.groupby("k")}; self.bykv = {kv: g.index.values for kv, g in self.fm.groupby(["k", "video"])}
        print(f"ext: voices {len(self.V)} faces {len(self.F)} ids {self.vm.k.nunique()}", flush=True)

    def sampler(self, drop_keys=()):
        ok = np.flatnonzero(~self.vm.k.isin(set(drop_keys)).to_numpy()); ks, vids = self.vm.k.to_numpy(), self.vm.video.to_numpy()
        pw = None
        if self.mav_w != 1.0: w = np.where(self.vm.source.to_numpy()[ok] == "mavceleb", self.mav_w, 1.0); pw = w / w.sum()
        def draw(rng, n):
            vi = rng.choice(ok, min(n, len(ok)), replace=False) if pw is None else rng.choice(ok, min(n, len(ok)), replace=False, p=pw); fi = np.empty(len(vi), int)
            for j, v in enumerate(vi):
                c = self.bykv.get((ks[v], vids[v])); c = c if c is not None and len(c) else self.byk[ks[v]]; fi[j] = c[rng.integers(len(c))]
            return self.F[fi], self.V[vi], ks[vi]
        return draw

    def drop_for(self, test_ids): return [k for k, f in OVERLAP.items() if f in set(test_ids)]

def gender_map():
    """external key -> 'm'/'f' for gender-pure batching (training only). MAV v1 + VoxCeleb1: ids.csv (v1 labels verified 98.6% vs a FLAG face
    classifier; v2 labels in ids.csv are wrong = chance). MAV v2: FLAG-trained face-gender LR on the identity's mean probability (>.8 m, <.2 f)."""
    c = D.parent / "ext_gender.json"
    if c.exists(): return json.load(open(c))
    from sklearn.linear_model import LogisticRegression
    L, F, _ = X.load_flag(); clf = LogisticRegression(C=1.0, max_iter=2000).fit(X.prep_face(F), (L.g.values == "m").astype(int))
    fm = pd.read_csv(D / "faces_meta.csv"); fm["k"] = key(fm); Fa = X.prep_face(np.load(D / "faces.npy", mmap_mode="r").astype(np.float32))
    p = pd.Series(clf.predict_proba(Fa)[:, 1]).groupby(fm.k.values).mean(); ids = pd.read_csv(D / "ids.csv"); g = ids.set_index(["source", "id"]).gender; out = {}
    for k in p.index:
        src, i = k.split("/")
        if src == "mavceleb_v1": v = g.get(("mavceleb", i))
        elif src == "voxceleb1": v = g.get(("voxceleb1", i))
        else: v = "m" if p[k] > .8 else "f" if p[k] < .2 else None
        if v in ("m", "f"): out[k] = v
    json.dump(out, open(c, "w")); return out

VBX = Path(__file__).resolve().parent / "vb_btc"


def vb_keep_ids():
    """VoxBlink identity filter (training only): Bengali speakers were already excluded at prep (VB metadata); here keep an identity only if
    our LID v3 web run covered it (missing = flagged) and no clip of it is flagged (any model P(bn)+P(as) > .5), and it is not on the ENX2/ultra
    exclusion list (LID flags, held-out ids)."""
    import glob
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(T3 / "outputs/lid_v3/web/lid/lid_*.csv"))]); d = d[d.src.eq("vb")]
    d["identity"] = d.key.str.split(":").str[1].str.split("/").str[0]
    bad = set(d.identity[(d[["vl_bnas", "wh_bnas", "mms_bnas"]] > .5).any(axis=1)])
    ex = pd.read_csv(T3 / "kaggle/enx2r_data/meta/exclude.csv"); bad |= set(ex.identity[ex.src.eq("vb")])
    return set(d.identity) - bad


def add_vb(fm, vm, F, V, frac=1.0):
    fr, vr = pd.read_csv(VBX / "vb_btc_face_rows.csv"), pd.read_csv(VBX / "vb_btc_voice_rows.csv")
    Fv, Vv = np.load(VBX / "vb_btc_face.f16.npy", mmap_mode="r"), np.load(VBX / "vb_btc_voice.f32.npy")
    assert len(Fv) == len(fr) and len(Vv) == len(vr)
    for r in (fr, vr): r["vbid"] = np.where(r.identity.str.startswith("vb_"), r.identity, "vb_" + r.identity)
    keep = vb_keep_ids(); lk = Path(__file__).resolve().parent / "vb_leak_drop.json"
    assert lk.exists(), "run vb_leak.py first"; keep -= set(json.load(open(lk)))   # VB ids that look like a FLAG id or a V3 hold-out id
    if frac < 1.0: u = sorted(keep); keep = set(np.random.default_rng(0).choice(u, int(frac * len(u)), replace=False))
    fk = (fr.ok & fr.vbid.isin(keep)).to_numpy(); vk = (vr.ok & vr.vbid.isin(keep)).to_numpy()
    f2 = pd.DataFrame(dict(source="voxblink", id=fr.vbid[fk].values, video=fr.video[fk].values, k="voxblink/" + fr.vbid[fk].values))
    v2 = pd.DataFrame(dict(source="voxblink", id=vr.vbid[vk].values, lang=vr.language[vk].values, video=vr.video[vk].values,
                           path="voxblink/" + vr.i[vk].astype(str).values, k="voxblink/" + vr.vbid[vk].values))
    assert not v2.lang.astype(str).str.lower().isin(["bengali", "bangla", "bn"]).any(), "no Bengali speech in external training data"
    print(f"voxblink: keep ids {len(keep)}; faces {fk.sum()} voices {vk.sum()} ids with voice {v2.k.nunique()}", flush=True)
    return (pd.concat([fm, f2], ignore_index=True), pd.concat([vm, v2], ignore_index=True),
            np.concatenate([np.asarray(F, np.float16), np.asarray(Fv[fk], np.float16)]), np.concatenate([np.asarray(V, np.float32), Vv[vk]]))


def mav_lid_cleared():
    """voices_meta paths of MAV Urdu/Hindi clips cleared by LID v3 (no model P(bn)+P(as) > .5 in any window, readable)."""
    d = pd.read_csv(Path(__file__).resolve().parent / "lid_xjx_urdu_hindi.csv")
    ok = ~(d[["vl_bnas", "wh_bnas", "mms_bnas"]] > .5).any(axis=1) & (d.dur > 0)
    L = pd.read_csv(T3 / "kaggle/xjx_lidlist/xjx_lid_clips.csv"); m = dict(zip(L.key, L.rel))
    return {"datasets/KAGGLE_USER/external-data-mavceleb/" + m[k] for k in d.key[ok]}


V2X = Path(__file__).resolve().parent / "vox2_btc"


def vox2_keep_ids():
    """VoxCeleb2 identity filter (training only): keep an identity only if our LID v3 vox run covered it (missing = flagged), none of its clips
    is flagged (any model P(bn)+P(as) > .5), and it is not on the ENX2/ultra exclusion list (LID flags, held-out ids)."""
    import glob
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(T3 / "outputs/lid_v3/vox/lid/lid_*.csv"))]); d = d[d.src.eq("vox2")]
    d["identity"] = d.key.str.split(":").str[1].str.split("/").str[0]
    bad = set(d.identity[(d[["vl_bnas", "wh_bnas", "mms_bnas"]] > .5).any(axis=1)])
    ex = pd.read_csv(T3 / "kaggle/enx2r_data/meta/exclude.csv"); bad |= set(ex.identity[ex.src.eq("vox2")])
    return set(d.identity) - bad


def add_vox2(fm, vm, F, V):
    fr, vr = pd.read_csv(V2X / "vox2_btc_face_rows.csv", dtype={"clip": str, "video": str}), pd.read_csv(V2X / "vox2_btc_voice_rows.csv", dtype={"clip": str, "video": str})
    Fv, Vv = np.load(V2X / "vox2_btc_face.f16.npy", mmap_mode="r"), np.load(V2X / "vox2_btc_voice.f32.npy")
    assert len(Fv) == len(fr) and len(Vv) == len(vr)
    keep = vox2_keep_ids(); lk = Path(__file__).resolve().parent / "vox2_leak_drop.json"
    assert lk.exists(), "run the vox2 leak check first"; keep -= set(json.load(open(lk)))
    fk = (fr.ok & fr.identity.isin(keep)).to_numpy(); vk = (vr.ok & vr.identity.isin(keep)).to_numpy()
    f2 = pd.DataFrame(dict(source="voxceleb2", id=fr.identity[fk].values, video=fr.video[fk].values, k="voxceleb2/" + fr.identity[fk].values))
    v2 = pd.DataFrame(dict(source="voxceleb2", id=vr.identity[vk].values, lang="unk", video=vr.video[vk].values,
                           path="voxceleb2/" + vr.identity[vk].values + "/" + vr.video[vk].values + "/" + vr["clip"][vk].values, k="voxceleb2/" + vr.identity[vk].values))
    print(f"voxceleb2: keep ids {len(keep)}; faces {fk.sum()} voices {vk.sum()} ids with voice {v2.k.nunique()}", flush=True)
    return (pd.concat([fm, f2], ignore_index=True), pd.concat([vm, v2], ignore_index=True),
            np.concatenate([np.asarray(F, np.float16), np.asarray(Fv[fk], np.float16)]), np.concatenate([np.asarray(V, np.float32), Vv[vk]]))


MXX = Path(__file__).resolve().parent / "mavx"


def add_mavx(fm, vm, F, V):
    """more MAV clips of kept identities (Kaggle flag-t3-xjx-mavx-run, 2 shards): a voice clip is added only if LID v3 cleared it (no model
    P(bn)+P(as) > .5 in any window, readable); a flagged clip drops its whole identity via bengali_keys(); keys = mavceleb_v1|v2/<id>."""
    fr, vr, Fs, Vs = [], [], [], []
    for sh in (0, 1):
        v = pd.read_csv(MXX / f"mavx{sh}_voice_rows.csv"); Vx = np.load(MXX / f"mavx{sh}_voice.f32.npy"); assert len(v) == len(Vx)
        d = pd.read_csv(MXX / f"mavx{sh}_lid.csv"); okl = ~(d[["vl_bnas", "wh_bnas", "mms_bnas"]] > .5).any(axis=1) & (d.dur > 0)
        lk = v.src.str.replace("mavceleb_", "") + ":" + v.rel.str.split("/voices/").str[1]; cleared = set(d.key[okl])
        assert set(lk) == set(d.key), f"shard {sh}: LID keys != selected clips"
        m = (v.ok & lk.isin(cleared)).to_numpy(); vr.append(v[m]); Vs.append(Vx[m])
        f = pd.read_csv(MXX / f"mavx{sh}_face_rows.csv"); Fx = np.load(MXX / f"mavx{sh}_face.f16.npy", mmap_mode="r"); assert len(f) == len(Fx)
        fr.append(f[f.ok]); Fs.append(np.asarray(Fx[f.ok.to_numpy()], np.float16))
    vr, fr = pd.concat(vr, ignore_index=True), pd.concat(fr, ignore_index=True)
    assert not vr.lang.isin(["bengali", "bangla"]).any()
    v2 = pd.DataFrame(dict(source="mavceleb", id=vr.id.values, lang=vr.lang.values, video=vr.video.values, path="mavx/" + vr.rel.values, k=vr.k.values))
    f2 = pd.DataFrame(dict(source="mavceleb", id=fr.id.values, video=fr.video.values, path="mavx/" + fr.rel.values, k=fr.k.values))
    print(f"mavx: +voices {len(v2)} (cleared) +faces {len(f2)} ids {v2.k.nunique()}", flush=True)
    return (pd.concat([fm, f2], ignore_index=True), pd.concat([vm, v2], ignore_index=True),
            np.concatenate([np.asarray(F, np.float16)] + Fs), np.concatenate([np.asarray(V, np.float32)] + Vs))

