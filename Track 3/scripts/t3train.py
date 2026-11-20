"""Clean fixed-schedule face-voice projector trainer (no early stopping on held IDs).

Recipe mirrors Track 2 IMG13: LayerNorm->dropout->linear->L2 per modality, shared
AAM classifier over fit identities for both modalities, cross-modal SupCon and
batch prototype alignment; gender-balanced identity batches.
"""
from __future__ import annotations
import os, copy
import numpy as np, torch
import torch.nn.functional as F
from t3lib import *
from track2.model import (DualLinearProjector, cross_modal_supervised_contrastive_loss,
                          batch_identity_prototype_alignment_loss)

DEFAULT = dict(dim=192, face_dropout=.75, voice_dropout=.40, margin=.2, scale=30., w_con=.2, temp=.1,
               w_proto=.3, epochs=15, ids_per_batch=32, per_id=2, lr=3e-4, wd=1e-4, clip=5., decouple=False,
               aux_xvideo=False, w_sg=0.0, simplex=False, w_sg_aux=0.0, w_sah_aux=0.0, sah_ids=64,   # 13/11 ENX: same-gender FLAG-like aux hard batch
               w_sgh=0.0, beta_h=5.0, sg_topk=0, w_sga=0.0, age_thr=0.10, w_sgaam=0.0, w_grl=0.0, bil_rank=0, proj="linear", proj_rank=64, w_ortho=0.0, init_scale=1.0, trunc_k=0, split_f=0, split_v=0, adrop=.5, ema=0.0, mix=0.0, aux_ids=0, pre_steps=0,
               sched="", warmup=0.03, lr_min=0.0, merge_aux=False,
               w_uni=0.0, uni_t=2.0, w_hub=0.0, hub_k=10,  # training-time hubness reduction (2026-10-17, CSLS banned at scoring time)
               snaps="",  # "15,40,60": with on_snap, also hand over the model after these epochs (2026-10-26 training-length CV)
               w_main=1.0,  # R2 arm D (2026-11-12): weight of the FLAG main-batch losses (AAM/SupCon/proto + w_sg); 1.0 = unchanged
               gender="")  # "m"/"f": gender-specific projector on FLAG identities of that gender only (aux: t3data.gender_split)


def pair_logits(model, f, v):
    """Cross-modal similarity matrix: cosine + optional low-rank bilinear term (f U)(v V)^T."""
    f = F.normalize(f, dim=1); v = F.normalize(v, dim=1); S = f @ v.T
    if getattr(model, "bil_U", None) is not None:
        S = S + (f @ model.bil_U) @ (v @ model.bil_V).T
    return S


def supcon_logits(S, lab, temp):
    pos = lab[:, None].eq(lab[None, :]).to(S.dtype)
    def direction(L):
        L = L / temp; L = L - L.max(dim=1, keepdim=True).values.detach()
        lp = L - torch.logsumexp(L, dim=1, keepdim=True)
        return -((pos * lp).sum(1) / pos.sum(1)).mean()
    return 0.5 * (direction(S) + direction(S.T))


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -g


_ID_AGE = None
def id_age():
    """Identity-median face age (mean of ViT-age and MiVOLO) over FLAG TRAIN rows only."""
    global _ID_AGE
    if _ID_AGE is None:
        import pandas as pd
        vit = np.load(ROOT / "Track 1/age_compatibility_experiments/cache/train_face_age.npy")
        mv = np.load(T3 / "caches/mivolo_face/face_age.npy")[:6485] / 100
        _ID_AGE = pd.Series((vit + mv) / 2).groupby(SPK).median().to_dict()
    return _ID_AGE


def masked_supcon(fe, ve, lab, cand, temp, weights=None):
    """Cross-modal SupCon where each anchor's negatives are restricted to `cand` (bool BxB, excludes
    positives) and optionally re-weighted (`weights`, BxB, normalised per row to mean 1 over cand)."""
    f = F.normalize(fe, dim=1); v = F.normalize(ve, dim=1)
    pos = lab[:, None].eq(lab[None, :])
    def direction(S, cand, w):
        S = S / temp
        S = S - S.max(dim=1, keepdim=True).values.detach()
        e = torch.exp(S)
        neg = (e * cand * (w if w is not None else 1.0)).sum(1, keepdim=True)
        denom = (e * pos).sum(1, keepdim=True) + neg
        lp = S - torch.log(denom + 1e-12)
        valid = cand.any(1)
        l = -((pos * lp).sum(1) / pos.sum(1))
        return l[valid].mean() if valid.any() else S.sum() * 0
    S = f @ v.T
    w1 = w2 = None
    if weights is not None:
        w1 = weights; w2 = weights.T
    return 0.5 * (direction(S, cand, w1) + direction(S.T, cand.T, w2))


def uniformity(x, t=2.0):
    """Wang & Isola (2020) uniformity: log mean exp(-t ||xi - xj||^2) over all pairs of L2-normalised rows (lower = more spread out)."""
    return torch.log(torch.exp(-t * torch.pdist(F.normalize(x, dim=1)).pow(2)).mean())


def hub_penalty(x, bank, own, k):
    """CSLS hub term learned at training time: mean similarity of each row of x to its k most similar identity prototypes of the
    external heads (bank, detached, L2-normalised), excluding the row's own prototype (own = bank index, -1 = none)."""
    S = F.normalize(x, dim=1) @ bank.T
    hit = own >= 0
    if hit.any():
        S = S.clone(); S[hit.nonzero().squeeze(1), own[hit]] = -2.0
    return S.topk(min(k, S.shape[1]), dim=1).values.mean()


def aam_logits(emb, W, labels, margin, scale):
    """Additive angular margin on the target logit only (the arc-cos/cos margin is computed for the B target entries, not
    the whole B x C matrix; same values as applying it everywhere and selecting the targets)."""
    cos = F.normalize(emb, dim=1) @ F.normalize(W, dim=1).T
    t = cos.gather(1, labels[:, None])
    target = torch.cos(torch.acos(t.clamp(-1 + 1e-6, 1 - 1e-6)) + margin)
    return scale * cos.scatter(1, labels[:, None], target)


def train_projector(faces, voices, rows, spk, seed=2027, aux=None, on_snap=None, **kw):
    """faces/voices: full arrays indexed by row; rows: fit row indices; spk: identity per row.
    aux: optional list of dicts(faces, voices, labels, weight) for external sources with own heads."""
    cfg = dict(DEFAULT, **kw)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    if cfg["gender"]:
        rows = rows[np.array([GENDER.get(s) == cfg["gender"] for s in spk[rows]])]
    ids = np.unique(spk[rows]); idx = {s: i for i, s in enumerate(ids)}
    by_id = {s: rows[spk[rows] == s] for s in ids}
    male = [s for s in ids if GENDER.get(s) == "m"]; female = [s for s in ids if GENDER.get(s) == "f"]
    if cfg["proj"] == "linear":
        model = DualLinearProjector(faces.shape[1], voices.shape[1], cfg["dim"], cfg["face_dropout"], cfg["voice_dropout"])
        if cfg["init_scale"] != 1.0:  # shrink the random init so the trained map is dominated by the learned part
            with torch.no_grad():
                model.face.linear.weight.mul_(cfg["init_scale"]); model.voice.linear.weight.mul_(cfg["init_scale"])
    else:
        from t3proj import Projector
        model = Projector(faces.shape[1], voices.shape[1], cfg["dim"], cfg["face_dropout"], cfg["voice_dropout"],
                          kind=cfg["proj"], **({"rank": cfg["proj_rank"]} if cfg["proj"] in ("lowrank", "orthres") else
                                               dict(split_f=cfg["split_f"], split_v=cfg["split_v"], adrop=cfg["adrop"]) if cfg["proj"] == "attrcat" else {}))
    if cfg["bil_rank"]:  # low-rank bilinear interaction W = I + U V^T (starts as plain cosine)
        model.bil_U = torch.nn.Parameter(torch.randn(cfg["dim"], cfg["bil_rank"]) * .01)
        model.bil_V = torch.nn.Parameter(torch.randn(cfg["dim"], cfg["bil_rank"]) * .01)
    dev = torch.device(os.environ.get("T3_TRAIN_DEVICE", "cpu"))  # cpu (default, unchanged) or mps/cuda
    model.to(dev)
    if cfg["simplex"]:  # maximum class separation: fixed regular-simplex class prototypes (not trained)
        from track2.model import regular_simplex
        W = regular_simplex(len(ids), cfg["dim"]).float().to(dev)
        params = list(model.parameters())
    else:
        W = torch.nn.Parameter((torch.randn(len(ids), cfg["dim"]) * .01).to(dev))
        params = list(model.parameters()) + [W]
    gmask = torch.tensor([GENDER.get(s) == "m" for s in ids]).to(dev)
    if cfg["w_grl"]:  # weak gender adversary on the shared identity space (gradient reversal)
        adv = torch.nn.Linear(cfg["dim"], 1).to(dev)
        params.append(adv.weight); params.append(adv.bias)
    heads = []
    if cfg["merge_aux"] and aux:  # ONE classifier over all external identities (labels offset per source): cross-source negatives
        sizes = [int(a["labels"].max()) + 1 for a in aux]; aux_off = np.r_[0, np.cumsum(sizes)[:-1]]
        head_all = torch.nn.Parameter((torch.randn(int(sum(sizes)), cfg["dim"]) * .01).to(dev)); params.append(head_all)
        heads = [None] * len(aux)
    else:
        for a in aux or []:
            h = torch.nn.Parameter((torch.randn(int(a["labels"].max()) + 1, cfg["dim"]) * .01).to(dev))
            heads.append(h); params.append(h)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])

    def groups(labels):  # label -> its row indices in ascending order (== np.flatnonzero(labels == c), computed once)
        o = np.argsort(labels, kind="stable"); sl = labels[o]
        cuts = np.flatnonzero(np.diff(sl)) + 1
        return {int(g[0]): o[a:b] for g, a, b in zip(np.split(sl, cuts), np.r_[0, cuts], np.r_[cuts, len(sl)])}
    aux_rows = [(np.unique(a["labels"]), groups(a["face_labels"]), groups(a["labels"])) for a in aux or []]
    hoff = (aux_off if cfg["merge_aux"] and aux else np.r_[0, np.cumsum([int(a["labels"].max()) + 1 for a in aux or []])])  # bank row offset per source
    aux_dev = [(torch.from_numpy(np.asarray(a["faces"], np.float32)).to(dev), torch.from_numpy(np.asarray(a["voices"], np.float32)).to(dev))
               if dev.type != "cpu" and os.environ.get("T3_AUX_DEVICE", "1") == "1" else None
               for a in aux or []]  # external rows kept on the accelerator (gathered per step); T3_AUX_DEVICE=0 keeps them in RAM
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if cfg["ema"] else None
    F_t = torch.from_numpy(np.asarray(faces, np.float32)).to(dev); V_t = torch.from_numpy(np.asarray(voices, np.float32)).to(dev)
    steps = cfg["epochs"] * max(1, len(rows) // (cfg["ids_per_batch"] * cfg["per_id"]))
    snap_at = {e * (steps // cfg["epochs"]): e for e in map(int, filter(None, str(cfg["snaps"]).split(",")))} if on_snap else {}
    assert not snap_at or (cfg["sched"] == "" and not cfg["ema"] and not cfg["trunc_k"]), "snapshots need constant LR, no EMA/trunc"
    half = cfg["ids_per_batch"] if cfg["gender"] else cfg["ids_per_batch"] // 2  # one gender: the whole FLAG batch from it

    def con(fe, ve, lab):
        if cfg["bil_rank"]:
            return supcon_logits(pair_logits(model, fe, ve), lab, cfg["temp"])
        return cross_modal_supervised_contrastive_loss(fe, ve, lab, cfg["temp"])

    def loss_on(fe, ve, lab, head):
        l = .5 * (F.cross_entropy(aam_logits(fe, head, lab, cfg["margin"], cfg["scale"]), lab)
                  + F.cross_entropy(aam_logits(ve, head, lab, cfg["margin"], cfg["scale"]), lab))
        l = l + cfg["w_con"] * con(fe, ve, lab)
        return l + cfg["w_proto"] * batch_identity_prototype_alignment_loss(fe, ve, lab)

    model.train()
    for step in range(-cfg["pre_steps"], steps):  # negative steps: external sources only (warm-up / pre-training)
        pre = step < 0
        pick = list(rng.choice(male, min(half, len(male)), replace=False)) + \
               list(rng.choice(female, min(half, len(female)), replace=False))
        b = np.concatenate([rng.choice(by_id[s], cfg["per_id"], replace=len(by_id[s]) < cfg["per_id"]) for s in pick])
        lab = torch.tensor([idx[s] for s in spk[b]]).to(dev)
        if cfg["decouple"]:  # face and voice rows drawn independently per ID -> positives are cross-row/cross-video
            bv = np.concatenate([rng.choice(by_id[s], cfg["per_id"], replace=len(by_id[s]) < cfg["per_id"]) for s in pick])
            fe, ve = model(F_t[b], V_t[bv])
        else:
            fe, ve = model(F_t[b], V_t[b])
        loss = loss_on(fe, ve, lab, W) * (0.0 if pre else cfg["w_main"])
        if cfg["w_sgh"] or cfg["sg_topk"] or cfg["w_sga"]:
            gm = gmask[lab]; same_g = gm[:, None].eq(gm[None, :]); notpos = ~lab[:, None].eq(lab[None, :])
            cand = same_g & notpos
            if cfg["w_sgh"]:  # soft hard-negative weighting within gender: w ∝ exp(beta * cos)
                with torch.no_grad():
                    sim = F.normalize(fe, dim=1) @ F.normalize(ve, dim=1).T
                    w = torch.exp(cfg["beta_h"] * sim) * cand
                    w = w / (w.sum(1, keepdim=True) + 1e-12) * cand.sum(1, keepdim=True)
                loss = loss + cfg["w_sgh"] * masked_supcon(fe, ve, lab, cand, cfg["temp"], w)
            if cfg["sg_topk"]:  # only the k most similar same-gender negatives per anchor
                with torch.no_grad():
                    sim = (F.normalize(fe, dim=1) @ F.normalize(ve, dim=1).T).masked_fill(~cand, -9)
                    kth = sim.topk(min(cfg["sg_topk"], sim.shape[1]), dim=1).values[:, -1:]
                    top = cand & (sim >= kth)
                loss = loss + cfg["w_sg"] * 0 + 0.5 * masked_supcon(fe, ve, lab, top, cfg["temp"])
            if cfg["w_sga"]:  # same gender AND similar (ID-median) age
                ages = torch.tensor([id_age()[ids[i]] for i in lab.tolist()]).to(dev)
                cand_a = cand & ((ages[:, None] - ages[None, :]).abs() < cfg["age_thr"])
                loss = loss + cfg["w_sga"] * masked_supcon(fe, ve, lab, cand_a, cfg["temp"])
        if cfg["w_grl"]:
            y = gmask[lab].float()
            for emb in (fe, ve):
                loss = loss + cfg["w_grl"] * 0.5 * F.binary_cross_entropy_with_logits(adv(_GRL.apply(emb)).squeeze(1), y)
        if cfg["w_sgaam"]:  # conditional AAM: classify identity among same-gender classes only
            gl = gmask[lab]
            other = gl[:, None] != gmask[None, :]
            for emb in (fe, ve):
                lg = aam_logits(emb, W, lab, cfg["margin"], cfg["scale"]).masked_fill(other, -1e4)
                loss = loss + cfg["w_sgaam"] * 0.5 * F.cross_entropy(lg, lab)
        if cfg["w_sg"] and not pre:  # same-gender hard negatives: contrast only within each gender
            gm = gmask[lab]
            for sel in (gm, ~gm):
                if sel.sum() > 2:
                    loss = loss + cfg["w_main"] * cfg["w_sg"] * con(fe[sel], ve[sel], lab[sel])
        mf, mv, ml, mw = [], [], [], 0.0  # merged-head batch (merge_aux)
        hubf, hubv, hubo = [fe], [ve], [torch.full((len(fe),), -1, device=dev)]  # every embedding of this step (w_uni / w_hub)
        for k, (a, h, (u, fgr, vgr), ad) in enumerate(zip(aux or [], heads, aux_rows, aux_dev)):
            labs = a["labels"]
            chosen = rng.choice(u, min(cfg["aux_ids"] or cfg["ids_per_batch"], len(u)), replace=False)
            fb, vb, lb = [], [], []
            for c in chosen:
                fr_ = fgr[int(c)]; vr_ = vgr[int(c)]
                vsel = rng.choice(vr_, cfg["per_id"])
                if cfg["aux_xvideo"] and "face_video" in a and "voice_video" in a:
                    fsel = []
                    for v_ in vsel:  # prefer a face from a different video (and language) than the voice
                        other = fr_[a["face_video"][fr_] != a["voice_video"][v_]]
                        fsel.append(rng.choice(other) if len(other) else rng.choice(fr_))
                else:
                    fsel = rng.choice(fr_, cfg["per_id"])
                fb += list(fsel); vb += list(vsel); lb += [c] * cfg["per_id"]
            if ad is None:
                fa_, va_ = model(torch.from_numpy(a["faces"][fb]).to(dev), torch.from_numpy(a["voices"][vb]).to(dev))
            else:
                fa_, va_ = model(ad[0][torch.as_tensor(np.asarray(fb, np.int64), device=dev)], ad[1][torch.as_tensor(np.asarray(vb, np.int64), device=dev)])
            lbt = torch.tensor(lb).to(dev)
            hubf.append(fa_); hubv.append(va_); hubo.append(lbt + int(hoff[k]))
            if cfg["merge_aux"]:  # same per-source sampling, but one joint batch / one head after the loop
                mf.append(fa_); mv.append(va_); ml.append(lbt + int(aux_off[k])); mw += a["weight"]; continue
            loss = loss + a["weight"] * loss_on(fa_, va_, lbt, h)
            if cfg["w_sg_aux"] and "label_male" in a:  # same-gender contrast on external identities too
                gm = torch.from_numpy(np.asarray(a["label_male"])[np.asarray(lb)]).to(dev)
                for sel in (gm, ~gm):
                    if sel.sum() > 2:
                        loss = loss + a["weight"] * cfg["w_sg_aux"] * cross_modal_supervised_contrastive_loss(fa_[sel], va_[sel], lbt[sel], cfg["temp"])
            if cfg["w_sah_aux"] and a.get("sah") is not None and not pre:   # 13/11 ENX (R2 arm EH idea): same-gender FLAG-like (South-Asian-looking) hard batch
                grp = a["sah"][int(rng.integers(2))]
                if len(grp) > 2:
                    fb2, vb2, lb2 = [], [], []
                    for c in rng.choice(grp, min(cfg["sah_ids"], len(grp)), replace=False):
                        fr_ = fgr[int(c)]; vr_ = vgr[int(c)]; vsel = rng.choice(vr_, cfg["per_id"])
                        for v_ in vsel:
                            other = fr_[a["face_video"][fr_] != a["voice_video"][v_]] if cfg["aux_xvideo"] and "face_video" in a and "voice_video" in a else fr_
                            fb2.append(rng.choice(other if len(other) else fr_))
                        vb2 += list(vsel); lb2 += [int(c)] * cfg["per_id"]
                    if ad is None:
                        fs_, vs_ = model(torch.from_numpy(a["faces"][fb2]).to(dev), torch.from_numpy(a["voices"][vb2]).to(dev))
                    else:
                        fs_, vs_ = model(ad[0][torch.as_tensor(np.asarray(fb2, np.int64), device=dev)], ad[1][torch.as_tensor(np.asarray(vb2, np.int64), device=dev)])
                    loss = loss + a["weight"] * cfg["w_sah_aux"] * con(fs_, vs_, torch.tensor(lb2).to(dev))
        if mf:  # joint external batch: AAM over all external identities + contrast/prototype across sources (weight = sum of source weights)
            loss = loss + mw * loss_on(torch.cat(mf), torch.cat(mv), torch.cat(ml), head_all)
        if cfg["w_uni"] and not pre:
            loss = loss + cfg["w_uni"] * .5 * (uniformity(torch.cat(hubf), cfg["uni_t"]) + uniformity(torch.cat(hubv), cfg["uni_t"]))
        if cfg["w_hub"] and not pre and aux:
            bank = F.normalize(torch.cat([head_all] if cfg["merge_aux"] else heads).detach(), dim=1); own = torch.cat(hubo)
            loss = loss + cfg["w_hub"] * .5 * (hub_penalty(torch.cat(hubf), bank, own, cfg["hub_k"]) + hub_penalty(torch.cat(hubv), bank, own, cfg["hub_k"]))
        if cfg["w_ortho"]:
            loss = loss + cfg["w_ortho"] * model.ortho_penalty()
        if cfg["sched"] == "cos" and not pre:  # linear warm-up, then cosine decay to lr_min*lr ("" = constant LR, unchanged)
            nw = max(1, int(cfg["warmup"] * steps))
            f = (step + 1) / nw if step < nw else \
                cfg["lr_min"] + (1 - cfg["lr_min"]) * .5 * (1 + np.cos(np.pi * (step - nw) / max(1, steps - nw)))
            for g in opt.param_groups:
                g["lr"] = cfg["lr"] * f
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["clip"]); opt.step()
        if ema is not None:  # exponential moving average of the projector weights (used at the end)
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point: ema[k].mul_(cfg["ema"]).add_(v, alpha=1 - cfg["ema"])
        if step + 1 in snap_at:  # model after E epochs = the final model of an E-epoch run (constant LR): training length CV in one run
            on_snap(snap_at[step + 1], copy.deepcopy(model).to("cpu").eval())
    if ema is not None:
        model.load_state_dict(ema)
    if cfg["trunc_k"]:  # post-hoc: keep the top-k singular directions of each effective map W·diag(gamma_LN)
        with torch.no_grad():
            for stem in (model.face, model.voice):
                ln, lin = stem.normalization, stem.linear
                A = lin.weight * ln.weight[None, :]; c = lin.weight @ ln.bias + lin.bias
                U, S, Vh = torch.linalg.svd(A, full_matrices=False); k = min(cfg["trunc_k"], len(S))
                lin.weight.copy_((U[:, :k] * S[:k]) @ Vh[:k]); lin.bias.copy_(c)
                ln.weight.fill_(1.0); ln.bias.zero_()
    model.to("cpu").eval()
    return model


def pair_cos(model, faces, voices, face_rows, voice_rows):
    with torch.no_grad():
        f, v = model(torch.from_numpy(np.asarray(faces[face_rows], np.float32)),
                     torch.from_numpy(np.asarray(voices[voice_rows], np.float32)))
        s = (f * v).sum(1)
        if getattr(model, "bil_U", None) is not None:  # pair-local bilinear term
            s = s + ((f @ model.bil_U) * (v @ model.bil_V)).sum(1)
    return s.numpy().astype(np.float64)


def oof_branch(faces, voices, seeds=(2027,), aux_fn=None, **kw):
    """OOF scores for the 4 canonical folds: {(fold, protocol): scores} (seed-mean)."""
    out = {}
    for fold in range(4):
        fit, _ = fold_ids(fold)
        rows = np.flatnonzero(np.isin(SPK, fit))
        models = [train_projector(faces, voices, rows, SPK, seed=s, aux=aux_fn(fold) if aux_fn else None, **kw) for s in seeds]
        for p in ("standard", "same_gender"):
            fr, _ = cell(fold, p)
            out[fold, p] = np.mean([pair_cos(m, faces, voices, fr.face_row.to_numpy(), fr.voice_row.to_numpy())
                                    for m in models], 0)
    return out
