"""Projector families for Track 3 (each returns L2-normalised face/voice embeddings; pair-local)."""
import torch
from torch import nn
import torch.nn.functional as F


class Stem(nn.Module):
    """One modality: LayerNorm -> Dropout -> mapping -> L2."""
    def __init__(self, din, dim, drop, kind, rank=64, hidden=1024, shared=None, split=None, adrop=.5):
        super().__init__()
        self.kind = kind
        if kind == "attrcat":  # identity path (all sources) + attribute path (rows with attributes only; zeros = missing)
            self.split = split
            self.norm, self.drop, self.lin = nn.LayerNorm(split), nn.Dropout(drop), nn.Linear(split, dim)
            self.anorm, self.adrop, self.alin = nn.LayerNorm(din - split), nn.Dropout(adrop), nn.Linear(din - split, dim)
            return
        self.norm, self.drop = nn.LayerNorm(din), nn.Dropout(drop)
        if kind in ("linear", "ortho"):
            self.lin = nn.Linear(din, dim)
        elif kind == "lowrank":
            self.down, self.up = nn.Linear(din, rank, bias=False), nn.Linear(rank, dim)
        elif kind == "mlp":
            self.h, self.o, self.hd = nn.Linear(din, hidden), nn.Linear(hidden, dim), nn.Dropout(.2)
        elif kind == "resmlp":  # linear map + zero-initialised nonlinear residual
            self.lin = nn.Linear(din, dim); self.h = nn.Linear(din, 256); self.o = nn.Linear(256, dim)
            nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)
        elif kind == "gated":   # linear map + gated nonlinear correction (sigmoid gate per output dim)
            self.lin = nn.Linear(din, dim); self.h = nn.Linear(din, 256); self.o = nn.Linear(256, dim)
            self.g = nn.Linear(din, dim); nn.init.zeros_(self.o.weight); nn.init.zeros_(self.o.bias)
            nn.init.constant_(self.g.bias, -2.0)
        elif kind in ("orthhard", "orthres"):  # hard (semi-)orthogonal map, + zero-init low-rank residual
            self.Wfree = nn.Parameter(torch.randn(dim, din) / din ** .5)  # orthogonalised by QR every forward
            self.bias = nn.Parameter(torch.zeros(dim))
            if kind == "orthres":
                self.rd = nn.Linear(din, rank, bias=False); self.ru = nn.Linear(rank, dim, bias=False)
                nn.init.zeros_(self.ru.weight)
        elif kind == "shared":  # private stem -> shared trunk (shared across modalities)
            self.h = nn.Linear(din, 512); self.shared = shared
        else:
            raise ValueError(kind)

    def ortho_W(self):
        W = self.Wfree
        if W.shape[0] >= W.shape[1]:  # tall: orthonormal columns
            Q, R = torch.linalg.qr(W); return Q * torch.sign(torch.diagonal(R))[None, :]
        Q, R = torch.linalg.qr(W.T); return (Q * torch.sign(torch.diagonal(R))[None, :]).T  # wide: orthonormal rows

    def forward(self, x):
        if self.kind == "attrcat":
            xi, xa = x[:, :self.split], x[:, self.split:]
            has = (xa.abs().sum(1, keepdim=True) > 0).to(x.dtype)
            z = self.lin(self.drop(self.norm(xi))) + has * self.alin(self.adrop(self.anorm(xa)))
            return F.normalize(z, dim=-1)
        x = self.drop(self.norm(x)); k = self.kind
        if k in ("linear", "ortho"): z = self.lin(x)
        elif k == "lowrank": z = self.up(self.down(x))
        elif k == "mlp": z = self.o(self.hd(F.gelu(self.h(x))))
        elif k == "resmlp": z = self.lin(x) + self.o(F.gelu(self.h(x)))
        elif k == "gated": z = self.lin(x) + torch.sigmoid(self.g(x)) * self.o(F.gelu(self.h(x)))
        elif k == "shared": z = self.shared(F.gelu(self.h(x)))
        elif k in ("orthhard", "orthres"):
            z = x @ self.ortho_W().T + self.bias
            if k == "orthres": z = z + self.ru(self.rd(x))
        return F.normalize(z, dim=-1)


class Projector(nn.Module):
    def __init__(self, dface, dvoice, dim, fdrop, vdrop, kind="linear", **kw):
        super().__init__()
        shared = nn.Linear(512, dim) if kind == "shared" else None
        fkw, vkw = dict(kw), dict(kw)
        if kind == "attrcat":
            fkw = dict(split=kw["split_f"], adrop=kw.get("adrop", .5)); vkw = dict(split=kw["split_v"], adrop=kw.get("adrop", .5))
        self.face = Stem(dface, dim, fdrop, kind, shared=shared, **fkw)
        self.voice = Stem(dvoice, dim, vdrop, kind, shared=shared, **vkw)
        self.kind = kind

    def forward(self, f, v):
        return self.face(f), self.voice(v)

    def ortho_penalty(self):
        pen = 0.
        for s in (self.face, self.voice):
            W = s.lin.weight  # dim x din
            G = W @ W.T if W.shape[0] <= W.shape[1] else W.T @ W
            pen = pen + ((G / G.diagonal().mean() - torch.eye(G.shape[0])) ** 2).mean()
        return pen
