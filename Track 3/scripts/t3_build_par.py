"""Parallel front-end for t3_build.py: trains every (branch, seed) projector in a single-threaded worker pool,
then runs the unchanged t3_build.main with its train_projector replaced by a lookup of those models
(same branch/seed order, same data, same trainer config -> same submission logic).
Usage: python t3_build_par.py <config.json> <out_dir> [workers=10]"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import multiprocessing as mp


def job(args):
    bi, seed, cfg_path = args
    ng = int(os.environ.get("T3_NGPU", "0"))
    if ng and "CUDA_VISIBLE_DEVICES" not in os.environ:  # spread pool workers over the GPUs (set before torch is imported)
        os.environ["CUDA_VISIBLE_DEVICES"] = str((mp.current_process()._identity[0] - 1) % ng)
    if int(os.environ.get("T3_MPS_WORKERS", "0")):  # Mac: the first k pool workers train on MPS, the rest on CPU
        os.environ["T3_TRAIN_DEVICE"] = "mps" if mp.current_process()._identity[0] <= int(os.environ["T3_MPS_WORKERS"]) else "cpu"
    import torch; torch.set_num_threads(1)
    import numpy as np
    import t3_build as TB, t3data as D
    from t3lib import SPK
    cfg = json.load(open(cfg_path)); b = cfg["branches"][bi]
    F, V = np.asarray(D.face(b["face"])), np.asarray(D.voice_full(b["voice"]))
    aux = TB.aux_for(b["face"], b["voice"], b.get("aux", cfg.get("aux", [])))
    if cfg.get("trainer", {}).get("w_sg_aux"):
        aux = [D.attach_gender(a) for a in aux]
    return bi, seed, TB.train_projector(F, V, np.arange(6485), SPK, seed=seed, aux=aux, **cfg.get("trainer", {}))


if __name__ == "__main__":
    cfg_path, out_dir = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    cfg = json.load(open(cfg_path))
    if os.path.exists(out_dir):
        raise FileExistsError(out_dir)
    import t3_build as TB  # sets up sys.path (track2 model classes) so worker results can be unpickled here
    jobs = [(bi, s, cfg_path) for bi in range(len(cfg["branches"])) for s in cfg["seeds"]]
    import torch
    ck = os.path.abspath(out_dir) + ".models"; os.makedirs(ck, exist_ok=True)  # every trained projector is kept (resume + reuse)
    models, t0 = {}, time.time()
    for bi, s, _ in jobs:
        f = os.path.join(ck, f"b{bi}_s{s}.pt")
        if os.path.exists(f): models[bi, s] = torch.load(f, weights_only=False)
    todo = [j for j in jobs if (j[0], j[1]) not in models]
    print(f"{len(jobs)} projector trainings ({len(jobs) - len(todo)} cached in {ck}) on {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(workers) as pool:
        for bi, seed, m in pool.imap_unordered(job, todo):
            models[bi, seed] = m; torch.save(m, os.path.join(ck, f"b{bi}_s{seed}.pt"))
            print(f"  [{(time.time() - t0) / 60:5.1f}m] {cfg['branches'][bi]['name']} seed {seed}", flush=True)
    queue = iter([(s, models[bi, s]) for bi in range(len(cfg["branches"])) for s in cfg["seeds"]])

    def trained(F, V, rows, spk, seed=None, aux=None, **kw):  # t3_build.main asks in branch-then-seed order
        s, m = next(queue)
        assert s == seed, (s, seed)
        return m
    TB.train_projector = trained
    TB.main(cfg_path, out_dir)
    print(f"total {(time.time() - t0) / 60:.1f} min", flush=True)
