"""Run a Python script and record every file it OPENS under a root folder (audit hook 'open'), split into read / written.
Used to build the sub126c release (which files a step really needs). Paths are LOGICAL (abspath, symlinks not resolved), so a file reached
through a symlinked folder is listed under the folder the code used; <out>.links.txt lists those whose real location differs.
Run with PYTHONPYCACHEPREFIX=<empty dir> so imported modules are read from source (otherwise .pyc caches hide the .py files).
  python trace_open.py <out_prefix> <root> <script.py> [args ...]   -> <out_prefix>.read.txt / .write.txt / .links.txt (paths relative to root)"""
import sys, os, runpy, atexit
out, root = sys.argv[1], os.path.abspath(sys.argv[2]); script, args = sys.argv[3], sys.argv[4:]
R, Wr, LK = set(), set(), {}
def hook(ev, a):
    if ev != "open" or not a or not isinstance(a[0], (str, bytes, os.PathLike)): return
    lp = os.path.abspath(os.fsdecode(a[0]))
    if not lp.startswith(root + os.sep): return
    mode, flags = (a[1] if len(a) > 1 else None), (a[2] if len(a) > 2 else 0)
    w = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT))
    rel = os.path.relpath(lp, root); (Wr if w else R).add(rel)
    rp = os.path.realpath(lp)
    if rp != lp: LK[rel] = rp
sys.addaudithook(hook)
def dump():
    open(out + ".read.txt", "w").write("".join(f"{p}\n" for p in sorted(R))); open(out + ".write.txt", "w").write("".join(f"{p}\n" for p in sorted(Wr)))
    open(out + ".links.txt", "w").write("".join(f"{k}\t{v}\n" for k, v in sorted(LK.items())))
atexit.register(dump)
sys.argv = [script] + args; sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
