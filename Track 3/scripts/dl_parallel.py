"""Parallel HTTP range download to a local file (resumable per chunk). Usage: python dl_parallel.py <url> <out> <size> [threads]"""
import sys, os, requests, threading, time
url, out, size = sys.argv[1], sys.argv[2], int(sys.argv[3]); T = int(sys.argv[4]) if len(sys.argv) > 4 else 12
CH = 64 << 20; chunks = [(a, min(size, a + CH) - 1) for a in range(0, size, CH)]
if not os.path.exists(out):
    with open(out, "wb") as f: f.truncate(size)
done_f = out + ".done"; done = set(open(done_f).read().split()) if os.path.exists(done_f) else set()
lock = threading.Lock(); todo = [c for c in chunks if str(c[0]) not in done]; t0 = time.time(); got = [0]
def worker():
    while True:
        with lock:
            if not todo: return
            a, b = todo.pop(0)
        for k in range(8):
            try:
                r = requests.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=120); r.raise_for_status()
                assert len(r.content) == b - a + 1
                with open(out, "r+b") as f: f.seek(a); f.write(r.content)
                with lock:
                    got[0] += len(r.content); open(done_f, "a").write(f"{a}\n")
                break
            except Exception:
                time.sleep(2 ** k)
        else:
            raise RuntimeError(f"chunk {a}")
ths = [threading.Thread(target=worker) for _ in range(T)]; [t.start() for t in ths]
while any(t.is_alive() for t in ths):
    time.sleep(30); print(f"{got[0] / 1e9:.2f} GB, {got[0] / (time.time() - t0) / 1e6:.1f} MB/s", flush=True)
print("complete", len(open(done_f).read().split()) == len(chunks), flush=True)
