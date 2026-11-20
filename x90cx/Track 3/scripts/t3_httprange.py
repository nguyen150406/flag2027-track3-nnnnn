"""Read-only seekable file over HTTP range requests (2026-10-18, SUB53: ReDimNet for VoxCeleb1 / AVSpeech without downloading whole
archives). zipfile / pyarrow read only the bytes they need: a zip's central directory and single members, a parquet footer and the
column chunks asked for (e.g. AVSpeech 'audio' without the much larger 'video' column). Usage: f = RangeFile(signed_url)."""
import io, time, requests


class RangeFile(io.RawIOBase):
    def __init__(self, url, session=None, retries=6):
        self.url, self.s, self.retries, self.pos = url, session or requests.Session(), retries, 0
        r = self._get(0, 0); self.size = int(r.headers["Content-Range"].split("/")[1]); self.fetched = 0

    def _get(self, a, b):
        for k in range(self.retries):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=300); r.raise_for_status()
                if r.status_code != 206: raise IOError(f"no range support (HTTP {r.status_code})")
                return r
            except Exception:
                if k == self.retries - 1: raise
                time.sleep(2 * (k + 1))

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b""
        d = self._get(self.pos, self.pos + n - 1).content; assert len(d) == n, (len(d), n)
        self.pos += n; self.fetched += n; return d

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


if __name__ == "__main__":  # self-check against a local file served as bytes (no network): emulate the HTTP layer
    class FakeSession:
        def __init__(self, data): self.data = data
        def get(self, url, headers, timeout):
            a, b = map(int, headers["Range"].split("=")[1].split("-")); r = requests.Response(); r.status_code = 206
            r._content = self.data[a:b + 1]; r.headers["Content-Range"] = f"bytes {a}-{b}/{len(self.data)}"; return r
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z: z.writestr("wav/a.wav", b"x" * 1000); z.writestr("wav/b.wav", b"y" * 10)
    f = RangeFile("mem://", FakeSession(buf.getvalue())); z = zipfile.ZipFile(f)
    assert z.read("wav/b.wav") == b"y" * 10 and sorted(z.namelist()) == ["wav/a.wav", "wav/b.wav"]
    print("t3_httprange self-check ok, fetched", f.fetched, "of", f.size, "bytes")
