"""Audio for the ReDimNet branches' missing external sources (SUB53, user 2026-10-18 "chạy A+B"): the clips the earlier pipelines used,
written as 16 kHz mono wavs in the layout t3_extract_wavlmsv.py reads.
  avs / avs2 : AVSpeech parquet files (HF mirror, signed URLs): only the 'clip_id' + 'audio' columns are range-read; a clip is kept when
               its key (clip_id with '/' -> '_') is a voice of rows_avs / rows_avs2; decoded exactly as colab/avs/pipeline_signed.py
               (ffmpeg -> 16 kHz mono s16, centred 8 s crop when longer) -> <out>/<src>/<clip_index>.wav
  vox        : VoxCeleb1 wavs range-read from the HF zips via caches/ext/vox1_manifest.csv (zip index, member offset, sizes)
               -> <out>/vox/<voice relpath>
Usage (library): avs_file(url, keys, out, ffmpeg) / vox1_member(urls, row, out)"""
import json, struct, subprocess, wave, zlib
from pathlib import Path
import numpy as np, requests


def pcm16k(data, ffmpeg, tmp):
    Path(tmp).write_bytes(data)
    r = subprocess.run([ffmpeg, "-v", "error", "-i", str(tmp), "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True)
    Path(tmp).unlink(); return np.frombuffer(r.stdout, np.int16)


def centred8(x):  # = pipeline_signed.process_file: clips under 1.5 s were never kept; longer than 8 s -> centred 8 s
    m = 8 * 16000
    if len(x) > m: s = (len(x) - m) // 2; x = x[s:s + m]
    return x


def write_wav(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.tobytes())


def avs_file(url, keys, out, ffmpeg):
    """keys: clip key -> list of (src, clip_index). Returns (clips written, bytes fetched)."""
    import pyarrow.parquet as pq
    from t3_httprange import RangeFile
    f = RangeFile(url); tab = pq.ParquetFile(f).read(columns=["clip_id", "audio"]).to_pylist(); n = 0
    for r in tab:
        k = r["clip_id"].replace("/", "_")
        if k not in keys or not r["audio"] or not r["audio"].get("bytes"): continue
        ext = Path(r["audio"].get("path") or "a.m4a").suffix or ".m4a"
        x = pcm16k(r["audio"]["bytes"], ffmpeg, Path(out) / f"tmp_{k}{ext}")
        if len(x) < 1.5 * 16000: continue
        x = centred8(x)
        for src, ci in keys[k]: write_wav(Path(out) / src / f"{ci}.wav", x); n += 1
    return n, f.fetched


def vox1_member(urls, row, out, session=None):
    """row: vox1_manifest row (zip, off, csize, size, ctype, voice). Range-reads the local header + data, inflates, writes the wav."""
    s = session or requests.Session(); url = urls[int(row.zip)]; a = int(row.off)
    h = s.get(url, headers={"Range": f"bytes={a}-{a + 29}"}, timeout=120).content; assert h[:4] == b"PK\x03\x04", "bad local header"
    n, m = struct.unpack("<HH", h[26:30]); d0 = a + 30 + n + m
    d = s.get(url, headers={"Range": f"bytes={d0}-{d0 + int(row.csize) - 1}"}, timeout=300).content; assert len(d) == int(row.csize)
    d = zlib.decompressobj(-15).decompress(d) if int(row.ctype) == 8 else d; assert len(d) == int(row.size)
    p = Path(out) / "vox" / row.voice; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(d)
    with wave.open(str(p)) as w: assert w.getframerate() == 16000 and w.getnchannels() == 1
    return len(d)


if __name__ == "__main__":  # self-check of the crop rule (no network)
    x = np.arange(10 * 16000, dtype=np.int16); y = centred8(x); assert len(y) == 8 * 16000 and y[0] == 16000
    assert len(centred8(x[:5 * 16000])) == 5 * 16000; print("t3_fetch_audio self-check ok")
