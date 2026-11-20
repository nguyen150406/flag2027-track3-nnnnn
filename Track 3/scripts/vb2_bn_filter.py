"""Strict Bengali exclusion for VoxBlink2 (a speaker is dropped if ANY signal fires):
 1. official Whisper language of any utterance in {bn, as} (Assamese shares the Bengali script)
 2. any official Whisper transcript containing Bengali-script characters (U+0980-U+09FF)
 3. YouTube metadata language (mirror voxblink2-metadata) starting with bn/as for any of the speaker's videos
 4. official speaker info (channel name, categories, tags) with Bengali script or Bengal/Bangladesh-related keywords
-> caches/vb2/exclude_bengali.csv (identity, reasons)."""
import re, tarfile, collections, glob
from pathlib import Path
import pandas as pd

D = Path(__file__).resolve().parents[1] / "caches/vb2"
BENG = re.compile("[ঀ-৿]")
KW = re.compile(r"bangla|bengali|bangladesh|kolkata|calcutta|dhaka|west\s*bengal|chittagong|chattogram|sylhet|rajshahi|khulna", re.I)
why = collections.defaultdict(set); n_utt = 0
with tarfile.open(D / "asr_res.tar.gz", "r|gz") as t:
    for m in t:
        if not m.isfile(): continue
        spk = m.name.split("/")[-1]
        for line in t.extractfile(m).read().decode(errors="replace").splitlines():
            p = line.split("\t")
            if len(p) < 3: continue
            n_utt += 1
            if p[1] in ("bn", "as"): why[spk].add(f"whisper_lang_{p[1]}")
            if BENG.search(p[2]): why[spk].add("bengali_script_transcript")
print("utterances scanned", n_utt, "| speakers flagged so far", len(why), flush=True)
for f in sorted(glob.glob(str(D / "m*.csv"))):
    m = pd.read_csv(f, dtype=str, usecols=["speaker_id", "language"])
    for s in m[m.language.fillna("").str.match(r"^(bn|as)\b")].speaker_id.unique(): why[s].add("youtube_lang")
with tarfile.open(D / "spk_info.tar.gz", "r:gz") as t:
    for m in t:
        if not m.isfile(): continue
        txt = t.extractfile(m).read().decode(errors="replace")
        if BENG.search(txt): why[m.name.split("/")[-1]].add("bengali_script_channel_or_tags")
        elif KW.search(txt): why[m.name.split("/")[-1]].add("bengal_keyword_channel_or_tags")
c = collections.Counter(r for v in why.values() for r in v)
print("speakers excluded:", len(why)); print("by signal:", dict(c))
pd.DataFrame([(s, ";".join(sorted(r))) for s, r in sorted(why.items())], columns=["identity", "reasons"]).to_csv(D / "exclude_bengali.csv", index=False)
