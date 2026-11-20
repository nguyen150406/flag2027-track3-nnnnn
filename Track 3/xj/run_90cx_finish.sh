#!/bin/bash
# Unattended finish of 90c-X (15/11, user: "khi xong ... cho script chạy tạo submission luôn một lượt"; prereg PREREG_gate_all.txt 22:12).
# 1) wait for the A+B chain (FLAG_2027_90cx, log x_chain_ab.log) -> soft gate -> HARD checks -> sub103a = sub107a English + 90c-X(A+B) Bangla
# 2) clone FLAG_2027_90cxc from the finished A+B clone -> option C chain -> soft gate -> HARD checks -> sub103b = sub107a English + 90c-X(A+B+C) Bangla
# HARD fail -> that ticket is not built. Everything is logged to Track 3/outputs/ultra/SUMMARY_90cx.txt. CPU only, <= 2 heavy jobs at a time.
#   caffeinate -i bash "$HOME/Documents/FLAG_2027/Track 3/xj/run_90cx_finish.sh"
set -uo pipefail
MAIN="$HOME/Documents/FLAG_2027/Track 3"; AB="$HOME/Documents/FLAG_2027_90cx"; C="$HOME/Documents/FLAG_2027_90cxc"
SUMM="$MAIN/outputs/ultra/SUMMARY_90cx.txt"; PRE="$MAIN/outputs/ultra/PREREG_gate_all.txt"
say() { echo "$(date '+%d/%m %H:%M:%S') $*" | tee -a "$SUMM"; }
build_ticket() {   # $1 clone root, $2 build name (sub90cx / sub90cxc), $3 ticket (sub103a / sub103b), $4 extra check flag
  local root="$1" b="$2" t="$3" flag="${4:-}"
  say "[$t] soft gate (descriptive) on $b"
  (cd "$root/Track 3/scripts" && T3_EXCLUDE="$root/Track 3/caches/ext/exclude_ultra.csv" python t3_90cx_gate.py "$b") >> "$SUMM" 2>&1 || say "[$t] gate script error (see above)"
  say "[$t] HARD checks"
  if (cd "$MAIN/xj" && python check_90cx.py "$b" "$root/Track 3" $flag) >> "$SUMM" 2>&1; then
    [[ -e "$MAIN/submissions/t3_$b" ]] || cp -R "$root/Track 3/submissions/t3_$b" "$MAIN/submissions/t3_$b"
    if (cd "$MAIN/xj" && python fuse_h.py "$t" sub107a "$b") >> "$SUMM" 2>&1; then
      if python - "$MAIN" "$t" "$b" >> "$SUMM" 2>&1 <<'EOF'
import sys, zipfile, hashlib
M, t, b = sys.argv[1:]; S = f"{M}/submissions"
z, e, n = (zipfile.ZipFile(f"{S}/t3_{s}/{s}.zip") for s in (t, "sub107a", b))
for f in sorted(z.namelist()):
    src = e if "English" in f else n; same = z.read(f) == src.read(f)
    print(f"  {f:45s} == {'sub107a' if src is e else b}: {same}"); assert same
print(f"  {t}.zip sha256 {hashlib.sha256(open(f'{S}/t3_{t}/{t}.zip','rb').read()).hexdigest()[:16]}")
EOF
      then
        cp -n "$MAIN/submissions/t3_$t/$t.zip" "$HOME/Downloads/$t.zip"; say "[$t] BUILT -> submissions/t3_$t/$t.zip (copy ~/Downloads/$t.zip)"
        echo "$(date '+%d/%m %H:%M') BUILD $t (unattended, xj/run_90cx_finish.sh): English = sub107a bytes, Bangla = $b; see SUMMARY_90cx.txt for gate/health." >> "$PRE"
      else say "[$t] byte check vs sources FAILED -> do not submit $t"; fi
    else say "[$t] fuse_h FAILED"; fi
  else say "[$t] HARD CHECK FAILED -> $t NOT built"; echo "$(date '+%d/%m %H:%M') $t NOT built: HARD check failed (SUMMARY_90cx.txt)." >> "$PRE"; fi
}

say "=== run_90cx_finish start: waiting for the A+B chain"
while pgrep -f "t3_90cx_chain.sh" > /dev/null; do sleep 60; done
if ! grep -q "CHAIN DONE" "$AB/Track 3/x_chain_ab.log"; then say "A+B chain ended WITHOUT 'CHAIN DONE' -> stop"; tail -30 "$AB/Track 3/x_chain_ab.log" >> "$SUMM"; exit 1; fi
say "A+B chain done"; tail -12 "$AB/Track 3/x_chain_ab.log" >> "$SUMM"
build_ticket "$AB" sub90cx sub103a ""

say "=== option C: clone + chain"
if [[ ! -e "$C" ]]; then cp -cR "$AB" "$C" || { say "clone FAILED -> stop"; exit 1; }; fi
(cd "$C/Track 3/scripts" && bash t3_90cx_chain_c.sh) > "$C/Track 3/c_chain.log" 2>&1
if ! grep -q "CHAIN C DONE" "$C/Track 3/c_chain.log"; then say "C chain ended WITHOUT 'CHAIN C DONE' -> sub103b not built"; tail -30 "$C/Track 3/c_chain.log" >> "$SUMM"; exit 1; fi
say "C chain done"; tail -12 "$C/Track 3/c_chain.log" >> "$SUMM"
build_ticket "$C" sub90cxc sub103b "--lgrl"
say "=== run_90cx_finish END"
