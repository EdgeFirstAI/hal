#!/usr/bin/env bash
# Size baseline for the packaging split (A3: self-contained wheels).
#
# Under A3 the meaningful unit is the shipped extension module, not the crate:
# each module statically links only the slice of its dependencies it calls, so
# per-crate rlib sizes do not predict wheel size. This measures artifacts.
#
# Usage: scripts/size_baseline.sh [output.json]
set -euo pipefail
OUT="${1:-docs/baselines/$(date +%Y-%m-%d)-baseline.json}"
mkdir -p "$(dirname "$OUT")"

py() { command -v python3 >/dev/null && python3 "$@"; }

deflate() {  # compressed size, i.e. what a user downloads
    python3 -c "import sys,zlib;print(len(zlib.compress(open(sys.argv[1],'rb').read(),9)))" "$1"
}

printf '{\n  "artifacts": {\n' > "$OUT"
first=1
for f in \
    target/release/libedgefirst_tensor.so target/release/libedgefirst_tensor.dylib \
    target/release/libedgefirst_codec.so target/release/libedgefirst_codec.dylib \
    target/release/libedgefirst_image.so target/release/libedgefirst_image.dylib \
    target/release/libedgefirst_decoder.so target/release/libedgefirst_decoder.dylib \
    target/release/libedgefirst_tracker.so target/release/libedgefirst_tracker.dylib
do
    [ -f "$f" ] || continue
    [ $first -eq 1 ] || printf ',\n' >> "$OUT"
    first=0
    printf '    "%s": {"raw": %s, "deflate": %s}' \
        "$(basename "$f")" \
        "$(wc -c < "$f" | tr -d ' ')" "$(deflate "$f")" >> "$OUT"
done
printf '\n  },\n  "wheels": {\n' >> "$OUT"
first=1
for w in target/wheels/*.whl; do
    [ -e "$w" ] || continue
    [ $first -eq 1 ] || printf ',\n' >> "$OUT"
    first=0
    printf '    "%s": {"on_disk": %s, "uncompressed_so": %s}' \
        "$(basename "$w")" "$(wc -c < "$w" | tr -d ' ')" \
        "$(python3 -c "
import sys,zipfile
z=zipfile.ZipFile(sys.argv[1])
print(sum(i.file_size for i in z.infolist() if i.filename.endswith(('.so','.dylib','.pyd'))) or 0)" "$w")" >> "$OUT"
done
printf '\n  },\n' >> "$OUT"
printf '  "commit": "%s",\n' "$(git rev-parse HEAD)" >> "$OUT"
printf '  "rustc": "%s"\n}\n' "$(rustc --version)" >> "$OUT"
echo "✓ baseline written to $OUT"
