#!/bin/bash
# Đưa file lớn lên Colab: chia phần 40 MB (API contents từ chối file lớn), ghép lại trên VM.
# Dùng: SESS=... bash colab/upbig.sh <file local> <đường dẫn trên VM>
set -e
. "$(dirname "$0")/_env.sh"
SRC=$1; DST=$2
T=$(mktemp -d)
split -b 40M -d -a 3 "$SRC" $T/part.
for p in $T/part.*; do
  for i in 1 2 3; do
    $C upload -s $SESS $p /content/_p_$(basename $p) 2>&1 | tail -1 | grep -q Uploaded && break
    echo "   thử lại $(basename $p)"
  done
done
cat > $T/join.py <<PY
import subprocess
r = subprocess.run("cat /content/_p_part.* > '$DST' && rm -f /content/_p_part.* && ls -la '$DST'",
                   shell=True, capture_output=True, text=True)
print(r.stdout, r.stderr)
PY
$C exec -s $SESS -f $T/join.py --timeout 300 2>&1 | tail -2
rm -rf $T
