#!/bin/bash
# Tạo session GPU, đưa code lên /content, cài thư viện. Không đụng key.
# Tuỳ chọn: WITH_DATA=1 (nuScenes mini, 6 CAM_*), WITH_PHENIKAA=1 (cần chủ dữ liệu đồng ý), WITH_FT=1 (finetune_nusc).
# Hai session: SESS=hack-a100 GPU=A100 bash colab/up.sh ; SESS=hack-g4 GPU=G4 bash colab/up.sh
set -e
. "$(dirname "$0")/_env.sh"
GPU=${GPU:-G4}
NUSC=${NUSC:-$H/data/mini-nuScenes}
PH=$H/data/phenikaa
CAMS="CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT"
T=$(mktemp -d)

echo "== 1/4 Tạo session $SESS ($GPU)"
$C new -s $SESS --gpu $GPU 2>&1 | tail -3

echo "== 2/4 Đóng gói và đưa code lên"
REQ=requirements-be.txt; [ -f $H/requirements.txt ] && REQ=requirements.txt
tar czf $T/code.tgz -C $H --exclude=__pycache__ src tools config $REQ $( [ -d $H/calib ] && echo calib ) $( [ -d $H/colab/jobs ] && echo colab/jobs )
$C upload -s $SESS $T/code.tgz /content/code.tgz 2>&1 | tail -1
if [ "${WITH_DATA:-0}" = "1" ]; then
  echo "   + nuScenes mini (6 CAM_* + v1.0-mini)"
  tar czf $T/nusc.tgz -C $NUSC $(for c in $CAMS; do echo samples/$c; done) v1.0-mini LICENSE
  bash $H/colab/upbig.sh $T/nusc.tgz /content/nusc.tgz
fi
if [ "${WITH_PHENIKAA:-0}" = "1" ]; then
  echo "   + Phenikaa (6 CAM_P_* + label + ego_pose + Calib Information)"
  ( cd "$PH" && tar cf $T/phenikaa.tar \
      "COMPLEX CITY/camera" "COMPLEX CITY/label" "COMPLEX CITY/ego_pose" \
      "COMPLEX URBAN/camera" "COMPLEX URBAN/label" "COMPLEX URBAN/ego_pose" \
      "Calib Information" --exclude='*(*' )
  bash $H/colab/upbig.sh $T/phenikaa.tar /content/phenikaa.tar
fi
if [ "${WITH_FT:-0}" = "1" ]; then
  echo "   + finetune_nusc"
  tar cf $T/ft.tar -C $H/data finetune_nusc
  bash $H/colab/upbig.sh $T/ft.tar /content/ft.tar
fi

echo "== 3/4 Giải nén"
cat > $T/unpack.py <<'PY'
import subprocess
cmd = ("mkdir -p /content/hack /content/nusc && tar xzf /content/code.tgz -C /content/hack && "
       "( [ -f /content/nusc.tgz ] && tar xzf /content/nusc.tgz -C /content/nusc || true ) && "
       "( [ -f /content/phenikaa.tar ] && mkdir -p /content/phenikaa && tar xf /content/phenikaa.tar -C /content/phenikaa || true ) && "
       "( [ -f /content/ft.tar ] && tar xf /content/ft.tar -C /content || true ) && "
       "ls /content/hack /content/nusc /content/phenikaa && nvidia-smi --query-gpu=name,memory.total --format=csv")
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-1000:])
PY
$C exec -s $SESS -f $T/unpack.py --timeout 300 2>&1 | tail -12

echo "== 4/4 Cài thư viện (nền, log /content/setup.log)"
cat > $T/setup.sh <<X
#!/bin/bash
cd /content/hack
pip install -q ultralytics; pip install -q --no-deps nuscenes-devkit; pip install -q pyquaternion descartes cachetools shapely tqdm
python -c "import ultralytics, torch; print('ultralytics', ultralytics.__version__, 'cuda', torch.cuda.is_available())"
echo SETUP_DONE
X
$C upload -s $SESS $T/setup.sh /content/setup.sh 2>&1 | tail -1
$C exec -s $SESS -f $H/colab/bg.py --env BG_SCRIPT=/content/setup.sh --env BG_LOG=/content/setup.log 2>&1 | tail -1
rm -rf $T
echo "Xong. Xem cài đặt: LOG=/content/setup.log bash colab/log.sh (chờ SETUP_DONE), rồi chạy colab/run.sh."
