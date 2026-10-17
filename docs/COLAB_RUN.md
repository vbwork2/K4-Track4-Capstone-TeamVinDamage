# Chay Colab: 2 may song song

Chay tu PowerShell. `$R` = duong dan repo. Moi lenh WSL can `WSLENV` de truyen bien.
**Chi dua Phenikaa len sau khi chu du lieu dong y** (neu chua: bo `WITH_PHENIKAA`, chi chay nuScenes).

```powershell
$R="/mnt/c/Users/Khanh/Documents/AInAction/K4-Track4-Capstone-TeamVinDamage"
$env:WSLENV="SESS/u:GPU/u:WITH_DATA/u:WITH_PHENIKAA/u:WITH_FT/u:JOB/u:LOG/u:N/u"
```

## 1. Tao 2 session

```powershell
# G4: cache yolo11m (demo chinh) + M3 -> chay truoc
$env:SESS="hack-g4"; $env:GPU="G4"; $env:WITH_DATA="1"; $env:WITH_PHENIKAA="1"; $env:WITH_FT=""
wsl -d Ubuntu -e bash $R/colab/up.sh
# A100 high-RAM: fine-tune yolo11m, cache yolo11m_ft, roi cache yolo11n (so sanh)
$env:SESS="hack-a100"; $env:GPU="A100"; $env:WITH_DATA="1"; $env:WITH_PHENIKAA="1"; $env:WITH_FT="1"
wsl -d Ubuntu -e bash $R/colab/up.sh
```

Cho `SETUP_DONE` o moi may: `$env:SESS="hack-g4"; $env:LOG="/content/setup.log"; wsl -d Ubuntu -e bash $R/colab/log.sh`

## 2. Chay job (nen)

```powershell
# G4: yolo11m (uu tien 1). Tu ghi M3 vao results/m3_colab_*.json
$env:SESS="hack-g4"; $env:LOG="/content/hack_job.log"
$env:JOB="MODEL=yolo11m.pt bash colab/jobs/cache_all.sh"
wsl -d Ubuntu -e bash $R/colab/run.sh

# A100: fine-tune truoc, roi cache yolo11n (neu thieu gio thi bo yolo11n)
$env:SESS="hack-a100"
$env:JOB="python tools/ft_yolo.py --data /content/finetune_nusc/data.yaml --model yolo11m.pt --batch 8 --out results/ ; MODEL=results/yolo11m_ft.pt bash colab/jobs/cache_all.sh ; MODEL=yolo11n.pt bash colab/jobs/cache_all.sh"
wsl -d Ubuntu -e bash $R/colab/run.sh
```

Xem log: `$env:SESS="hack-g4"; $env:LOG="/content/hack_job.log"; $env:N="30"; wsl -d Ubuntu -e bash $R/colab/log.sh`
(xong khi thay `CACHE_ALL_DONE`).

## 3. Tai ve

`dl.sh` tai `results/*.json` ve `results/colab/`. Cache nam trong tar:

```powershell
$env:SESS="hack-g4";   wsl -d Ubuntu -e bash $R/colab/dl.sh
wsl -d Ubuntu -e bash -c "export HOME=/home/khanh/colab9; /home/khanh/.local/bin/colab download -s hack-g4 /content/out_yolo11m.tar $R/out_yolo11m.tar"
$env:SESS="hack-a100"; wsl -d Ubuntu -e bash $R/colab/dl.sh
wsl -d Ubuntu -e bash -c "export HOME=/home/khanh/colab9; /home/khanh/.local/bin/colab download -s hack-a100 /content/out_yolo11n.tar $R/out_yolo11n.tar"
```

Giai nen tai repo (tao `cache/<dataset>/<scene>/<model>.jsonl` va `results/m3_colab_*.json`):

```powershell
tar xf out_yolo11m.tar; tar xf out_yolo11n.tar
```

Kiem tra: moi scene Phenikaa 200 khung x 6 cam = 1200 dong; kiem `results/m3_colab_*.json`.

## 4. Tat

```powershell
$env:SESS="hack-g4";   wsl -d Ubuntu -e bash $R/colab/stop.sh
$env:SESS="hack-a100"; wsl -d Ubuntu -e bash $R/colab/stop.sh
```

Luu y: `src/detect.py:detect` (Live, RTX 4050) mac dinh yolo11n; muon dung 11m cho Live thi tao `Detector("yolo11m.pt")`.
Weights fine-tune: `results/<model>_ft.pt`; cache bang no: `MODEL=results/yolo11n_ft.pt bash colab/jobs/cache_all.sh` (ten file cache = `yolo11n_ft.jsonl`).
