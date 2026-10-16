#!/bin/bash
# Chạy trên Colab: cache 2 scene Phenikaa + 3 scene nuScenes, rồi tar kết quả.
# MODEL=yolo11m.pt (mặc định) | yolo11n.pt | đường dẫn weights fine-tune.
MODEL=${MODEL:-yolo11m.pt}
cd /content/hack && mkdir -p cache results
S=$(basename "$MODEL" .pt)
for sc in "COMPLEX CITY" "COMPLEX URBAN"; do
  [ -d "/content/phenikaa/$sc" ] || { echo "skip Phenikaa $sc (chua upload)"; continue; }
  python -m src.detect_cache --dataset phenikaa --root /content/phenikaa --scene "$sc" --model "$MODEL" --out cache
done
for sc in scene-0553 scene-0796 scene-1094; do
  python -m src.detect_cache --dataset nuscenes --root /content/nusc --scene "$sc" --model "$MODEL" --out cache
done
tar cf /content/out_$S.tar cache results
echo "CACHE_ALL_DONE $S"
