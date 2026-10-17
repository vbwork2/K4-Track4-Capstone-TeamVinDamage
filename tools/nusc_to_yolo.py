"""nuScenes mini → bộ YOLO (ảnh + nhãn 2D chiếu từ box 3D) để fine-tune trên Colab.

Chạy: python -I tools/nusc_to_yolo.py <nuScenes mini đủ 10 scene> data/finetune_nusc
Tên lớp giữ đúng tên COCO để detect.py lọc theo tên như cũ.
"""
import json, os, shutil, sys
from collections import Counter

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points

SPLITS = {  # test = 3 scene dùng cho M2/M4: chỉ để đo, không bao giờ train
    "train": ["scene-0103", "scene-0655", "scene-0757", "scene-0916", "scene-1077", "scene-1100"],
    "val": ["scene-0061"],
    "test": ["scene-0553", "scene-0796", "scene-1094"],
}
# Giữ đúng id COCO: so được với yolo11n gốc trên cùng tập test, fine-tune không phải khởi tạo lại đầu ra
COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella "
        "handbag tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
        "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange "
        "broccoli carrot hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop "
        "mouse remote keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors "
        "teddy_bear hair_drier toothbrush").split()
COCO = [n.replace("_", " ") for n in COCO]
assert len(COCO) == 80
CAMS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
W, H = 1600, 900
MIN_IN = 0.5   # giữ nhãn có >= 50% box chiếu nằm trong ảnh (giống luật M2)
MIN_H = 8      # px


def to_cls(cat):
    if cat.startswith("human.pedestrian."): return "person"
    if cat == "vehicle.bicycle": return "bicycle"
    if cat == "vehicle.motorcycle": return "motorcycle"   # box nuScenes đã gồm người lái
    if cat in ("vehicle.car", "vehicle.emergency.police"): return "car"
    if cat.startswith("vehicle.bus."): return "bus"
    if cat == "vehicle.truck": return "truck"
    return None  # trailer, xe công trình, xe cứu thương, vật cản: bỏ


def box2d(box, K):
    c = box.corners()
    if (c[2] <= 0.1).any():
        return None  # có góc nằm sau camera: chiếu sai, bỏ
    p = view_points(c, K, normalize=True)[:2]
    x0, y0, x1, y1 = p[0].min(), p[1].min(), p[0].max(), p[1].max()
    a = (x1 - x0) * (y1 - y0)
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if cx1 <= cx0 or cy1 <= cy0 or a <= 0: return None
    if (cx1 - cx0) * (cy1 - cy0) / a < MIN_IN or cy1 - cy0 < MIN_H: return None
    return cx0, cy0, cx1, cy1


def main(src, dst):
    nusc = NuScenes("v1.0-mini", dataroot=src, verbose=False)
    by_name = {s["name"]: s for s in nusc.scene}
    stats = {}
    for split, scenes in SPLITS.items():
        img_dir, lbl_dir = os.path.join(dst, "images", split), os.path.join(dst, "labels", split)
        os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
        cnt, n_img = Counter(), 0
        for name in scenes:
            tok = by_name[name]["first_sample_token"]
            while tok:
                s = nusc.get("sample", tok)
                for cam in CAMS:
                    path, boxes, K = nusc.get_sample_data(s["data"][cam], box_vis_level=BoxVisibility.ANY)
                    lines = []
                    for b in boxes:
                        ann = nusc.get("sample_annotation", b.token)
                        cls = to_cls(ann["category_name"])
                        if cls is None or int(ann["visibility_token"]) < 2: continue
                        r = box2d(b, K)
                        if r is None: continue
                        x0, y0, x1, y1 = r
                        lines.append(f"{COCO.index(cls)} {(x0+x1)/2/W:.6f} {(y0+y1)/2/H:.6f} "
                                     f"{(x1-x0)/W:.6f} {(y1-y0)/H:.6f}")
                        cnt[cls] += 1
                    stem = f"{name}__{cam}__{s['timestamp']}"
                    shutil.copy2(path, os.path.join(img_dir, stem + ".jpg"))
                    open(os.path.join(lbl_dir, stem + ".txt"), "w").write("\n".join(lines))
                    n_img += 1
                tok = s["next"]
        stats[split] = {"scenes": scenes, "images": n_img, "boxes": dict(cnt)}
        print(split, n_img, dict(cnt))
    with open(os.path.join(dst, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# Đường dẫn trên Colab; local thì sửa path\n"
                "path: /content/finetune_nusc\ntrain: images/train\nval: images/val\ntest: images/test\n"
                "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(COCO)))
    json.dump(stats, open(os.path.join(dst, "stats.json"), "w"), indent=1)
    shutil.copy2(os.path.join(src, "LICENSE"), os.path.join(dst, "LICENSE"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
