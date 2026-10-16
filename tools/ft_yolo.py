"""Fine-tune nhẹ YOLO trên Colab (đóng băng backbone, chỉ train neck + head), đo trước/sau trên cùng tập test.

Chạy trên Colab:
  python tools/ft_yolo.py --data /content/finetune_nusc/data.yaml --model yolo11n.pt --out results/
Ra: results/ft_<model>.json (mAP gốc vs fine-tune, theo lớp) và results/<model>_ft.pt.
"""
import argparse, json, os, shutil, time

from ultralytics import YOLO

CLS = [0, 1, 2, 3, 5, 7]  # person bicycle car motorcycle bus truck (id COCO)


def score(model, a, split):
    m = model.val(data=a.data, split=split, imgsz=a.imgsz, batch=a.batch, classes=CLS,
                  conf=0.001, plots=False, verbose=False)
    names = model.names
    return {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
            "per_class_AP50": {names[int(c)]: round(float(v), 4)
                               for c, v in zip(m.box.ap_class_index, m.box.ap50)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--freeze", type=int, default=10, help="số layer đầu bị đóng băng; yolo11: 0-9 là backbone")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--out", default="results/")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.model))[0]

    res = {"model": a.model, "freeze": a.freeze, "epochs": a.epochs, "imgsz": a.imgsz}
    res["coco_test"] = score(YOLO(a.model), a, "test")
    print("COCO gốc, test:", res["coco_test"])

    t0 = time.time()
    m = YOLO(a.model)
    m.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, freeze=a.freeze,
            optimizer="AdamW", lr0=0.001, cos_lr=True, close_mosaic=5, patience=10,
            project="runs_ft", name=tag, exist_ok=True, plots=True)
    res["train_min"] = round((time.time() - t0) / 60, 1)

    best = os.path.join("runs_ft", tag, "weights", "best.pt")
    res["ft_test"] = score(YOLO(best), a, "test")
    print("Fine-tune, test:", res["ft_test"])
    shutil.copy2(best, os.path.join(a.out, f"{tag}_ft.pt"))
    json.dump(res, open(os.path.join(a.out, f"ft_{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
