"""Cắt nuScenes mini còn vài scene: lọc bảng JSON + chép file cảm biến."""
import json, os, shutil, sys

src, dst = sys.argv[1], sys.argv[2]
scenes = sys.argv[3].split(",")
meta = "v1.0-mini"

def load(t):
    return json.load(open(os.path.join(src, meta, t + ".json"), encoding="utf-8"))

T = {t: load(t) for t in ["scene", "sample", "sample_data", "sample_annotation", "instance",
                          "ego_pose", "log", "map", "calibrated_sensor", "sensor",
                          "category", "attribute", "visibility"]}

sc = [s for s in T["scene"] if s["name"] in scenes]
assert len(sc) == len(scenes), "thiếu scene"
sc_tok = {s["token"] for s in sc}
log_tok = {s["log_token"] for s in sc}
smp = [s for s in T["sample"] if s["scene_token"] in sc_tok]
smp_tok = {s["token"] for s in smp}
sensor_ch = {s["token"]: s["channel"] for s in T["sensor"]}
cs_ch = {c["token"]: sensor_ch[c["sensor_token"]] for c in T["calibrated_sensor"]}

def keep_sd(sd):
    if sd["sample_token"] not in smp_tok:
        return False
    ch = cs_ch[sd["calibrated_sensor_token"]]
    # camera: chỉ keyframe; LiDAR + radar: cả sweep
    return sd["is_key_frame"] if ch.startswith("CAM_") else True

sd = [d for d in T["sample_data"] if keep_sd(d)]
sd_tok = {d["token"] for d in sd}
for d in sd:  # cắt liên kết prev/next trỏ tới sweep camera đã bỏ
    if d["prev"] and d["prev"] not in sd_tok: d["prev"] = ""
    if d["next"] and d["next"] not in sd_tok: d["next"] = ""
# nối lại chuỗi keyframe camera để prev/next vẫn đi được
by_ch = {}
for d in sd:
    by_ch.setdefault((cs_ch[d["calibrated_sensor_token"]], d["sample_token"] and
                      next(s["scene_token"] for s in smp if s["token"] == d["sample_token"])), []).append(d)
for (ch, _), lst in by_ch.items():
    if not ch.startswith("CAM_"): continue
    lst.sort(key=lambda d: d["timestamp"])
    for a, b in zip(lst, lst[1:]):
        a["next"], b["prev"] = b["token"], a["token"]

ann = [a for a in T["sample_annotation"] if a["sample_token"] in smp_tok]
inst_tok = {a["instance_token"] for a in ann}
ego_tok = {d["ego_pose_token"] for d in sd}
out = dict(T)
out.update(scene=sc, sample=smp, sample_data=sd, sample_annotation=ann,
           instance=[i for i in T["instance"] if i["token"] in inst_tok],
           ego_pose=[e for e in T["ego_pose"] if e["token"] in ego_tok],
           log=[l for l in T["log"] if l["token"] in log_tok])
out["map"] = [dict(m, log_tokens=[t for t in m["log_tokens"] if t in log_tok])
              for m in T["map"] if set(m["log_tokens"]) & log_tok]

os.makedirs(os.path.join(dst, meta), exist_ok=True)
for t, rows in out.items():
    json.dump(rows, open(os.path.join(dst, meta, t + ".json"), "w", encoding="utf-8"), indent=0)

n = 0
for d in sd:
    s, o = os.path.join(src, d["filename"]), os.path.join(dst, d["filename"])
    os.makedirs(os.path.dirname(o), exist_ok=True)
    shutil.copy2(s, o); n += 1
maps = {m["filename"] for m in out["map"]}
for f in maps:
    os.makedirs(os.path.dirname(os.path.join(dst, f)), exist_ok=True)
    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
for f in ["LICENSE", ".v1.0-mini.txt"]:
    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
print(f"scenes={len(sc)} samples={len(smp)} sample_data={len(sd)} anns={len(ann)} files={n} maps={len(maps)}")
