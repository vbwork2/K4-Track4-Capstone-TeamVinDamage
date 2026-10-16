"""Merge rider boxes and nearby cross-camera observations."""

from dataclasses import replace
from math import hypot

from .contracts import Observation

COCO_CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
ALLOWED = set(COCO_CLASSES.values()) | {"rider"}


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, right-left) * max(0, bottom-top)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def merge_riders(boxes, min_conf=0.4, rider_iou=0.3):
    """Pair each person with at most one cycle; preserve the cycle ground point."""
    import math
    valid = []
    for original in boxes:
        item = dict(original)
        item["cls"] = COCO_CLASSES.get(item["cls"], item["cls"])
        conf = float(item["conf"])
        xyxy = list(map(float, item["xyxy"]))
        if item["cls"] not in ALLOWED or not math.isfinite(conf) or not min_conf <= conf <= 1:
            continue
        if len(xyxy) != 4 or not all(map(math.isfinite, xyxy)) or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            continue
        item.update(conf=conf, xyxy=xyxy)
        valid.append(item)
    candidates = []
    for pi, person in enumerate(valid):
        if person["cls"] != "person":
            continue
        p = person["xyxy"]
        px, py = (p[0]+p[2])/2, p[3]
        for bi, bike in enumerate(valid):
            if bike["cls"] not in ("motorcycle", "bicycle") or person.get("cam") != bike.get("cam"):
                continue
            b = bike["xyxy"]
            overlap = iou(p, b)
            inside = b[0] <= px <= b[2] and b[1] <= py <= b[3]
            if inside or overlap > rider_iou:
                candidates.append((overlap + int(inside), pi, bi))
    used = set()
    result = []
    for _, pi, bi in sorted(candidates, reverse=True):
        if pi in used or bi in used:
            continue
        person, bike = valid[pi], valid[bi]
        p, b = person["xyxy"], bike["xyxy"]
        rider = dict(bike)
        rider.update(cls="rider", conf=min(person["conf"], bike["conf"]), ground_xyxy=b[:],
                     xyxy=[min(p[0], b[0]), min(p[1], b[1]), max(p[2], b[2]), max(p[3], b[3])])
        result.append(rider)
        used.update((pi, bi))
    result.extend(item for index, item in enumerate(valid) if index not in used)
    return result


def canonical_class(cls):
    return "rider" if cls in ("motorcycle", "bicycle", "rider") else "car" if cls in ("bus", "truck", "car") else cls


def fuse_observations(observations, distance_m=1.0, disagreement_m=1.5, association_m=3.0):
    """Use complete-link clusters with at most one observation per camera.

    A mutual nearest pair outside the fusion gate is only a disagreement
    candidate. Keep both measurements and flag them instead of merging them.
    """
    obs = [replace(o, cls=canonical_class(o.cls), dets=[dict(d) for d in o.dets],
                   reasons=list(o.reasons)) for o in observations]
    def cameras(o):
        return {d["cam"] for d in o.dets}
    def distance(a, b):
        return hypot(a.x-b.x, a.y-b.y)
    for i, a in enumerate(obs):
        candidates = [(distance(a, b), j) for j, b in enumerate(obs)
                      if i != j and a.cls == b.cls and cameras(a).isdisjoint(cameras(b))]
        if not candidates:
            continue
        dist, j = min(candidates)
        b = obs[j]
        reverse = [(distance(b, c), k) for k, c in enumerate(obs)
                   if j != k and b.cls == c.cls and cameras(b).isdisjoint(cameras(c))]
        if min(reverse)[1] == i and disagreement_m < dist <= association_m:
            for item in (a, b):
                item.uncertain = True
                if "camera_disagreement" not in item.reasons:
                    item.reasons.append("camera_disagreement")
    clusters = []
    for item in sorted(obs, key=lambda o: -o.conf):
        eligible = [(max(distance(item, member) for member in group), index)
                    for index, group in enumerate(clusters)
                    if all(item.cls == member.cls and cameras(item).isdisjoint(cameras(member))
                           and distance(item, member) < distance_m for member in group)]
        if eligible:
            clusters[min(eligible)[1]].append(item)
        else:
            clusters.append([item])
    result = []
    for group in clusters:
        result.append(Observation(group[0].cls, max(o.conf for o in group),
                                  sum(o.x for o in group)/len(group), sum(o.y for o in group)/len(group),
                                  [d for o in group for d in o.dets], any(o.uncertain for o in group),
                                  sorted({reason for o in group for reason in o.reasons})))
    return result
