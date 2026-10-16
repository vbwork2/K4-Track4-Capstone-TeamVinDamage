"""Pure warning rules shared by the application and Core's M4 evaluator."""

import math

ZONES = ("SAT_SUON", "TAT_DAU", "PHIA_TRUOC")


def in_side_zone(x, y, ego_box, zone_cfg):
    front = ego_box["x_offset"] + ego_box["length"]/2
    rear = ego_box["x_offset"] - ego_box["length"]/2
    half_width = ego_box["width"]/2
    return (rear-zone_cfg["longitudinal_margin_m"] <= x <= front+zone_cfg["longitudinal_margin_m"]
            and half_width <= abs(y) <= half_width+zone_cfg["side_margin_m"])


def evaluate_track(track, ego_box, zone_cfg, source="phenikaa"):
    """Return every active zone plus a primary zone for the FE contract.

    Primary priority is severity, then cut-in, side, front. Side and cut-in
    default to yellow because their red thresholds are not specified.
    """
    result = dict(track)
    x, y, vx, vy = (float(track[key]) for key in ("x", "y", "vx", "vy"))
    reasons = list(track.get("reasons", []))
    if not all(math.isfinite(v) for v in (x, y, vx, vy)):
        result.update(zone=None, zones=[], level=0, ttc=None, uncertain=True,
                      reasons=sorted(set(reasons + ["invalid_state"])))
        return result
    uncertain = bool(track.get("uncertain", False))
    if math.hypot(x, y) > zone_cfg["max_range_m"]:
        uncertain = True
        reasons.append("out_of_range")
    lane_edge = ego_box["width"]/2 + zone_cfg["lane_margin_m"]
    velocity_valid = track.get("velocity_valid", True)
    candidates = []
    ttc = None
    if source != "webcam" and in_side_zone(x, y, ego_box, zone_cfg):
        candidates.append((int(zone_cfg.get("side_level", 1)), 2, "SAT_SUON"))
    if source != "webcam" and velocity_valid and 0 < x <= zone_cfg["cut_in_range_m"]:
        if abs(y) > lane_edge and y*vy < 0:
            entry_time = (abs(y)-lane_edge) / abs(vy)
            if entry_time < zone_cfg["cut_in_time_s"]:
                candidates.append((int(zone_cfg.get("cut_in_level", 1)), 3, "TAT_DAU"))
    # A non-closing object has no finite TTC, including at the ego origin.
    if velocity_valid and x > 0 and abs(y) <= lane_edge and vx < -1e-6:
        ttc = x / max(-vx, 1e-6)
        if ttc < zone_cfg["ttc_warning_s"]:
            level = 2 if ttc < zone_cfg["ttc_danger_s"] else 1
            candidates.append((level, 1, "PHIA_TRUOC"))
    level, _, primary = max(candidates) if candidates else (0, 0, None)
    result.update(zone=primary, zones=[c[2] for c in candidates],
                  level=min(level, 1) if uncertain else level, ttc=ttc,
                  uncertain=uncertain, reasons=sorted(set(reasons)))
    return result


def evaluate_tracks(tracks, ego_box, zone_cfg, source="phenikaa"):
    return [evaluate_track(track, ego_box, zone_cfg, source) for track in tracks]


def alert_counts(tracks):
    return {zone: sum(track["level"] > 0 and zone in track.get("zones", [track["zone"]])
                      for track in tracks) for zone in ZONES}
