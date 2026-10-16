"""Load defaults and dataset overrides without modifying shared configuration."""

import json
import math
from copy import deepcopy
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def load_config(dataset, path=None):
    with (CONFIG_DIR / "default.json").open(encoding="utf-8") as stream:
        result = json.load(stream)
    dataset_path = CONFIG_DIR / f"{dataset}.json"
    if dataset_path.is_file():
        with dataset_path.open(encoding="utf-8") as stream:
            _merge(result, json.load(stream))
    if path:
        with Path(path).open(encoding="utf-8") as stream:
            _merge(result, json.load(stream))
    _validate(result)
    return result


def _validate(config):
    """Fail before opening sources when configuration thresholds are invalid."""
    positive = {
        "ego_box": ("length", "width"),
        "fusion": ("distance_m", "disagreement_m", "association_m"),
        "tracker": ("gate_m", "max_age_s", "trail_s", "initial_velocity_var"),
        "zone_cfg": ("side_margin_m", "longitudinal_margin_m", "lane_margin_m", "cut_in_range_m",
                     "cut_in_time_s", "ttc_warning_s", "ttc_danger_s", "max_range_m"),
    }
    for section, keys in positive.items():
        for key in keys:
            value = float(config[section][key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{section}.{key} must be finite and positive")
    detector = config["detection"]
    if not 0 <= detector["min_conf"] <= detector["uncertain_conf"] <= 1:
        raise ValueError("Detection thresholds must satisfy 0 <= min_conf <= uncertain_conf <= 1")
    if not 0 <= detector["rider_iou"] <= 1:
        raise ValueError("detection.rider_iou must be between 0 and 1")
    zones = config["zone_cfg"]
    if zones["ttc_danger_s"] >= zones["ttc_warning_s"]:
        raise ValueError("TTC danger threshold must be below warning threshold")
    if any(zones[key] not in (1, 2) for key in ("side_level", "cut_in_level")):
        raise ValueError("Side and cut-in warning levels must be 1 or 2")
    fusion = config["fusion"]
    if not fusion["distance_m"] <= fusion["disagreement_m"] <= fusion["association_m"]:
        raise ValueError("Fusion thresholds must satisfy distance <= disagreement <= association")
    if not math.isfinite(config["ego_box"]["x_offset"]):
        raise ValueError("ego_box.x_offset must be finite")
    if not math.isfinite(config["tracker"]["process_var"]) or config["tracker"]["process_var"] < 0:
        raise ValueError("tracker.process_var must be finite and nonnegative")
    for cls, height in config["height_m"].items():
        if not math.isfinite(height) or height <= 0:
            raise ValueError(f"height_m.{cls} must be finite and positive")
