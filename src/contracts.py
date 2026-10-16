"""Shared data contracts. Positions use meters in the ego vehicle frame."""

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import numpy as np

Zone = Literal["SAT_SUON", "TAT_DAU", "PHIA_TRUOC"]


@dataclass
class CamCalib:
    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int]
    T_cam2ego: np.ndarray
    z_ground: float = 0.0

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=float).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=float).reshape(-1)
        self.T_cam2ego = np.asarray(self.T_cam2ego, dtype=float).reshape(4, 4)
        self.size = tuple(map(int, self.size))


@dataclass
class Hit:
    x: float
    y: float
    dist: float
    method: Literal["ground", "height"]
    at_edge: bool = False


@dataclass
class Frame:
    t: float
    images: dict[str, np.ndarray]
    calib: dict[str, CamCalib]
    ego_T: np.ndarray
    gt: list[dict] | None = None
    frame: int = 0


@dataclass
class Observation:
    cls: str
    conf: float
    x: float
    y: float
    dets: list[dict] = field(default_factory=list)
    uncertain: bool = False
    reasons: list[str] = field(default_factory=list)


class TrackState(TypedDict):
    id: int
    cls: str
    x: float
    y: float
    vx: float
    vy: float
    trail: list[list[float]]
    zone: Zone | None
    level: int
    ttc: float | None
    uncertain: bool


class RenderState(TypedDict):
    t: float
    source: str
    scene: str
    model: str
    images: dict[str, np.ndarray]
    dets: list[dict]
    tracks: list[TrackState]
    gt: list[dict]
    ego_box: dict[str, float]
    zone_cfg: dict[str, Any]
    stats: dict[str, Any]
