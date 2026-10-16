"""Track objects with a constant-velocity Kalman filter in world coordinates."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Track:
    id: int
    cls: str
    state: np.ndarray
    covariance: np.ndarray
    seen: float
    conf: float
    hits: int = 1
    uncertain: bool = False
    reasons: list[str] = field(default_factory=list)
    dets: list[dict] = field(default_factory=list)
    history: deque = field(default_factory=deque)


class Tracker:
    """Associate by BEV distance; never use image IoU across cameras.

    The internal velocity is absolute in world coordinates. Public vx and vy
    are relative to the moving ego frame, including its yaw rate, for TTC.
    """

    def __init__(self, gate_m=1.5, max_age_s=1.0, trail_s=1.0, process_var=2.0,
                 measurement_var=1.0, initial_velocity_var=25.0):
        self.gate_m = gate_m
        self.max_age_s = max_age_s
        self.trail_s = trail_s
        self.process_var = process_var
        self.initial_velocity_var = initial_velocity_var
        self.set_measurement_noise(measurement_var)
        self.reset()

    def set_measurement_noise(self, variance):
        """Accept a measured variance in m squared or a 2 by 2 covariance."""
        value = np.asarray(variance, dtype=float)
        self.R = np.eye(2) * float(value) if value.ndim == 0 else value.reshape(2, 2)
        if not np.all(np.isfinite(self.R)) or not np.allclose(self.R, self.R.T) or np.min(np.linalg.eigvalsh(self.R)) <= 0:
            raise ValueError("Kalman measurement covariance must be symmetric positive definite")

    def reset(self):
        self.tracks = []
        self.next_id = 1
        self.last_t = None
        self.last_ego = None

    def update(self, observations, t, ego_T):
        ego = np.asarray(ego_T, dtype=float).reshape(4, 4)
        if not np.isfinite(t) or not np.all(np.isfinite(ego)):
            raise ValueError("Tracking requires finite timestamps and ego poses")
        if self.last_t is not None and t <= self.last_t:
            raise ValueError("Tracking timestamps must increase; reset on replay seek")
        dt = t - self.last_t if self.last_t is not None else 0.0
        rotation = ego[:2, :2]
        translation = ego[:2, 3]
        # The ground-plane model assumes roll and pitch are negligible.
        if not np.allclose(rotation.T @ rotation, np.eye(2), atol=0.05):
            raise ValueError("Tracker requires a planar ego pose (negligible roll and pitch)")
        ego_velocity = np.zeros(2)
        yaw_rate = 0.0
        if dt:
            ego_velocity = (translation - self.last_ego[:2, 3]) / dt
            yaw = np.arctan2(ego[1, 0], ego[0, 0])
            old_yaw = np.arctan2(self.last_ego[1, 0], self.last_ego[0, 0])
            yaw_rate = np.arctan2(np.sin(yaw-old_yaw), np.cos(yaw-old_yaw)) / dt
        self.tracks = [track for track in self.tracks if t-track.seen <= self.max_age_s]
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt*dt/2, 0], [0, dt*dt/2], [dt, 0], [0, dt]])
        Q = self.process_var * G @ G.T
        for track in self.tracks:
            track.state = F @ track.state
            track.covariance = F @ track.covariance @ F.T + Q
            track.dets = []
        positions = [rotation @ np.array([o.x, o.y]) + translation for o in observations]
        pairs = []
        for ti, track in enumerate(self.tracks):
            for oi, (obs, position) in enumerate(zip(observations, positions)):
                distance = np.linalg.norm(track.state[:2]-position)
                if track.cls == obs.cls and distance <= self.gate_m:
                    pairs.append((distance, ti, oi))
        matched_tracks, matched_obs = set(), set()
        H = np.eye(2, 4)
        world_R = rotation @ self.R @ rotation.T
        for _, ti, oi in sorted(pairs):
            if ti in matched_tracks or oi in matched_obs:
                continue
            track, obs = self.tracks[ti], observations[oi]
            residual = positions[oi] - H @ track.state
            S = H @ track.covariance @ H.T + world_R
            outlier = float(residual @ np.linalg.solve(S, residual)) > 9.0
            gain = np.linalg.solve(S, H @ track.covariance).T
            track.state += gain @ residual
            correction = np.eye(4) - gain @ H
            track.covariance = correction @ track.covariance @ correction.T + gain @ world_R @ gain.T
            track.hits += 1
            track.seen = t
            track.conf = obs.conf
            track.uncertain = obs.uncertain or outlier
            track.reasons = sorted(set(obs.reasons + (["kalman_residual"] if outlier else [])))
            track.dets = obs.dets
            matched_tracks.add(ti)
            matched_obs.add(oi)
        for oi, obs in enumerate(observations):
            if oi in matched_obs:
                continue
            covariance = np.zeros((4, 4))
            covariance[:2, :2] = world_R
            covariance[2:, 2:] = np.eye(2) * self.initial_velocity_var
            track = _Track(self.next_id, obs.cls, np.r_[positions[oi], 0., 0.], covariance,
                           t, obs.conf, uncertain=obs.uncertain,
                           reasons=list(obs.reasons), dets=obs.dets)
            self.next_id += 1
            self.tracks.append(track)
        result = []
        for track in self.tracks:
            position = rotation.T @ (track.state[:2]-translation)
            velocity = rotation.T @ (track.state[2:]-ego_velocity)
            velocity -= yaw_rate * np.array([-position[1], position[0]])
            if track.dets:
                track.history.append((t, track.state[:2].copy()))
            while track.history and t-track.history[0][0] > self.trail_s:
                track.history.popleft()
            trail = [(rotation.T @ (p-translation)).tolist() for _, p in track.history]
            stale = track.seen < t
            result.append(dict(id=track.id, cls=track.cls, conf=track.conf,
                               x=float(position[0]), y=float(position[1]),
                               vx=float(velocity[0]), vy=float(velocity[1]), trail=trail,
                               zone=None, level=0, ttc=None, velocity_valid=track.hits >= 2,
                               uncertain=track.uncertain or stale,
                               reasons=track.reasons + (["predicted_only"] if stale else []),
                               dets=track.dets))
        self.last_t, self.last_ego = t, ego.copy()
        return result
