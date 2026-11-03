"""Visual planar odometry with training-only pose supervision.

Inputs are two RGB observations and the control that was really executed.
Minecraft yaw is measured from +Z towards -X. Coordinates never enter forward.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def wrap_angle(value):
    """Wrap radians, including differences across the +/-pi seam."""
    if torch.is_tensor(value):
        return torch.atan2(torch.sin(value), torch.cos(value))
    return np.arctan2(np.sin(value), np.cos(value))


def local_pose_delta(previous_position, position, previous_yaw, yaw):
    """Convert simulator XYZ/degrees to local dx,dz,dyaw-radians labels only."""
    previous_position, position = np.asarray(previous_position), np.asarray(position)
    delta = position[..., [0, 2]] - previous_position[..., [0, 2]]
    shape = delta.shape[:-1]
    def heading_array(value):
        value = np.asarray(value)
        if value.shape == (*shape, 1):
            value = value[..., 0]
        return np.deg2rad(np.broadcast_to(value, shape))
    heading = heading_array(previous_yaw)
    x = np.cos(heading)*delta[..., 0] + np.sin(heading)*delta[..., 1]
    z = -np.sin(heading)*delta[..., 0] + np.cos(heading)*delta[..., 1]
    angle = wrap_angle(heading_array(yaw)-heading)
    return np.stack((x, z, angle), -1).astype(np.float32)


def normalized_rgb(image):
    image = torch.as_tensor(image)
    if image.shape[-3:] != (64, 64, 3):
        raise ValueError("PoseNet expects RGB64 with channels last")
    return image.float()/255. if image.dtype == torch.uint8 else image.float()


class PoseNet(nn.Module):
    """Small paired-frame regressor; no command-only translation dead reckoning."""
    def __init__(self, hidden=128, min_samples=256, ready_error=.25, max_translation=1.5,
                 min_calibration=32, min_moving=8, calibration_ratio=.8, cache_size=4096):
        super().__init__()
        self.min_samples, self.ready_error = int(min_samples), float(ready_error)
        self.max_translation = float(max_translation)
        self.min_calibration, self.min_moving = int(min_calibration), int(min_moving)
        self.calibration_ratio, self.cache_size = float(calibration_ratio), int(cache_size)
        if self.cache_size < max(self.min_samples, self.min_calibration, 2*self.min_moving):
            raise ValueError("Pose calibration cache is smaller than required support")
        if self.min_moving < 1 or self.min_calibration < 1 or not 0 < self.calibration_ratio < 1:
            raise ValueError("Pose readiness needs independent moving/static calibration")
        self.visual = nn.Sequential(nn.Conv2d(9, 16, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 48, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.fusion = nn.Sequential(nn.Linear(48*4+9, hidden), nn.LayerNorm(hidden),
                                    nn.SiLU(), nn.Linear(hidden, 6))
        # A fresh model must not invent a path from randomly initialized means.
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)
        with torch.no_grad():
            self.fusion[-1].bias[3:] = torch.tensor([-.7, -.7, -3.])
        self.register_buffer("supervised_samples", torch.zeros((), dtype=torch.long))
        self.register_buffer("position_error_ema", torch.full((), 1000.))
        self.register_buffer("angle_error_ema", torch.full((), math.pi))
        self.register_buffer("moving_samples", torch.zeros((), dtype=torch.long))
        self.register_buffer("calibration_samples", torch.zeros((), dtype=torch.long))
        self.register_buffer("calibration_moving_samples", torch.zeros((), dtype=torch.long))
        self.register_buffer("calibration_static_samples", torch.zeros((), dtype=torch.long))
        self.register_buffer("calibration_error", torch.full((), 1000.))
        self.register_buffer("calibration_moving_error", torch.full((), 1000.))
        self.register_buffer("calibration_static_error", torch.full((), 1000.))
        self.register_buffer("calibration_moving_baseline", torch.zeros(()))
        self.register_buffer("calibration_angle_error", torch.full((), math.pi))
        self.register_buffer("identity_counter", torch.zeros((), dtype=torch.long))
        for prefix in ("train", "cal"):
            self.register_buffer(prefix+"_identity", torch.full((self.cache_size,), -1, dtype=torch.long))
            self.register_buffer(prefix+"_error", torch.zeros(self.cache_size))
            self.register_buffer(prefix+"_baseline", torch.zeros(self.cache_size))
            self.register_buffer(prefix+"_angle", torch.zeros(self.cache_size))

    @property
    def ready(self):
        return bool((self.supervised_samples >= self.min_samples) &
                    (self.moving_samples >= self.min_moving) &
                    (self.calibration_samples >= self.min_calibration) &
                    (self.calibration_moving_samples >= self.min_moving) &
                    (self.calibration_static_samples >= self.min_moving) &
                    (self.calibration_error <= self.ready_error) &
                    # The general .25 m bound exceeds normal walking per step.
                    # Static/collision cases must reject a command-only walker
                    # that predicts motion even when the camera did not move.
                    (self.calibration_static_error <= min(self.ready_error,.03)) &
                    (self.calibration_moving_error < self.calibration_ratio*self.calibration_moving_baseline))

    @torch.no_grad()
    def record_predictions(self, delta, target, train_mask, calibration_mask, frame_ids=None):
        """Refresh bounded unique-identity evidence, never count resampling twice.

        Train and calibration masks are world-disjoint in spatial replay. Cache
        collisions evict an old owner; recomputing totals removes its support.
        This is model qualification only, not a gradient-bearing objective.
        """
        shape = target.shape[:-1]
        if frame_ids is None:
            frame_ids = torch.arange(target.numel()//3, device=target.device).reshape(shape)+self.identity_counter
            self.identity_counter.add_(target.numel()//3)
        frame_ids = torch.as_tensor(frame_ids, device=target.device).long()
        if frame_ids.shape != shape:
            raise ValueError("Pose frame identities do not match supervision")
        error = torch.linalg.vector_norm(delta[..., :2]-target[..., :2], dim=-1).detach().flatten()
        baseline = torch.linalg.vector_norm(target[..., :2], dim=-1).detach().flatten()
        angle = wrap_angle(delta[..., 2]-target[..., 2]).abs().detach().flatten()
        ids = frame_ids.flatten()
        slots = torch.remainder(ids*31+ids//17, self.cache_size)
        for prefix, mask in (("train", train_mask), ("cal", calibration_mask)):
            positions = mask.flatten().nonzero().squeeze(-1)
            if len(positions):
                positions = positions[torch.argsort(slots[positions], stable=True)]
                # One last observation per physical cache slot in this batch.
                keep = torch.ones(len(positions), device=target.device, dtype=torch.bool)
                keep[:-1] = slots[positions[:-1]] != slots[positions[1:]]
                positions = positions[keep]
                destination = slots[positions]
                getattr(self, prefix+"_identity")[destination] = ids[positions]
                getattr(self, prefix+"_error")[destination] = error[positions]
                getattr(self, prefix+"_baseline")[destination] = baseline[positions]
                getattr(self, prefix+"_angle")[destination] = angle[positions]
        valid = self.train_identity >= 0
        moving = valid & (self.train_baseline > .03)
        self.supervised_samples.copy_(valid.sum())
        self.moving_samples.copy_(moving.sum())
        if bool(valid.any()):
            # Historical names retained; these now summarize the current cache.
            self.position_error_ema.copy_(self.train_error[valid].mean())
            self.angle_error_ema.copy_(self.train_angle[valid].mean())
        valid = self.cal_identity >= 0
        moving = valid & (self.cal_baseline > .03)
        stationary = valid & ~moving
        self.calibration_samples.copy_(valid.sum())
        self.calibration_moving_samples.copy_(moving.sum())
        self.calibration_static_samples.copy_(stationary.sum())
        if bool(valid.any()):
            self.calibration_error.copy_(self.cal_error[valid].mean())
            self.calibration_angle_error.copy_(self.cal_angle[valid].mean())
        if bool(moving.any()):
            self.calibration_moving_error.copy_(self.cal_error[moving].mean())
            self.calibration_moving_baseline.copy_(self.cal_baseline[moving].mean())
        if bool(stationary.any()):
            self.calibration_static_error.copy_(self.cal_error[stationary].mean())

    def forward(self, previous_image, image, control):
        previous_image = normalized_rgb(previous_image)
        image = normalized_rgb(image).to(previous_image.device)
        if previous_image.shape != image.shape:
            raise ValueError("PoseNet image pair has different shapes")
        lead = image.shape[:-3]
        control = torch.as_tensor(control, device=image.device).float()
        if control.shape != (*lead, 9):
            raise ValueError("PoseNet requires nine applied-control components")
        pair = torch.cat((previous_image, image, image-previous_image), -1)
        encoded = self.visual(pair.reshape(-1, 64, 64, 9).permute(0, 3, 1, 2))
        scale = control.new_tensor([1, 1, 1, 1, 1, 1, 1, 10, 10])
        raw = self.fusion(torch.cat((encoded, (control/scale).reshape(-1, 9)), -1))
        raw = raw.reshape(*lead, 6)
        translation = torch.tanh(raw[..., :2])*self.max_translation
        # Camera commands are known legal proprioception, not world yaw. A small
        # learned residual models synchronization; body translation stays visual.
        yaw_prior = control[..., 8]*math.pi/180.
        angle = wrap_angle(yaw_prior + torch.tanh(raw[..., 2])*.1)
        mean = torch.cat((translation, angle[..., None]), -1)
        log_std = raw[..., 3:].clamp(-5., 1.)
        return dict(delta=mean, log_std=log_std, sigma=log_std.exp())

    def supervised_loss(self, previous_image, image, control, target, valid, record=True,
                        calibration_mask=None, frame_ids=None):
        prediction = self(previous_image, image, control)
        target = torch.as_tensor(target, device=prediction["delta"].device).float()
        valid = torch.as_tensor(valid, device=target.device).bool().clone()
        if target.shape != prediction["delta"].shape or valid.shape != target.shape[:-1]:
            raise ValueError("Pose supervision shape mismatch")
        known = torch.isfinite(target).all(-1)
        # Teleports / failed reset pairs cannot supervise locomotion.
        known &= torch.linalg.vector_norm(target[..., :2].nan_to_num(), dim=-1) <= 3.
        valid &= known
        calibration = (torch.zeros_like(valid) if calibration_mask is None else
                       torch.as_tensor(calibration_mask, device=target.device).bool()) & known
        if calibration.shape != valid.shape or bool((calibration & valid).any()):
            raise ValueError("Pose calibration must be shape-compatible and separate from training")
        target = target.nan_to_num(nan=0., posinf=0., neginf=0.)
        error = prediction["delta"]-target
        error = torch.cat((error[..., :2], wrap_angle(error[..., 2])[..., None]), -1)
        # Heteroscedastic Laplace NLL plus a robust mean regression term.
        nll = error.abs()*prediction["log_std"].neg().exp()+prediction["log_std"]
        per_row = nll.mean(-1)+F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none").mean(-1)
        loss = (per_row*valid).sum()/valid.sum().clamp_min(1)
        count = valid.sum().detach()
        pos_error = (torch.linalg.vector_norm(error[..., :2], dim=-1)*valid).sum()/count.clamp_min(1)
        yaw_error = (error[..., 2].abs()*valid).sum()/count.clamp_min(1)
        if record:
            self.record_predictions(prediction["delta"], target, valid, calibration, frame_ids)
        metrics = {"B/pose_loss": loss.detach(), "B/pose_samples": count,
            "B/pose_unique_samples": self.supervised_samples.detach().clone(),
            "B/pose_moving_samples": self.moving_samples.detach().clone(),
            "B/pose_calibration_samples": self.calibration_samples.detach().clone(),
            "B/pose_calibration_moving": self.calibration_moving_samples.detach().clone(),
            "B/pose_calibration_static": self.calibration_static_samples.detach().clone(),
            "B/pose_ready": float(self.ready)}
        # No-label/uncalibrated sentinels are internal qualification state, not
        # measured zero error or a measured 1000m error on TensorBoard.
        if bool(count > 0):
            metrics.update({"B/pose_position_error": pos_error.detach(),
                            "B/pose_angle_error": yaw_error.detach()})
        if bool(self.calibration_samples > 0):
            metrics.update({"B/pose_calibration_error": self.calibration_error.detach().clone(),
                            "B/pose_calibration_angle_error": self.calibration_angle_error.detach().clone()})
        if bool(self.calibration_moving_samples > 0):
            metrics.update({"B/pose_calibration_moving_error": self.calibration_moving_error.detach().clone(),
                            "B/pose_zero_motion_baseline": self.calibration_moving_baseline.detach().clone()})
        if bool(self.calibration_static_samples > 0):
            metrics["B/pose_calibration_static_error"] = self.calibration_static_error.detach().clone()
        return loss, metrics, prediction
