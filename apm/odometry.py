"""Visual planar odometry with training-only pose supervision.

Inputs are two RGB observations and the control that was really executed.
Minecraft yaw is measured from +Z towards -X. Coordinates never enter forward.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


MIN_TRANSLATION = .03
MIN_MOTION_PROBABILITY = .8
MAX_TRANSLATION_SIGMA = .1
MAX_STATIC_TAIL = .05


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
                 min_calibration=32, min_moving=8, calibration_ratio=.8, cache_size=4096,
                 fix_version=1, loss_version=1):
        super().__init__()
        self.min_samples, self.ready_error = int(min_samples), float(ready_error)
        self.max_translation = float(max_translation)
        self.min_calibration, self.min_moving = int(min_calibration), int(min_moving)
        self.calibration_ratio, self.cache_size = float(calibration_ratio), int(cache_size)
        self.fix_version = int(fix_version)
        self.loss_version = int(loss_version)
        if self.loss_version not in (1, 2):
            raise ValueError("Unknown pose loss revision")
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
            if self.fix_version:
                # Preserve the six-output parameter shape. Protocol revision 1
                # uses output 2 as a supervised motion logit and output 5 as the
                # small yaw residual; old revision 0 keeps its original layout.
                self.fusion[-1].bias[2] = -2.
                self.fusion[-1].bias[5] = 0.
        if self.loss_version == 2:
            axis = (torch.arange(64, dtype=torch.float32)+.5)*2/64-1
            self.register_buffer("camera_grid", torch.stack(torch.meshgrid(axis, axis, indexing="xy"), -1),
                                 persistent=False)
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
                    (self.calibration_moving_error < self.calibration_ratio*self.calibration_moving_baseline) &
                    # A small mean can hide a minority of large false steps.
                    # This tail guard qualifies the integrated estimator only;
                    # its cache never feeds back into individual motion gates.
                    ((self.static_error_bound() <= MAX_STATIC_TAIL) if self.fix_version else True))

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
        # Slow/ambiguous translations must not become "stationary" calibration
        # examples. The old protocol used every <=3 cm observation as static.
        stationary = valid & (self.cal_baseline <= (.01 if self.fix_version else .03))
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

    @torch.no_grad()
    def static_error_bound(self):
        """Held-out static tail, not the easy all-scene average error.

        This uses the existing identity cache (no checkpoint shape change).
        Unavailable support has no authority to certify a measured error.
        It is a qualification/diagnostic statistic, not an inference threshold.
        """
        mask = (self.cal_identity >= 0) & (self.cal_baseline <= .01)
        if not bool(mask.any()):
            return self.calibration_static_error.new_tensor(.01)
        return self.cal_error[mask].quantile(.95)

    @torch.no_grad()
    def translation_support(self, prediction, previous_image, image):
        """Conservative per-pair motion evidence; never equate commands to motion.

        A separately supervised static/moving logit prevents ordinary regression
        from averaging "walk" and "blocked walk" into continuous fake motion.
        A translation must also exceed the fixed 3cm pose-label resolution. Exact
        or almost unchanged frames cannot support drift from a command shortcut.
        Camera rotations are not rejected categorically: a calibrated model can
        still predict sufficiently supported translation while turning.
        """
        mean, sigma = prediction["delta"][..., :2], prediction["sigma"][..., :2]
        raw_motion = torch.linalg.vector_norm(mean, dim=-1)
        # This fixed threshold is intentionally independent of gated calibration
        # errors. Otherwise rejecting a pair would zero its cached error, lower
        # the next acceptance threshold and create a qualification feedback loop.
        radius = torch.full_like(raw_motion, MIN_TRANSLATION)
        before, after = normalized_rgb(previous_image), normalized_rgb(image)
        visual_difference = prediction.get("visual_difference")
        if visual_difference is None:
            visual_difference = (after-before).abs().mean((-1, -2, -3))
        finite = torch.isfinite(mean).all(-1) & torch.isfinite(sigma).all(-1)
        finite &= torch.isfinite(before).flatten(-3).all(-1) & torch.isfinite(after).flatten(-3).all(-1)
        probability = prediction.get("motion_probability", torch.zeros_like(raw_motion))
        supported = finite & (visual_difference > (1./255.)) & (raw_motion >= radius)
        supported &= (probability >= MIN_MOTION_PROBABILITY) & (sigma.mean(-1) <= MAX_TRANSLATION_SIGMA)
        return supported, raw_motion, radius

    def forward(self, previous_image, image, control):
        previous_image = normalized_rgb(previous_image)
        image = normalized_rgb(image).to(previous_image.device)
        if previous_image.shape != image.shape:
            raise ValueError("PoseNet image pair has different shapes")
        lead = image.shape[:-3]
        control = torch.as_tensor(control, device=image.device).float()
        if control.shape != (*lead, 9):
            raise ValueError("PoseNet requires nine applied-control components")
        difference = image - previous_image
        residual = difference
        if self.loss_version == 2:
            residual = self._camera_residual(previous_image, image, control)
        pair = torch.cat((previous_image, image, residual), -1)
        encoded = self.visual(pair.reshape(-1, 64, 64, 9).permute(0, 3, 1, 2))
        scale = control.new_tensor([1, 1, 1, 1, 1, 1, 1, 10, 10])
        raw = self.fusion(torch.cat((encoded, (control/scale).reshape(-1, 9)), -1))
        raw = raw.reshape(*lead, 6)
        translation = torch.tanh(raw[..., :2])*self.max_translation
        # Camera commands are known legal proprioception, not world yaw. A small
        # learned residual models synchronization; body translation stays visual.
        yaw_prior = control[..., 8]*math.pi/180.
        angle = wrap_angle(yaw_prior + torch.tanh(raw[..., 5 if self.fix_version else 2])*.1)
        mean = torch.cat((translation, angle[..., None]), -1)
        if self.fix_version:
            log_std = torch.cat((raw[..., 3:5].clamp(-5., 1.), torch.full_like(raw[..., 5:6], -3.)), -1)
        else:
            log_std = raw[..., 3:].clamp(-5., 1.)
        result = dict(delta=mean, log_std=log_std, sigma=log_std.exp())
        if self.fix_version:
            result.update(motion_logit=raw[..., 2], motion_probability=raw[..., 2].sigmoid())
        if self.loss_version == 2:
            result["visual_difference"] = difference.abs().mean((-1, -2, -3))
        return result

    def _camera_residual(self, previous_image, image, control):
        # Approximate the center-of-view rotation from own applied pitch/yaw
        # and MineDojo's default 256x160 render aspect. Keep both original frames
        # available so the model can handle parallax and imperfect alignment.
        previous = previous_image.reshape(-1, 64, 64, 3).permute(0, 3, 1, 2)
        camera = control[..., [8, 7]].reshape(-1, 2)
        shift = camera.deg2rad().tan() / math.tan(math.radians(35.))
        shift = shift / shift.new_tensor([256./160., 1.])
        grid = self.camera_grid + shift[:, None, None, :]
        aligned = F.grid_sample(previous, grid, padding_mode="border", align_corners=False)
        return image - aligned.permute(0, 2, 3, 1).reshape_as(image)

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
        static = valid & (torch.linalg.vector_norm(target[..., :2], dim=-1) <= .01)
        moving = valid & (torch.linalg.vector_norm(target[..., :2], dim=-1) > .03)
        bias_loss = (error[..., :2].square().sum(-1)*static).sum()/static.sum().clamp_min(1)
        classification_loss = prediction["delta"].sum()*0.
        if self.fix_version:
            # Random replay can contain many more successful moving pairs than
            # collisions (or vice versa). Equal class means prevent the easy
            # majority from hiding the exact command-only failure seen in 400k.
            groups = (static, moving, valid & ~static & ~moving)
            means = torch.stack([(per_row*m).sum()/m.sum().clamp_min(1) for m in groups])
            present = torch.stack([m.any() for m in groups])
            balanced = (means*present).sum()/present.sum().clamp_min(1)
            # An explicit mean penalty complements the heteroscedastic NLL;
            # claiming larger variance cannot excuse a stationary motion bias.
            classification = F.binary_cross_entropy_with_logits(prediction["motion_logit"], moving.float(), reduction="none")
            group_classification = torch.stack([(classification*m).sum()/m.sum().clamp_min(1) for m in (static, moving)])
            present_classification = torch.stack([static.any(), moving.any()])
            classification_loss = (group_classification*present_classification).sum()/present_classification.sum().clamp_min(1)
            if self.loss_version == 2:
                stratified_classification, moving_risk, static_tail = self._gated_translation_loss(
                    prediction, error, target, control, previous_image, image, static, moving, classification)
                loss = balanced + 10.*bias_loss + 3.*stratified_classification + 5.*moving_risk + 8.*static_tail
            else:
                loss = balanced+10.*bias_loss+3.*classification_loss
        else:
            loss = (per_row*valid).sum()/valid.sum().clamp_min(1)
        count = valid.sum().detach()
        pos_error = (torch.linalg.vector_norm(error[..., :2], dim=-1)*valid).sum()/count.clamp_min(1)
        yaw_error = (error[..., 2].abs()*valid).sum()/count.clamp_min(1)
        if record:
            qualification_delta = prediction["delta"].detach().clone()
            if self.fix_version:
                # Readiness concerns the actually integrated estimator. A model
                # that always classifies every pair as static cannot qualify by
                # its unused raw regression output beating the zero baseline.
                support, _, _ = self.translation_support(prediction, previous_image, image)
                qualification_delta[..., :2] *= support[..., None]
            self.record_predictions(qualification_delta, target, valid, calibration, frame_ids)
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
        if self.fix_version:
            metrics.update({"B/pose_static_samples": static.sum().detach(),
                "B/pose_static_bias_loss": bias_loss.detach(),
                "B/pose_motion_classification_loss": classification_loss.detach(),
                "B/pose_motion_train_samples": moving.sum().detach()})
            if self.loss_version == 2:
                metrics.update({"B/pose_stratified_motion_loss": stratified_classification.detach(),
                                "B/pose_gated_moving_risk": moving_risk.detach(),
                                "B/pose_gated_static_tail_loss": static_tail.detach()})
            with torch.no_grad():
                accepted, _, _ = self.translation_support(prediction, previous_image, image)
                if bool(static.any()):
                    metrics["B/pose_static_false_motion_rate"] = (accepted & static).sum()/static.sum()
                if bool(moving.any()):
                    metrics["B/pose_moving_acceptance_rate"] = (accepted & moving).sum()/moving.sum()
            if bool(static.any()):
                metrics["B/pose_static_position_error"] = (torch.linalg.vector_norm(error[..., :2],dim=-1)*static).sum().detach()/static.sum()
        if bool(self.calibration_samples > 0):
            metrics.update({"B/pose_calibration_error": self.calibration_error.detach().clone(),
                            "B/pose_calibration_angle_error": self.calibration_angle_error.detach().clone()})
        if bool(self.calibration_moving_samples > 0):
            metrics.update({"B/pose_calibration_moving_error": self.calibration_moving_error.detach().clone(),
                            "B/pose_zero_motion_baseline": self.calibration_moving_baseline.detach().clone()})
        if bool(self.calibration_static_samples > 0):
            metrics["B/pose_calibration_static_error"] = self.calibration_static_error.detach().clone()
            if self.fix_version:
                metrics["B/pose_calibration_static_p95"] = self.static_error_bound().detach().clone()
        return loss, metrics, prediction

    @staticmethod
    def _gated_translation_loss(prediction, error, target, control, previous, image, static, moving, classification):
        """Optimize the motion estimator used by the map, including rare false steps."""
        control = torch.as_tensor(control, device=target.device)
        commanded = (control[..., 1:5].abs() > 0).any(-1)
        # Blocked walks and continued motion without a fresh walk command are
        # real labels. Give each subgroup support independent of its frequency.
        groups = torch.stack((static & ~commanded, static & commanded, moving & ~commanded, moving & commanded))
        counts = groups.reshape(4, -1).sum(-1)
        means = (classification * groups).reshape(4, -1).sum(-1) / counts.clamp_min(1)
        present = (counts > 0).reshape(2, 2)
        class_means = means.reshape(2, 2).sum(-1) / present.sum(-1).clamp_min(1)
        stratified = class_means.sum() / present.any(-1).sum().clamp_min(1)

        # Soft training surrogate for the unchanged hard inference gate. The
        # rejected branch costs the same error as a zero-displacement estimate.
        magnitude = torch.linalg.vector_norm(prediction["delta"][..., :2].float(), dim=-1)
        logit_threshold = math.log(MIN_MOTION_PROBABILITY / (1 - MIN_MOTION_PROBABILITY))
        probability = ((prediction["motion_logit"].float() - logit_threshold) * 2.).sigmoid()
        sigma = prediction["sigma"][..., :2].float().mean(-1)
        probability = probability * ((MAX_TRANSLATION_SIGMA - sigma) / .02).sigmoid()
        probability = probability * ((magnitude - MIN_TRANSLATION) / .01).sigmoid()
        visual_change = prediction.get("visual_difference")
        if visual_change is None:
            visual_change = (normalized_rgb(image).to(target.device) -
                             normalized_rgb(previous).to(target.device)).abs().mean((-1, -2, -3))
        probability = probability * (visual_change > (1./255.))
        position_error = torch.linalg.vector_norm(error[..., :2], dim=-1)
        zero_error = torch.linalg.vector_norm(target[..., :2], dim=-1)
        risk = torch.lerp(zero_error, position_error, probability)
        moving_risk = (risk * moving).sum() / moving.sum().clamp_min(1)

        # The worst static pairs should not disappear into an easy static mean.
        # Rank differentiable excess over the held-out tail bound, keeping only
        # true supervised static pairs. Calibration is never used for gradients.
        excess = ((position_error - MAX_STATIC_TAIL).relu() * probability)[static]
        tail_count = math.ceil(.05 * excess.numel())
        static_tail = excess.topk(tail_count).values.sum() / max(1, tail_count)
        return stratified, moving_risk, static_tail
