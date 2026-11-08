"""Category reference matching with sparse, training-only supervision."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .actuator import world_group_id


VERSION = 1
GRID = 16
REFERENCE_SIZE = 64
TASKS = {
    "harvest_log_in_plains": "log",
    "harvest_water_with_bucket": "water",
    "harvest_sand": "sand",
    "mine_iron_ore": "iron_ore",
    "shear_sheep": "sheep",
}


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _task_name(task):
    return str(task).removeprefix("minedojo_")


def _entry_ok(entry, task):
    return (_task_name(entry.get("task_id")) == task
            and entry.get("split") == "train"
            and (entry.get("world_seed") is not None or entry.get("world_group") is not None))


def _entry_worlds(entry):
    groups = set()
    if entry.get("world_group") is not None:
        groups.add(int(entry["world_group"]))
    if entry.get("world_seed") is not None:
        seed = str(entry["world_seed"])
        try:
            seed = str(int(seed))
        except ValueError:
            pass
        groups.add(world_group_id("fixed-world-seed:" + seed, 0))
    return groups


def load_reference_bank(root, task, max_references):
    """Load only training assets matching the fixed run category."""
    root = Path(root).resolve()
    task_root = root / task
    if (task_root / "manifest.json").is_file():
        root = task_root.resolve()
    source = root / "manifest.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing goal reference manifest {source}. Build it from verified training views.")
    manifest = json.loads(source.read_text(encoding="utf-8-sig"))
    if manifest.get("version") != VERSION:
        raise ValueError("Unsupported goal reference manifest version")
    entries, pixels, negative_entries, negatives = [], [], [], []
    for entry in manifest["entries"]:
        if _task_name(entry.get("task_id")) != task:
            continue
        if not _entry_ok(entry, task):
            raise ValueError("References must record a training-world seed")
        negative = entry.get("role") == "negative"
        if (negative and len(negatives) >= max_references) or (not negative and len(pixels) >= max_references):
            continue
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Reference image must be inside its manifest directory")
        if _sha(path.read_bytes()) != entry["sha256"]:
            raise ValueError(f"Goal reference SHA256 mismatch: {path.name}")
        with Image.open(path) as image:
            image = image.convert("RGB")
            if entry.get("roi") is not None:
                roi = np.asarray(entry["roi"], np.float64)
                if (roi.shape != (4,) or not np.isfinite(roi).all()
                        or (roi < 0).any() or (roi > 1).any()
                        or (roi[2:] <= roi[:2]).any()):
                    raise ValueError("Reference ROI must be normalized xyxy")
                width, height = image.size
                x0, y0, x1, y1 = roi * (width, height, width, height)
                image = image.crop((int(np.floor(x0)), int(np.floor(y0)),
                                    int(np.ceil(x1)), int(np.ceil(y1))))
            sample = np.asarray(image.resize((REFERENCE_SIZE, REFERENCE_SIZE), Image.BILINEAR), dtype=np.uint8)
        if negative:
            negatives.append(sample)
            negative_entries.append(dict(entry))
        else:
            pixels.append(sample)
            entries.append(dict(entry))
    if not pixels:
        raise ValueError(f"No verified training references for {task} in {source}")
    pixels = np.stack(pixels)
    negatives = (np.stack(negatives) if negatives else
                 np.empty((0, REFERENCE_SIZE, REFERENCE_SIZE, 3), dtype=np.uint8))
    frozen = dict(version=VERSION, task_id=task, target_class=TASKS[task],
                  entries=entries, pixels_sha256=_sha(pixels.tobytes()),
                  negative_entries=negative_entries, negative_pixels_sha256=_sha(negatives.tobytes()))
    return {"manifest": frozen, "pixels": pixels, "negative_pixels": negatives}


def _rgb(image, device):
    image = torch.as_tensor(image, device=device)
    if image.ndim < 3 or image.shape[-1] != 3:
        raise ValueError("D expects channels-last RGB")
    rgb = image.reshape(-1, *image.shape[-3:]).permute(0, 3, 1, 2).float()
    if image.dtype == torch.uint8:
        rgb = rgb / 255.
    return torch.nan_to_num(rgb).clamp(0., 1.)


def _heatmap(value, shape, device):
    value = torch.as_tensor(value, device=device)
    if value.ndim == len(shape) + 3 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if value.shape[:-2] != shape:
        raise ValueError("D heatmap and image leading dimensions differ")
    heatmap = value.reshape(-1, 1, *value.shape[-2:]).float()
    if value.dtype == torch.uint8:
        heatmap = heatmap / 255.
    return torch.nan_to_num(heatmap).clamp(0., 1.)


def _masked_mean(values, mask):
    weight = mask.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.)


def _balanced_bce(logits, label, valid, hard=None):
    pos, neg = valid & label, valid & ~label
    losses = F.binary_cross_entropy_with_logits(logits, label.float(), reduction="none")
    weight = neg.float()
    if hard is not None:
        weight = weight * (1. + 2. * hard.float())
    return (_masked_mean(losses, pos) + (losses * weight).sum() / weight.sum().clamp_min(1.)) / (
        pos.any().float() + neg.any().float()).clamp_min(1.)


class GoalReference(nn.Module):
    """Lightweight shared RGB encoder, reference matcher, and supervised heads."""

    def __init__(self, config, bank=None):
        super().__init__()
        task = _task_name(config.task)
        if task not in TASKS:
            raise ValueError(f"Unsupported D target task: {task}")
        if int(getattr(config, "goal_ref_version", VERSION)) != VERSION:
            raise ValueError("Unsupported D model version")
        max_references = int(getattr(config, "goal_ref_max_references", 32))
        if not 1 <= max_references <= 32:
            raise ValueError("D supports 1–32 fixed references")
        self.max_candidates = int(getattr(config, "goal_ref_max_candidates", 8))
        if self.max_candidates != 8:
            raise ValueError("D v1 uses eight fixed candidate slots")
        bank = (load_reference_bank(config.goal_ref_bank_root, task, max_references)
                if bank is None else bank)
        manifest = json.loads(json.dumps(bank["manifest"], sort_keys=True))
        pixels = torch.as_tensor(bank["pixels"]).detach().cpu().contiguous()
        entries = manifest["entries"]
        negative_entries = manifest.get("negative_entries", [])
        negatives = torch.as_tensor(bank.get("negative_pixels", torch.empty(
            (0, REFERENCE_SIZE, REFERENCE_SIZE, 3), dtype=torch.uint8))).detach().cpu().contiguous()
        if (manifest["version"] != VERSION or manifest["task_id"] != task
                or manifest["target_class"] != TASKS[task]
                or not 1 <= len(entries) <= max_references
                or not all(_entry_ok(entry, task) and entry.get("role") != "negative" for entry in entries)
                or len(negative_entries) > max_references
                or not all(_entry_ok(entry, task) and entry.get("role") == "negative"
                           for entry in negative_entries)
                or negatives.shape != (len(negative_entries), REFERENCE_SIZE, REFERENCE_SIZE, 3)
                or negatives.dtype != torch.uint8
                or _sha(negatives.numpy().tobytes()) != manifest.get("negative_pixels_sha256", _sha(b""))
                or pixels.shape != (len(entries), REFERENCE_SIZE, REFERENCE_SIZE, 3)
                or pixels.dtype != torch.uint8
                or _sha(pixels.numpy().tobytes()) != manifest["pixels_sha256"]):
            raise ValueError("Invalid or incompatible checkpoint goal reference bank")
        self.bank_manifest = manifest
        self.bank_fingerprint = _sha(json.dumps(
            manifest, sort_keys=True, separators=(",", ":")).encode())
        reference_worlds = [_entry_worlds(entry) for entry in entries]
        groups = set().union(*reference_worlds, *map(_entry_worlds, negative_entries))
        self.register_buffer("bank_world_groups", torch.tensor(sorted(groups), dtype=torch.long),
                             persistent=False)
        self.register_buffer("cross_world_references", torch.tensor([
            [left.isdisjoint(right) for right in reference_worlds] for left in reference_worlds
        ], dtype=torch.bool), persistent=False)
        self.register_buffer("bank_pixels", pixels.clone())
        self.register_buffer("negative_pixels", negatives.clone())
        self.register_buffer("reference_features", torch.empty(0), persistent=False)
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        dim = int(getattr(config, "goal_ref_dim", 128))
        self.encoder = nn.Sequential(nn.Conv2d(3, 24, 5, 2, 2), nn.GroupNorm(4, 24), nn.SiLU(),
                                     nn.Conv2d(24, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
                                     nn.Conv2d(32, dim, 3, 1, 1))
        self.match = nn.Sequential(nn.Conv2d(dim + 3, 32, 1), nn.SiLU(),
                                   nn.Conv2d(32, 2, 1))
        self.visibility = nn.Sequential(nn.Linear(dim + 3, 32), nn.SiLU(), nn.Linear(32, 1))
        self.interaction = nn.Sequential(nn.Linear(dim + 3, 32), nn.SiLU(), nn.Linear(32, 1))
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=float(getattr(config, "goal_ref_lr", 1e-4)))
        self.frame_limit = max(2, int(getattr(config, "goal_ref_batch_frames", 64)))
        self.chunk = max(1, int(getattr(config, "goal_ref_chunk", 64)))
        self.loss_weight = float(getattr(config, "goal_ref_loss_weight", 1.))
        self.use_raw_image = bool(getattr(config, "goal_ref_use_raw_image", True))
        self.input_size = 96 if self.use_raw_image else 64
        self.fusion_enabled = bool(getattr(config, "goal_fusion_enabled", True))
        self.max_gate = float(getattr(config, "goal_fusion_max_gate", .8))
        self.fusion_epsilon = float(getattr(config, "goal_fusion_epsilon", .05))
        self.fusion_overlap = float(getattr(config, "goal_fusion_overlap_weight", .5))
        if not (0. <= self.max_gate <= .8 and 0. <= self.fusion_epsilon <= 1.
                and 0. <= self.fusion_overlap <= 1.):
            raise ValueError("D fusion bounds are invalid")
        self.hard_negatives = bool(getattr(config, "goal_ref_hard_negative_enabled", True))
        # Fixed, unique frame IDs. Allocate separate halves so a rare visible
        # category is not immediately evicted by commonly observed background.
        capacity = 64
        self.register_buffer("cal_ids", torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer("cal_world_groups", torch.full((capacity,), -1, dtype=torch.long),
                             persistent=False)
        self.register_buffer("cal_images", torch.zeros((capacity, self.input_size, self.input_size, 3), dtype=torch.uint8))
        self.register_buffer("cal_labels", torch.zeros((capacity, 2, GRID, GRID), dtype=torch.bool))
        self.register_buffer("cal_frame_labels", torch.zeros((capacity, 2, 2), dtype=torch.bool))
        self.register_buffer("temperature", torch.ones(3))
        self.register_buffer("reliability", torch.zeros(3))
        self.register_buffer("cal_counts", torch.zeros((3, 2), dtype=torch.long))
        self.register_buffer("cal_brier", torch.full((3,), .25))

    @property
    def device(self):
        return self.bank_pixels.device

    def export_bank(self):
        return {"manifest": json.loads(json.dumps(self.bank_manifest)),
                "pixels": self.bank_pixels.detach().cpu().clone(),
                "negative_pixels": self.negative_pixels.detach().cpu().clone()}

    def _load_from_state_dict(self, state_dict, prefix, *args):
        for name in ("bank_pixels", "negative_pixels"):
            key = prefix + name
            if key in state_dict and not torch.equal(state_dict[key].cpu(), getattr(self, name).cpu()):
                raise ValueError("Saved D bank pixels differ from the verified bank manifest")
        result = super()._load_from_state_dict(state_dict, prefix, *args)
        self.reference_features = self.bank_pixels.new_empty(0, dtype=torch.float32)
        return result

    def observation_images(self, raw_obs):
        """Use native same-frame pixels where present, including mixed replay."""
        raw = raw_obs["image"]
        leading = tuple(raw.shape[:-3])
        rgb = _rgb(raw, self.device)
        size = (self.input_size, self.input_size)
        if rgb.shape[-2:] != size:
            rgb = F.interpolate(rgb, size, mode="bilinear", align_corners=False)
        if self.use_raw_image and raw_obs.get("goal_image") is not None:
            if raw_obs.get("goal_image_valid") is None:
                raise ValueError("Native RGB needs its availability marker")
            native = _rgb(raw_obs["goal_image"], self.device)
            if native.shape[-2:] != size:
                raise ValueError("Native goal image must use 96x96 pixels")
            valid = torch.as_tensor(raw_obs["goal_image_valid"], device=self.device).bool().reshape(-1)
            rgb = torch.where(valid[:, None, None, None], native, rgb)
        return (rgb.permute(0, 2, 3, 1).reshape(*leading, *size, 3)*255.).round().byte()

    def _encode(self, rgb):
        return F.adaptive_avg_pool2d(self.encoder(rgb), (GRID, GRID))

    def _reference(self):
        ref = self._encode(_rgb(self.bank_pixels, self.device)).mean((-2, -1))
        return F.normalize(ref, dim=-1)

    @torch.no_grad()
    def _cached_reference(self):
        if not self.reference_features.numel():
            self.reference_features = self._reference()
        return self.reference_features

    def _match(self, rgb, references, valid_references=None):
        features = self._encode(rgb)
        embeddings = F.normalize(features, dim=1)
        similarities = torch.einsum("nchw,rc->nrhw", embeddings, references)
        matches = similarities
        if valid_references is not None:
            matches = matches.masked_fill(~valid_references[..., None, None], -torch.inf)
        top = matches.topk(min(4, len(references)), dim=1).values
        best = top[:, :1]
        if valid_references is None:
            mean, worst = top.mean(1, keepdim=True), top[:, -1:]
        else:
            valid = torch.isfinite(top)
            mean = top.masked_fill(~valid, 0.).sum(1, keepdim=True) / valid.sum(1, keepdim=True).clamp_min(1)
            worst = torch.where(valid, top, best).amin(1, keepdim=True)
        stats = torch.cat((best, mean, best - worst), 1)
        context = torch.cat((features, stats), 1)
        ensemble = self.match(context)
        return dict(region_logits=ensemble.mean(1), ensemble=ensemble, embeddings=embeddings,
                    similarities=similarities, saliency=top[:, 0],
                    visible_logits=self.visibility(context.amax((-2, -1))).squeeze(-1),
                    interaction_logits=self.interaction(
                        context[:, :, GRID//2-1:GRID//2+1, GRID//2-1:GRID//2+1].mean((-2,-1))).squeeze(-1))

    @torch.no_grad()
    def forward(self, image, heatmap):
        """Pure observation inference, accepting [B] or [B,T] frame dimensions."""
        rgb = _rgb(image, self.device)
        leading = tuple(image.shape[:-3])
        heat = _heatmap(heatmap, leading, self.device)
        pieces = []
        refs = self._cached_reference()
        for start in range(0, len(rgb), self.chunk):
            result = self._match(rgb[start:start+self.chunk], refs)
            pieces.append({key: value for key, value in result.items() if key != "similarities"})
        result = {key: torch.cat([p[key] for p in pieces]) for key in pieces[0]}
        p_region = torch.sigmoid(result["region_logits"] / self.temperature[0])
        p_visible = torch.sigmoid(result["visible_logits"] / self.temperature[1])
        p_interact = torch.sigmoid(result["interaction_logits"] / self.temperature[2])
        uncertainty = (result["ensemble"] / self.temperature[0]).sigmoid().std(dim=1, unbiased=False)
        return dict(result, leading=leading, heatmap=heat, probability=p_region,
                    visible_probability=p_visible, interaction_probability=p_interact,
                    region_uncertainty=uncertainty)

    def _supervision(self, data):
        device = self.device
        shape = tuple(data["image"].shape[:-3])
        rows = int(np.prod(shape))
        def get(key, suffix=(), dtype=torch.float32):
            if key not in data:
                return torch.zeros((rows, *suffix), dtype=dtype, device=device)
            return torch.as_tensor(data[key], device=device).reshape(rows, *suffix).to(dtype=dtype)
        def label_pair(value_key, valid_key, suffix=()):
            if value_key not in data or valid_key not in data:
                unknown = torch.zeros((rows, *suffix), dtype=torch.bool, device=device)
                return unknown, unknown
            value = get(value_key, suffix)
            marker = get(valid_key, suffix)
            valid = (marker == 1.) & ((value == 0.) | (value == 1.))
            if "goal_label_version" in data:
                version = get("goal_label_version").reshape(rows, *([1]*len(suffix)))
                valid &= version == VERSION
            return (value == 1.) & valid, valid

        labels, valid = label_pair("goal_region_gt", "goal_region_valid", (GRID, GRID))
        visible, visible_valid = label_pair("goal_visible_gt", "goal_visible_valid")
        interact, interact_valid = label_pair("goal_interaction_outcome", "goal_interaction_valid")
        # Inventory/success results arrive on the next observation. Predict
        # their immediate outcome from the source frame of an actual attempt.
        interaction_labels = torch.zeros((rows, 2), dtype=torch.bool, device=device)
        if (len(shape) == 2 and shape[-1] > 1 and "log_applied_control" in data
                and "log_actuator_valid" in data and "is_first" in data):
            applied = get("log_applied_control", (9,)).reshape(*shape, 9)[:, 1:]
            first = get("is_first").reshape(shape)[:, 1:]
            actual = get("log_actuator_valid").reshape(shape)[:, 1:]
            attempted = (applied[..., 0] > .5) | (applied[..., 6] > .5)
            good = interact_valid.reshape(shape)[:, 1:] & attempted & (first == 0.) & (actual == 1.)
            interaction_labels = interaction_labels.reshape(*shape, 2)
            interaction_labels[:, :-1, 0] = interact.reshape(shape)[:, 1:] & good
            interaction_labels[:, :-1, 1] = good
            interaction_labels = interaction_labels.reshape(rows, 2)
        frame_labels = torch.stack((torch.stack((visible, visible_valid), -1),
                                    interaction_labels), 1)
        return labels, valid, frame_labels, get("goal_holdout") == 1.

    def _loss(self, data, labels, valid, frame_labels, holdout):
        usable = (valid.flatten(1).any(-1) | frame_labels[..., 1].any(-1)) & ~holdout
        positive = labels.flatten(1).any(-1) | (frame_labels[..., 0] & frame_labels[..., 1]).any(-1)
        # Reserve room for backgrounds regardless of class imbalance. Rotate
        # within each class using the update counter without private RNG state.
        selected = []
        for mask in (usable & positive, usable & ~positive):
            indices = mask.nonzero().flatten()
            count = len(indices)
            if count:
                order = (torch.arange(count, device=self.device) + self.updates) % count
                selected.append(indices[order[:max(1, self.frame_limit//2)]])
        if not selected:
            return None, {"D/update": 0., "D/train_frames": 0.}
        indices = torch.cat(selected).sort().values
        images = self.observation_images(data)
        frames = images.reshape(-1, *images.shape[-3:])
        rgb = _rgb(frames[indices], self.device)
        references = self._reference()
        result = self._match(rgb, references)
        labels, valid = labels[indices], valid[indices]
        visible, interact = frame_labels[indices].unbind(1)
        raw = _heatmap(data["heatmap"], tuple(data["image"].shape[:-3]), self.device)
        hot = F.adaptive_avg_pool2d(raw[indices], (GRID, GRID)).squeeze(1)
        hard = torch.zeros_like(valid)
        if self.hard_negatives:
            eligible = valid & ~labels & (hot >= .5)
            difficulty = (hot + result["region_logits"].detach().sigmoid()).masked_fill(~eligible, -torch.inf)
            picks = difficulty.flatten(1).topk(4, dim=-1).indices
            hard = hard.flatten(1).scatter(1, picks, True).reshape_as(hard) & eligible
        from .goal_guidance import proposals
        candidate_idx, _, _ = proposals(raw[indices], result["saliency"].detach(),
                                        result["region_logits"].detach().sigmoid())
        candidate_valid = valid.flatten(1).gather(1, candidate_idx)
        candidate_label = labels.flatten(1).gather(1, candidate_idx)
        candidate_logits = result["region_logits"].flatten(1).gather(1, candidate_idx)
        loss_candidate = _balanced_bce(candidate_logits, candidate_label, candidate_valid)
        loss_region = .5 * sum(_balanced_bce(head, labels, valid, hard)
                               for head in result["ensemble"].unbind(1))
        loss_visible = _balanced_bce(result["visible_logits"], visible[:, 0], visible[:, 1])
        loss_interact = _balanced_bce(result["interaction_logits"], interact[:, 0], interact[:, 1])
        brier = (_masked_mean((result["region_logits"].sigmoid()-labels.float()).square(), valid)
                 + _masked_mean((result["visible_logits"].sigmoid()-visible[:, 0].float()).square(), visible[:, 1])
                 + _masked_mean((result["interaction_logits"].sigmoid()-interact[:, 0].float()).square(), interact[:, 1]))
        # InfoNCE contrasts real same-category patches against verified non-target
        # patches. No unlabeled pixels are treated as distractors.
        similarity = (result["similarities"] / .15).logsumexp(1) - np.log(len(self.bank_pixels))
        positive_logits = similarity[valid & labels]
        negative_logits = similarity[valid & ~labels]
        loss_reference_assets, reference_assets = self._reference_crop_loss(references)
        loss_negative_assets = similarity.new_zeros(())
        if self.hard_negatives and len(self.negative_pixels):
            assets = (torch.arange(min(4, len(self.negative_pixels)), device=self.device)
                      + self.updates).remainder(len(self.negative_pixels))
            crops = _rgb(self.negative_pixels[assets], self.device)
            crops = F.interpolate(crops, (self.input_size, self.input_size),
                                  mode="bilinear", align_corners=False)
            distractors = self._match(crops, references)
            center = GRID//2
            loss_negative_assets = F.softplus(distractors["ensemble"][:, :, center, center]).mean()
            neg = distractors["similarities"][:, :, center, center] / .15
            negative_logits = torch.cat((negative_logits, neg.logsumexp(1)-np.log(len(references))))
        contrast = similarity.new_zeros(())
        if len(positive_logits) and len(negative_logits):
            negative_mass = negative_logits.logsumexp(0) - np.log(len(negative_logits))
            contrast = F.softplus(negative_mass - positive_logits).mean()
        crossview, temporal, pairs = self._paired_loss(data, indices, result, labels, valid)
        loss = self.loss_weight * (loss_visible + loss_region + .5 * loss_interact
                                   + .5 * loss_candidate + .25 * (loss_negative_assets + loss_reference_assets)
                                   + .2 * contrast + .2 * crossview + .1 * temporal + .1 * brier)
        metrics = {"D/update": 1., "D/loss": loss, "D/region_loss": loss_region,
                   "D/visible_loss": loss_visible, "D/interact_loss": loss_interact,
                   "D/contrast_loss": contrast, "D/crossview_loss": crossview,
                   "D/candidate_loss": loss_candidate, "D/candidate_labels": candidate_valid.sum(),
                   "D/negative_asset_loss": loss_negative_assets,
                   "D/reference_asset_loss": loss_reference_assets, "D/reference_assets": reference_assets,
                   "D/temporal_loss": temporal, "D/brier_loss": brier,
                   "D/train_frames": len(indices), "D/region_labels": valid.sum(),
                   "D/positive_regions": (valid & labels).sum(), "D/visible_labels": visible[:, 1].sum(),
                   "D/interaction_labels": interact[:, 1].sum(),
                   "D/hard_negatives": hard.sum(), "D/crossview_pairs": pairs}
        return loss, metrics

    def _reference_crop_loss(self, references):
        """Use verified positive crops with other-world references and no pixel labels."""
        eligible = self.cross_world_references.any(1).nonzero().flatten()
        count = min(4, len(eligible))
        if not count:
            return references.new_zeros(()), 0
        indices = eligible[(torch.arange(count, device=self.device) + self.updates) % len(eligible)]
        crops = F.interpolate(_rgb(self.bank_pixels[indices], self.device),
                              (self.input_size, self.input_size),
                              mode="bilinear", align_corners=False)
        predictions = self._match(crops, references, self.cross_world_references[indices])
        # The annotation confirms the object somewhere in the crop. It does
        # not make every cell a positive or certify the source frame.
        crop_logits = predictions["ensemble"].flatten(2).topk(4, dim=2).values.mean(2)
        return F.softplus(-crop_logits).mean(), count

    def _paired_loss(self, data, indices, result, labels, valid):
        """Match observed regions sharing a train-only instance across real adjacent actions."""
        zeros = result["region_logits"].new_zeros(())
        shape = data["image"].shape[:-3]
        if (len(shape) != 2 or "goal_instance_group" not in data
                or "log_applied_control" not in data or "log_actuator_valid" not in data):
            return zeros, zeros, zeros
        device = indices.device
        eligible = (indices[1:] == indices[:-1]+1) & (indices[1:] % shape[-1] != 0)
        first = torch.as_tensor(data["is_first"], device=device).reshape(-1).bool()
        controls = torch.as_tensor(data["log_applied_control"], device=device).reshape(-1, 9)
        actual = torch.as_tensor(data["log_actuator_valid"], device=device).reshape(-1).bool()
        incoming = indices[1:]
        eligible &= ~first[incoming] & actual[incoming] & torch.isfinite(controls[incoming]).all(-1)
        group = torch.as_tensor(data["goal_instance_group"], device=device).reshape(-1, GRID*GRID)[indices]
        use = (valid & labels).flatten(1) & (group > 0)
        # Cap the sparse labeled positive cells at 16 to bound the pair search.
        region_idx = use.to(torch.int8).topk(16, dim=-1).indices
        group = group.gather(1, region_idx) * use.gather(1, region_idx)
        same = (group[:-1, :, None] == group[1:, None, :]) & (group[:-1, :, None] > 0)
        same &= eligible[:, None, None]
        embed = result["embeddings"].flatten(2).gather(
            2, region_idx[:, None, :].expand(-1, result["embeddings"].shape[1], -1))
        cosine = torch.einsum("nci,ncj->nij", embed[:-1], embed[1:])
        scores = result["region_logits"].sigmoid().flatten(1).gather(1, region_idx)
        # The temporal term follows real controls. Larger viewpoint changes
        # retain instance supervision but weaken the probability tie.
        angular = controls[incoming, -2:].abs().sum(-1).clamp_max(90.)
        weight = torch.exp(-angular/45.)[:, None, None]
        temporal = (scores[:-1, :, None]-scores[1:, None, :]).square()*weight
        return _masked_mean(1.-cosine, same), _masked_mean(temporal, same), same.sum()

    def loss(self, data):
        return self._loss(data, *self._supervision(data))

    @torch.no_grad()
    def _calibrate(self, data, labels, valid, frame_labels, holdout):
        if "vnext_frame_id" not in data:
            return
        ids = torch.as_tensor(data["vnext_frame_id"], device=self.device).reshape(-1).long()
        group = torch.full_like(ids, -1)
        if "log_vnext_world_group" in data:
            group = torch.as_tensor(data["log_vnext_world_group"], device=self.device).reshape(-1).long()
        independent = ~(group[:, None] == self.bank_world_groups).any(-1)
        eligible = (holdout & independent & (ids >= 0)
                    & (valid.flatten(1).any(-1) | frame_labels[..., 1].any(-1)))
        indices = eligible.nonzero().flatten()
        if not len(indices):
            # Recheck cached examples after training even if this batch contains
            # no new holdout frames. Do not retain stale probabilities.
            if not bool((self.cal_ids >= 0).any()):
                return
        else:
            # Hash collisions retain one frame, never multiply support on repeat.
            half = len(self.cal_ids)//2
            positive = (labels[indices] & valid[indices]).flatten(1).any(-1)
            slots = (ids[indices]*31 + ids[indices]//17).remainder(half) + positive.long()*half
            order = slots.argsort(stable=True)
            slots, indices = slots[order], indices[order]
            keep = torch.ones_like(slots, dtype=torch.bool)
            keep[:-1] = slots[:-1] != slots[1:]
            slots, indices = slots[keep], indices[keep]
            images = self.observation_images(data)
            rgb = images.reshape(-1, *images.shape[-3:])[indices]
            self.cal_ids[slots] = ids[indices]
            self.cal_world_groups[slots] = group[indices]
            self.cal_images[slots] = rgb
            self.cal_labels[slots] = torch.stack((labels[indices], valid[indices]), 1)
            self.cal_frame_labels[slots] = frame_labels[indices]
        known = self.cal_ids >= 0
        outputs = self._match(_rgb(self.cal_images[known], self.device), self._cached_reference())
        region, region_valid = self.cal_labels[known].unbind(1)
        frame = self.cal_frame_labels[known]
        for index, (pred, label, mask) in enumerate((
            (outputs["region_logits"], region, region_valid),
            (outputs["visible_logits"], frame[:, 0, 0], frame[:, 0, 1]),
            (outputs["interaction_logits"], frame[:, 1, 0], frame[:, 1, 1]),
        )):
            pos, neg = mask & label, mask & ~label
            pos_count = pos.reshape(len(pos), -1).any(-1).sum()
            neg_count = neg.reshape(len(neg), -1).any(-1).sum()
            self.cal_counts[index] = torch.stack((pos_count, neg_count))
            self.reliability[index] = 0.
            if not int(self.updates) or min(int(pos_count), int(neg_count)) < 4:
                continue
            # Fit temperature with balanced held-out NLL. Balanced Brier skill
            # over the uninformative prior controls how much fusion may change.
            choices = pred.new_tensor([.5, .75, 1., 1.5, 2., 3., 4.])
            plus = pred[pos][None] / choices[:, None]
            minus = pred[neg][None] / choices[:, None]
            best = (F.softplus(-plus).mean(-1) + F.softplus(minus).mean(-1)).argmin()
            self.temperature[index] = choices[best]
            score = .5*((1.-plus[best].sigmoid()).square().mean() + minus[best].sigmoid().square().mean())
            self.cal_brier[index] = score
            self.reliability[index] = (1.-score/.25).clamp(0., 1.)

    def train_batch(self, data):
        """A separate optimizer owns all gradients and refreshes reference cache."""
        supervision = self._supervision(data)
        parameters = list(self.parameters())
        before = [p.requires_grad for p in parameters]
        try:
            with torch.enable_grad():
                self.requires_grad_(True)
                loss, metrics = self._loss(data, *supervision)
                if loss is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(parameters, 10., error_if_nonfinite=True)
                    self.optimizer.step()
                    self.updates.add_(1)
                    self.reference_features = self.bank_pixels.new_empty(0, dtype=torch.float32)
                    metrics["D/grad_norm"] = norm
        finally:
            for param, flag in zip(parameters, before):
                param.requires_grad_(flag)
        self._calibrate(data, *supervision)
        metrics["D/updates"] = self.updates
        worlds = self.cal_world_groups[(self.cal_ids >= 0) & (self.cal_world_groups > 0)].unique()
        metrics["D/heldout_worlds"] = len(worlds)
        for index, name in enumerate(("region", "visible", "interact")):
            metrics[f"D/{name}_cal_reliability"] = self.reliability[index]
            metrics[f"D/{name}_cal_brier"] = self.cal_brier[index]
            metrics[f"D/{name}_cal_positive"] = self.cal_counts[index, 0]
            metrics[f"D/{name}_cal_negative"] = self.cal_counts[index, 1]
        return {key: float(value.detach()) if torch.is_tensor(value) else float(value)
                for key, value in metrics.items()}
