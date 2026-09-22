from typing import List, Tuple, Dict
import torch as th
from omegaconf import OmegaConf
from mineclip import MineCLIP
from abc import ABC, abstractstaticmethod

import torch.nn.functional as F

import mineclip.utils as U
from mineclip.mineclip.base import (
    MC_IMAGE_MEAN,
    MC_IMAGE_STD,
)


class ClipReward(ABC):
    """
    Shared MineCLIP feature extractor used by both ClipWrapper and
    ConcentrationWrapper.

    Production spatial paths:

      Value patch -> Temporal(L=1) -> cosine
          used for the parameter-free relevance map

      NACLIP(K-K + Gaussian) -> Temporal(L=1) -> cosine -> P85
          used for ScoreStorage progress

    The expensive ViT stem + first 11 transformer blocks are executed only
    once for a normal observation. The last-block input is cached temporarily
    inside ``obs['_mineclip_bundle']`` and is dropped by LSImagineWrapper
    before replay storage.
    """

    def __init__(self, ckpt="weights/mineclip_attn.pth", **kwargs) -> None:
        # ------------------------------------------------------------
        # NACLIP parameters are global representation parameters.
        # They are intentionally NOT task-specific calibration terms.
        # ------------------------------------------------------------
        self.naclip_gaussian_std = float(
            kwargs.pop("naclip_gaussian_std", 5.0)
        )
        self.naclip_gaussian_weight = float(
            kwargs.pop("naclip_gaussian_weight", 1.0)
        )
        self.naclip_include_cls = bool(
            kwargs.pop("naclip_include_cls", True)
        )

        kwargs["arch"] = kwargs.pop("arch", "vit_base_p16_fz.v2.t2")
        kwargs["hidden_dim"] = kwargs.pop("hidden_dim", 512)
        kwargs["image_feature_dim"] = kwargs.pop("image_feature_dim", 512)
        kwargs["mlp_adapter_spec"] = kwargs.pop("mlp_adapter_spec", "v0-2.t0")
        kwargs["pool_type"] = kwargs.pop("pool_type", "attn.d2.nh8.glusw")
        kwargs["resolution"] = [160, 256]

        self.resolution = self.get_resolution()
        self.device = kwargs.pop("device", "cuda")
        self.model = None

        self._load_mineclip(ckpt, kwargs)

        self._text_cache = {}
        self._naclip_gaussian_cache = {}

        # Debug counter: one increment == one MineCLIP vision encoding.
        self.vision_forward_count = 0

    # ================================================================
    # Shared vision encoding
    # ================================================================

    @th.no_grad()
    def _encode_to_last_block_input(
            self,
            curr_frame: th.Tensor,
    ):
        """
        Run MineCLIP vision up to (but not including) the final ViT block.

        Returns:
            x:
                [L,B,768], input to the final transformer block
            last_block:
                final MineCLIP ViT block
            grid_size:
                (10,16) for 160x256 / patch16
        """
        if curr_frame.ndim == 3:
            curr_frame = curr_frame.unsqueeze(0)

        assert curr_frame.ndim == 4
        assert curr_frame.shape[1] == 3
        assert tuple(curr_frame.shape[-2:]) == self.resolution, (
            curr_frame.shape,
            self.resolution,
        )

        curr_frame = curr_frame.to(self.device)

        frames = U.basic_image_tensor_preprocess(
            curr_frame,
            mean=MC_IMAGE_MEAN,
            std=MC_IMAGE_STD,
        )

        vit = self.model.clip_model.vision_model
        self.vision_forward_count += 1

        x = vit.conv1(frames)
        B, _, grid_h, grid_w = x.shape

        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)
        x = th.cat([vit.cls_token.repeat(B, 1, 1), x], dim=1)
        x = vit.ln_pre(x + vit.pos_embed)
        x = x.permute(1, 0, 2)  # [L,B,D]

        blocks = list(vit.blocks.children())
        for block in blocks[:-1]:
            x = block(x)

        return x, blocks[-1], (int(grid_h), int(grid_w))

    @th.no_grad()
    def _value_patch_from_last_input(
            self,
            x,
            last_block,
    ):
        """MaskCLIP-style Value projection from final-block input."""
        vit = self.model.clip_model.vision_model
        x_ln = last_block.ln_1(x)
        attn = last_block.attn
        embed_dim = x_ln.shape[-1]

        v_weight = attn.in_proj_weight[
            2 * embed_dim:
            3 * embed_dim
        ]
        v_bias = (
            None
            if attn.in_proj_bias is None
            else attn.in_proj_bias[2 * embed_dim:3 * embed_dim]
        )

        value = F.linear(x_ln, v_weight, v_bias)
        value = F.linear(
            value,
            attn.out_proj.weight,
            attn.out_proj.bias,
        )

        patch_feat = value[1:].permute(1, 0, 2)  # remove CLS
        patch_feat = vit.ln_post(patch_feat)

        if vit.projection is not None:
            patch_feat = patch_feat @ vit.projection

        return patch_feat

    @th.no_grad()
    def _global_from_last_input(
            self,
            x,
            last_block,
    ):
        """Original MineCLIP global CLS path."""
        vit = self.model.clip_model.vision_model
        x_global = last_block(x).permute(1, 0, 2)
        global_feat = vit.ln_post(x_global[:, 0, :])

        if vit.projection is not None:
            global_feat = global_feat @ vit.projection

        return global_feat

    @th.no_grad()
    def _make_bundle_from_tensor(
            self,
            curr_frame,
            need_global=True,
    ):
        """
        Build a temporary feature bundle from one RGB frame.

        ``_last_block_input`` is kept only so NACLIP can be computed lazily
        without a second ViT pass. LSImagineWrapper later reconstructs the
        public observation dictionary, so this tensor never enters replay.
        """
        x, last_block, grid_size = self._encode_to_last_block_input(
            curr_frame
        )

        patch_feat = self._value_patch_from_last_input(
            x,
            last_block,
        )

        global_feat = (
            self._global_from_last_input(x, last_block)
            if need_global
            else None
        )

        return {
            "global_feat": global_feat,
            "patch_feat": patch_feat,
            "grid_size": grid_size,
            "_last_block_input": x.detach(),
        }

    @th.no_grad()
    def forward_image_and_patch(
            self,
            curr_frame: th.Tensor,
            need_global: bool = True,
    ):
        """
        Backward-compatible public helper.

        Returns exactly the same 3 values as the previous implementation:
            global_feat, Value patch_feat, grid_size
        """
        bundle = self._make_bundle_from_tensor(
            curr_frame,
            need_global=need_global,
        )

        return (
            bundle["global_feat"],
            bundle["patch_feat"],
            bundle["grid_size"],
        )

    @th.no_grad()
    def get_frame_bundle(
            self,
            obs: Dict,
    ):
        """
        One normal observation is vision-encoded only once across wrappers.
        """
        if "_mineclip_bundle" in obs:
            return obs["_mineclip_bundle"]

        curr_frame = self._get_curr_frame(obs)
        bundle = self._make_bundle_from_tensor(
            curr_frame,
            need_global=True,
        )

        obs["_mineclip_bundle"] = bundle
        return bundle

    # ================================================================
    # Text / global temporal
    # ================================================================

    @th.no_grad()
    def get_text_feats_cached(
            self,
            prompts,
    ):
        prompts = list(prompts)

        missing = [
            p
            for p in prompts
            if p not in self._text_cache
        ]

        if len(missing) > 0:
            new_feats = self.model.encode_text(missing)
            for prompt, feat in zip(missing, new_feats):
                self._text_cache[prompt] = feat.detach()

        return th.stack(
            [self._text_cache[p] for p in prompts],
            dim=0,
        )

    @th.no_grad()
    def get_video_feat_from_global(
            self,
            curr_global_feat,
            past_frames=None,
    ):
        """Original 16-frame MineCLIP temporal path for global reward."""
        assert curr_global_feat.shape == (1, 512)

        if past_frames is None:
            past_frames = curr_global_feat.new_zeros(1, 15, 512)
        else:
            past_frames = past_frames.to(self.device).unsqueeze(0)

        current = curr_global_feat.unsqueeze(1)
        image_feats = th.cat([past_frames, current], dim=1)
        assert image_feats.shape == (1, 16, 512)

        video_feat = self.model.forward_video_features(image_feats)
        new_past = image_feats[0, 1:].detach()

        return video_feat, new_past

    @th.no_grad()
    def get_logits_from_video(
            self,
            video_feat,
            prompts,
    ):
        text_feats = self.get_text_feats_cached(prompts)

        return self.model.forward_reward_head(
            video_feat,
            text_tokens=text_feats,
        )[0][0]

    # ================================================================
    # Value + Temporal(L=1): relevance map branch
    # ================================================================

    @th.no_grad()
    def _temporal_l1(
            self,
            patch_feat,
    ):
        """
        Apply MineCLIP temporal adapter to each patch independently with L=1.
        This is the same adapter used in the validated offline experiments.
        """
        B, N, D = patch_feat.shape
        patch_sequence = patch_feat.reshape(B * N, 1, D)
        aligned = self.model.forward_video_features(patch_sequence)
        return aligned.reshape(B, N, -1)

    @th.no_grad()
    def get_aligned_patch(
            self,
            bundle,
    ):
        if "aligned_patch_feat" not in bundle:
            bundle["aligned_patch_feat"] = self._temporal_l1(
                bundle["patch_feat"]
            )

        return bundle["aligned_patch_feat"]

    @th.no_grad()
    def _cosine_similarity(
            self,
            patch_feat,
            prompts,
    ):
        text_feat = self.get_text_feats_cached(prompts)
        patch_norm = F.normalize(patch_feat, dim=-1)
        text_norm = F.normalize(text_feat, dim=-1)

        return th.einsum(
            "bnd,pd->bpn",
            patch_norm,
            text_norm,
        )

    @th.no_grad()
    def get_patch_similarity(
            self,
            bundle,
            prompts,
    ):
        """
        Value + Temporal(L=1) raw cosine similarity.

        Returns:
            [B,P,N]
        """
        return self._cosine_similarity(
            self.get_aligned_patch(bundle),
            prompts,
        )

    # ================================================================
    # NACLIP + Temporal(L=1): ScoreStorage branch
    # ================================================================

    def _get_naclip_gaussian(
            self,
            grid_h,
            grid_w,
            dtype,
            device,
            include_cls,
    ):
        key = (
            int(grid_h),
            int(grid_w),
            bool(include_cls),
            str(dtype),
            str(device),
        )

        if key in self._naclip_gaussian_cache:
            return self._naclip_gaussian_cache[key]

        ys, xs = th.meshgrid(
            th.arange(
                grid_h,
                device=device,
                dtype=th.float32,
            ),
            th.arange(
                grid_w,
                device=device,
                dtype=th.float32,
            ),
            indexing="ij",
        )

        coords = th.stack(
            [ys.reshape(-1), xs.reshape(-1)],
            dim=1,
        )

        delta = coords[:, None, :] - coords[None, :, :]
        dist2 = (delta * delta).sum(dim=-1)

        patch_gaussian = th.exp(
            -dist2
            / (
                2.0
                * self.naclip_gaussian_std
                ** 2
            )
        )

        if include_cls:
            N = grid_h * grid_w
            gaussian = th.zeros(
                (N + 1, N + 1),
                device=device,
                dtype=th.float32,
            )
            gaussian[1:, 1:] = patch_gaussian
        else:
            gaussian = patch_gaussian

        gaussian = gaussian.to(dtype=dtype)
        self._naclip_gaussian_cache[key] = gaussian
        return gaussian

    @th.no_grad()
    def get_naclip_patch(
            self,
            bundle,
    ):
        """
        NACLIP reduced final block used in our validated experiments:

            softmax(K K^T / sqrt(d_head) + Gaussian) @ V
            -> out_proj -> ln_post -> vision projection

        No residual / FFN is applied in this NACLIP branch.
        """
        if "naclip_patch_feat" in bundle:
            return bundle["naclip_patch_feat"]

        x = bundle["_last_block_input"]
        grid_h, grid_w = bundle["grid_size"]

        vit = self.model.clip_model.vision_model
        last_block = list(vit.blocks.children())[-1]
        z = last_block.ln_1(x)
        attn = last_block.attn

        L, B, D = z.shape
        num_heads = attn.num_heads
        head_dim = D // num_heads
        scale = head_dim ** -0.5

        # q is intentionally unused; this matches the tested NACLIP branch.
        _, k, v = F.linear(
            z,
            attn.in_proj_weight,
            attn.in_proj_bias,
        ).chunk(3, dim=-1)

        k = (
            k.contiguous()
            .view(L, B * num_heads, head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(L, B * num_heads, head_dim)
            .transpose(0, 1)
        )

        if self.naclip_include_cls:
            weights = th.bmm(
                k,
                k.transpose(1, 2),
            ) * scale

            gaussian = self._get_naclip_gaussian(
                grid_h,
                grid_w,
                weights.dtype,
                weights.device,
                include_cls=True,
            )

            weights = F.softmax(
                weights
                + self.naclip_gaussian_weight
                * gaussian.unsqueeze(0),
                dim=-1,
            )

            out = th.bmm(weights, v)
            out = (
                out.transpose(0, 1)
                .contiguous()
                .view(L, B, D)
            )

            out = attn.out_proj(out).permute(1, 0, 2)
            out = vit.ln_post(out)

            if vit.projection is not None:
                out = out @ vit.projection

            patch = out[:, 1:, :]

        else:
            kp = k[:, 1:, :]
            vp = v[:, 1:, :]

            weights = th.bmm(
                kp,
                kp.transpose(1, 2),
            ) * scale

            gaussian = self._get_naclip_gaussian(
                grid_h,
                grid_w,
                weights.dtype,
                weights.device,
                include_cls=False,
            )

            weights = F.softmax(
                weights
                + self.naclip_gaussian_weight
                * gaussian.unsqueeze(0),
                dim=-1,
            )

            N = grid_h * grid_w
            out = th.bmm(weights, vp)
            out = (
                out.transpose(0, 1)
                .contiguous()
                .view(N, B, D)
            )

            out = attn.out_proj(out).permute(1, 0, 2)
            out = vit.ln_post(out)

            if vit.projection is not None:
                out = out @ vit.projection

            patch = out

        bundle["naclip_patch_feat"] = patch
        return patch

    @th.no_grad()
    def get_naclip_aligned_patch(
            self,
            bundle,
    ):
        if "naclip_aligned_patch_feat" not in bundle:
            bundle["naclip_aligned_patch_feat"] = self._temporal_l1(
                self.get_naclip_patch(bundle)
            )

        return bundle["naclip_aligned_patch_feat"]

    @th.no_grad()
    def get_naclip_patch_similarity(
            self,
            bundle,
            prompts,
    ):
        """Return NACLIP+Temporal(L=1) raw cosine, shape [B,P,N]."""
        return self._cosine_similarity(
            self.get_naclip_aligned_patch(bundle),
            prompts,
        )

    @th.no_grad()
    def get_naclip_progress_score(
            self,
            bundle,
            prompts,
            percentile=85.0,
    ):
        """
        Symmetric ScoreStorage metric.

        For multiple prompts:
            max prompt cosine for each patch
            -> percentile across patches

        Returns:
            [B]
        """
        percentile = float(percentile)
        assert 0.0 <= percentile <= 100.0

        similarity = self.get_naclip_patch_similarity(
            bundle,
            prompts,
        )

        per_patch = th.max(
            similarity,
            dim=1,
        ).values

        return th.quantile(
            per_patch,
            q=percentile / 100.0,
            dim=-1,
        )

    @th.no_grad()
    def make_bundle_from_frame(
            self,
            curr_frame,
            need_global=False,
    ):
        """Build an uncached bundle for a genuinely new RGB frame."""
        return self._make_bundle_from_tensor(
            curr_frame,
            need_global=need_global,
        )

    @abstractstaticmethod
    def get_resolution():
        raise NotImplementedError()

    @abstractstaticmethod
    def _get_curr_frame(obs):
        raise NotImplementedError()

    def _load_mineclip(self, ckpt, config):
        config = OmegaConf.create(config)
        self.model = MineCLIP(**config).to(self.device)
        self.model.load_ckpt(ckpt, strict=True)
        if self.resolution != (160, 256):  # Not ideal, but we need to resize the relative position embedding
            self.model.clip_model.vision_model._resolution = th.tensor([160, 256])  # This isn't updated from when mineclip resized it
            self.model.clip_model.vision_model.resize_pos_embed(self.resolution)
        self.model.eval()

    def _get_reward_from_logits(
        self,
        logits: th.Tensor  # P
    ) -> float:
        probs = th.softmax(logits, 0)
        return max(probs[0].item() - 1 / logits.shape[0], 0)

    def _get_image_feats(
        self,
        curr_frame: th.Tensor,
        past_frames: th.Tensor = None
    ) -> th.Tensor:
        while len(curr_frame.shape) < 5:
            curr_frame = curr_frame.unsqueeze(0)
        assert curr_frame.shape == (1, 1, 3) + self.resolution, "Found shape {}".format(curr_frame.shape)
        curr_frame_feats = self.model.forward_image_features(curr_frame.to(self.device))  # 1 x 1 x 512

        if past_frames is None:
            past_frames = th.zeros((15, curr_frame_feats.shape[-1]))
        past_frames = past_frames.to(self.device)

        while len(past_frames.shape) < 3:
            past_frames = past_frames.unsqueeze(0)
        assert past_frames.shape == (1, 15, curr_frame_feats.shape[-1]), "Found shape {}".format(past_frames.shape)

        return th.cat((past_frames, curr_frame_feats), dim=1)

    def _get_video_feats(
        self,
        image_feats: th.Tensor
    ) -> th.Tensor:
        return self.model.forward_video_features(image_feats.to(self.device))  # 1 x 512

    def _get_text_feats(
        self,
        prompts: str
    ) -> th.Tensor:
        text_feats = self.model.encode_text(prompts)  # P x 512
        assert len(text_feats.shape) == 2 and text_feats.shape[0] == len(prompts), "Found shape {}".format(text_feats.shape)
        return text_feats

    def get_logits(
        self,
        obs: Dict,  # 3 x 160 x 256
        prompts: List[str],
        state: Tuple[th.Tensor, th.Tensor] = None  # history x 512
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        curr_frame = self._get_curr_frame(obs)
        past_frames, text_feats = state

        with th.no_grad():
            if text_feats is None:
                text_feats = self._get_text_feats(prompts)

            image_feats = self._get_image_feats(curr_frame, past_frames)
            video_feats = self._get_video_feats(image_feats)
            logits = self.model.forward_reward_head(video_feats.to(self.device), text_tokens=text_feats.to(self.device))[0][0]  # P

        return logits, (image_feats[0, 1:].cpu(), text_feats.cpu())

    def get_reward(
        self,
        obs: Dict,
        prompt: str,
        neg_prompts: List[str],
        state: Tuple[th.Tensor, th.Tensor] = None  # history x 512
    ) -> Tuple[float, Tuple[th.Tensor, th.Tensor]]:
        logits, state = self.get_logits(
            obs,
            [prompt] + neg_prompts,
            state
        )
        reward = self._get_reward_from_logits(logits)

        return reward, state

    def get_rewards(
        self,
        obs: Dict,
        prompts: List[str],
        neg_prompts: List[str],
        state: Tuple[th.Tensor, th.Tensor] = None  # history x 512
    ) -> Tuple[List[float], Tuple[th.Tensor, th.Tensor]]:
        logits, state = self.get_logits(
            obs,
            prompts + neg_prompts,
            state
        )
        rewards = []
        for i in range(len(prompts)):
            rewards.append(self._get_reward_from_logits(th.cat((
                logits[i:i+1],
                logits[len(prompts):]
            ))))
        return rewards, state
