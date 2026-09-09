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
    def __init__(self, ckpt="weights/mineclip_attn.pth", **kwargs) -> None:
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
        # 仅用于调试：
        self.vision_forward_count = 0

    @th.no_grad()
    def forward_image_and_patch(
            self,
            curr_frame: th.Tensor,
            need_global: bool = True,
    ):
        """
        Args:
            curr_frame:
                [3,H,W] 或 [B,3,H,W]
                uint8, 0~255

        Returns:
            global_feat:
                [B,512]，need_global=False 时为 None
            patch_feat:
                [B,N,512]
            grid_size:
                (grid_h, grid_w)
        关键：
            ViT 前 11 层只运行一次；
            最后一个 block 的输入同时分出：
              1. 正常最后 block -> Global CLS
              2. MaskCLIP Value -> Patch
            不再额外调用 forward_image_features()。
        """

        if curr_frame.ndim == 3:
            curr_frame = curr_frame.unsqueeze(0)

        assert curr_frame.ndim == 4
        assert curr_frame.shape[1] == 3
        assert tuple(curr_frame.shape[-2:]) == self.resolution

        curr_frame = curr_frame.to(self.device)

        # 和 MineCLIP 官方 forward_image_features 完全相同的预处理
        frames = U.basic_image_tensor_preprocess(
            curr_frame,
            mean=MC_IMAGE_MEAN,
            std=MC_IMAGE_STD,
        )

        vit = self.model.clip_model.vision_model

        self.vision_forward_count += 1

        # ========================================================
        # Patch Embedding
        # [B,3,160,256]
        # ->
        # [B,768,10,16]
        # ========================================================
        x = vit.conv1(frames)

        B = x.shape[0]
        grid_h = x.shape[2]
        grid_w = x.shape[3]

        # [B,768,10,16]
        # ->
        # [B,160,768]
        x = x.reshape(B,x.shape[1],-1,).permute(0,2,1,)

        # ========================================================
        # CLS + Position
        # ========================================================

        cls_token = vit.cls_token.repeat(B,1,1,)

        x = th.cat([cls_token, x],dim=1,)

        x = x + vit.pos_embed
        x = vit.ln_pre(x)

        # [B,L,D] -> [L,B,D]
        x = x.permute(1,0,2,)

        # ========================================================
        # 前 11 个 Transformer Block
        # 只运行一次
        # ========================================================
        blocks = list(vit.blocks.children())

        for block in blocks[:-1]:
            x = block(x)

        # 当前 x 是最后一个 Transformer block 的输入
        last_block = blocks[-1]

        # ========================================================
        # 分支 1：
        # MaskCLIP Value Patch
        # ========================================================
        x_ln = last_block.ln_1(x)

        attn = last_block.attn
        embed_dim = x_ln.shape[-1]

        # nn.MultiheadAttention:
        # [Wq]
        # [Wk]
        # [Wv]
        v_weight = attn.in_proj_weight[
            2 * embed_dim:
            3 * embed_dim
        ]

        if attn.in_proj_bias is not None:
            v_bias = attn.in_proj_bias[
                2 * embed_dim:
                3 * embed_dim
            ]
        else:
            v_bias = None

        # Value projection
        value = F.linear(
            x_ln,
            v_weight,
            v_bias,
        )

        # MHA output projection
        value = F.linear(
            value,
            attn.out_proj.weight,
            attn.out_proj.bias,
        )

        # 删除 CLS
        #
        # [161,B,768]
        # ->
        # [B,160,768]
        patch_feat = value[1:].permute(1,0,2,)

        # MineCLIP 原视觉 projection
        #
        # 768 -> 512
        patch_feat = vit.ln_post(
            patch_feat
        )

        if vit.projection is not None:
            patch_feat = (
                    patch_feat
                    @ vit.projection
            )

        # ========================================================
        # 分支 2：
        # 原 MineCLIP Global CLS
        # ========================================================
        global_feat = None

        if need_global:
            # 最后一个完整 Transformer Block
            # 这里只运行一次
            x_global = last_block(x)

            x_global = x_global.permute(1,0,2,)

            global_feat = vit.ln_post(
                x_global[:, 0, :]
            )

            if vit.projection is not None:
                global_feat = (
                        global_feat
                        @ vit.projection
                )

        return (
            global_feat,
            patch_feat,
            (grid_h, grid_w),
        )

    @th.no_grad()
    def get_frame_bundle(
            self,
            obs: Dict,
    ):
        """
        一个 obs 在整个 wrapper 链中只编码一次。
        ClipWrapper 第一次调用：
            运行 ViT
        ConcentrationWrapper 再调用：
            直接读取缓存
            不再运行 ViT
        """

        if "_mineclip_bundle" in obs:
            return obs["_mineclip_bundle"]

        curr_frame = self._get_curr_frame(obs)

        global_feat, patch_feat, grid_size = (
            self.forward_image_and_patch(
                curr_frame,
                need_global=True,
            )
        )

        bundle = {
            "global_feat": global_feat,
            "patch_feat": patch_feat,
            "grid_size": grid_size,
        }

        # 只在 wrapper 内部临时存在。
        # 外层 LSImagineWrapper 会重新构造 obs，
        # 不会进入 replay buffer。
        obs["_mineclip_bundle"] = bundle

        return bundle

    @th.no_grad()
    def get_text_feats_cached(
            self,
            prompts,
    ):
        """
        每个字符串只运行一次 Text Transformer。
        """
        prompts = list(prompts)

        missing = [
            p
            for p in prompts
            if p not in self._text_cache
        ]

        if len(missing) > 0:
            new_feats = self.model.encode_text(missing)
            for prompt, feat in zip(
                    missing,
                    new_feats,
            ):
                self._text_cache[prompt] = (
                    feat.detach()
                )

        return th.stack(
            [
                self._text_cache[p]
                for p in prompts
            ],
            dim=0,
        )

    @th.no_grad()
    def get_video_feat_from_global(
            self,
            curr_global_feat,
            past_frames=None,
    ):
        """
        让 Global Temporal 也只执行一次
        curr_global_feat:
            [1,512]
        past_frames:
            [15,512]

        Returns:
            video_feat:
                [1,512]
            new_past:
                [15,512]
        """
        assert curr_global_feat.shape == (1,512,)

        if past_frames is None:
            past_frames = (
                curr_global_feat
                .new_zeros(1,15,512,)
            )

        else:
            past_frames = (
                past_frames
                .to(self.device)
                .unsqueeze(0)
            )

        current = curr_global_feat.unsqueeze(1)
        image_feats = th.cat([past_frames,current,],dim=1,)

        assert image_feats.shape == (1,16,512,)

        video_feat = (self.model.forward_video_features(image_feats))

        new_past = (image_feats[0,1:].detach())

        return video_feat, new_past

    @th.no_grad()
    def get_logits_from_video(
            self,
            video_feat,
            prompts,
    ):
        text_feats = self.get_text_feats_cached(prompts)

        logits = (
            self.model.forward_reward_head(
                video_feat,
                text_tokens=text_feats,
            )[0][0]
        )

        return logits

    @th.no_grad()
    def get_aligned_patch(
            self,
            bundle,
    ):
        """
        Patch
        -> MineCLIP Temporal
        -> aligned Patch
        同一个 bundle 只计算一次。
        """
        if "aligned_patch_feat" in bundle:
            return bundle["aligned_patch_feat"]

        patch_feat = bundle["patch_feat"]

        B, N, D = patch_feat.shape

        patch_sequence = (patch_feat.reshape(B * N,1,D,))

        aligned_patch = (self.model.forward_video_features(patch_sequence))
        aligned_patch = (aligned_patch.reshape(B,N,-1,))

        bundle["aligned_patch_feat"] = aligned_patch

        return aligned_patch

    @th.no_grad()
    def get_patch_similarity(
            self,
            bundle,
            prompts,
    ):
        """
        Returns:
            [B,P,N]
        """
        patch_feat = self.get_aligned_patch(bundle)

        text_feat = (
            self.get_text_feats_cached(prompts)
        )

        patch_norm = F.normalize(patch_feat,dim=-1,)
        text_norm = F.normalize(text_feat,dim=-1,)

        similarity = th.einsum(
            "bnd,pd->bpn",
            patch_norm,
            text_norm,
        )

        return similarity

    @th.no_grad()
    def make_bundle_from_frame(
            self,
            curr_frame,
            need_global=False,
    ):

        global_feat, patch_feat, grid_size = (
            self.forward_image_and_patch(
                curr_frame,
                need_global=need_global,
            )
        )

        return {
            "global_feat": global_feat,
            "patch_feat": patch_feat,
            "grid_size": grid_size,
        }

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
