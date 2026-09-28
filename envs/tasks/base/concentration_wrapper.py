from gym import Wrapper
import numpy as np

class ConcentrationWrapper(Wrapper):
    """
    Keep the original LS-Imagine observation dictionary unchanged, but change
    the *meaning* of the two ScoreStorage fields:

        obs['score']
            = current NACLIP + Temporal(L=1) raw P85

        obs['score_on_zoomed']
            = zoomed NACLIP + Temporal(L=1) raw P85

    Natural-Zoom heatmap and intrinsic shaping remain separate from these two
    progress fields.
    """

    def __init__(
            self,
            env,
            concentration,
            prompts=None,
            dense_reward=0.01,
            mineclip_dense_reward=0.01,
            max_steps=1000,
            gaussian_reward_weight=1.0,
            wm_heatmap_mode="spatial",
            **kwargs,
    ):
        super().__init__(env)
        self.concentration = concentration
        self.wrapper_name = "ConcentrationWrapper"

        assert prompts is not None
        self.prompt = prompts
        self.dense_reward = dense_reward
        self.mineclip_dense_reward = mineclip_dense_reward
        self.gaussian_reward_weight = gaussian_reward_weight

        # ------------------------------------------------------------
        # Role-Decoupled WM heatmap input.
        #
        # spatial:
        #     原来的完整二维 relevance map -> World Model
        #
        # frame_mean:
        #     Natural Zoom 仍使用完整二维 relevance map；
        #     只有送给 World Model 的 heatmap 被变成整帧均值。
        # ------------------------------------------------------------
        self.wm_heatmap_mode = str(wm_heatmap_mode).lower()

        assert self.wm_heatmap_mode in ("spatial", "frame_mean"), (
            f"Unsupported wm_heatmap_mode={self.wm_heatmap_mode}"
        )

        self.episode = 0
        self.steps = 0

        # This is still the last best Gaussian/map shaping score.
        # It is NOT the ScoreStorage progress score.
        self.last_score = 0

        self.last_zoom_in_mineclip_score = 0
        self.last_zoom_in_gaussian_score = 0

        self.max_steps = max_steps

    def _to_world_model_heatmap(self, spatial_heatmap):
        """
        只转换送给 World Model / replay 的 heatmap。

        Natural Zoom、Gaussian shaping、zoom acceptance
        在调用这里之前已经使用了完整 spatial heatmap，
        因此不会受 frame_mean 影响。
        """
        heatmap = np.asarray(
            spatial_heatmap,
            dtype=np.float32
        )

        if self.wm_heatmap_mode == "spatial":
            return heatmap.copy()

        # frame_mean:
        # 保留每一帧整体 relevance 强度，
        # 删除 heatmap 的二维空间位置。
        frame_mean = np.float32(
            np.mean(
                heatmap,
                dtype=np.float64
            )
        )

        if self.steps < 3:
            print(
                "[ROLE-DECOUPLED] "
                f"mode={self.wm_heatmap_mode}, "
                f"input_mean={heatmap.mean():.6f}, "
                f"input_std={heatmap.std():.6f}, "
                f"wm_mean={frame_mean:.6f}, "
                f"wm_std=0.000000"
            )

        return np.full(
            heatmap.shape,
            frame_mean,
            dtype=np.float32
        )

    def reset(self, **kwargs):
        self.episode += 1
        self.steps = 0

        self.last_score = 0
        self.last_zoom_in_mineclip_score = 0
        self.last_zoom_in_gaussian_score = 0

        obs = self.env.reset(**kwargs)

        # ------------------------------------------------------------
        # Current frame:
        #   map      = Value+Temporal(L=1) -> (cos+1)/2
        #   progress = NACLIP+Temporal(L=1) -> raw P85
        # ------------------------------------------------------------
        gaussian_score, zoom_in_prob, check_threshold = (
            self.concentration.get_reward(
                obs,
                self.prompt,
                self.episode,
                self.steps,
            )
        )

        current_progress = self.concentration.get_progress_score(
            is_zoomed=False
        )

        zoomed_image, is_check = (
            self.concentration.generate_zoom_in_frame()
        )

        if is_check:
            (
                zoomed_reward,
                gaussian_on_zoomed,
                zoom_in_prob_on_zoomed,
                is_zoomed,
                jump,
            ) = self.concentration.compute_reward_on_zoomed_image()

            zoomed_progress = self.concentration.get_progress_score(
                is_zoomed=True
            )
        else:
            zoomed_reward = 0.0
            gaussian_on_zoomed = 0.0
            zoom_in_prob_on_zoomed = 0.0
            zoomed_progress = 0.0
            is_zoomed = False
            jump = False

        # ------------------------------------------------------------
        # Keep all public observation keys unchanged.
        # ------------------------------------------------------------
        obs['is_zoomed'] = is_zoomed
        obs['jump'] = jump
        obs['jumping_steps'] = self.max_steps
        obs['accumulated_reward'] = 0.0
        obs['is_calculated'] = False
        obs['reward_on_zoomed'] = 0.0
        obs['intrinsic_on_zoomed'] = 0.0
        obs['zoomed_image'] = zoomed_image

        # ------------------------------------------------------------
        # Keep the old Gaussian/map intrinsic shaping.
        # This is deliberately decoupled from ScoreStorage.
        # ------------------------------------------------------------
        if gaussian_score > self.last_score:
            obs['intrinsic'] += (
                self.dense_reward
                * gaussian_score
                * self.gaussian_reward_weight
            )
            self.last_score = gaussian_score

        if is_zoomed:
            if (
                gaussian_on_zoomed > self.last_score
                and gaussian_on_zoomed > self.last_zoom_in_gaussian_score
            ):
                obs['intrinsic_on_zoomed'] += (
                    self.dense_reward
                    * gaussian_on_zoomed
                    * self.gaussian_reward_weight
                )
                self.last_zoom_in_gaussian_score = gaussian_on_zoomed

            # Preserve the previous zoomed-map intrinsic shaping quantity.
            if zoomed_reward > self.last_zoom_in_mineclip_score:
                obs['intrinsic_on_zoomed'] += (
                    self.mineclip_dense_reward
                    * zoomed_reward
                )
                self.last_zoom_in_mineclip_score = zoomed_reward

        # ------------------------------------------------------------
        # IMPORTANT: ScoreStorage now sees one symmetric metric only.
        # Do NOT multiply these values by dense_reward.
        # Do NOT add Gaussian / global-CLIP values here.
        # ------------------------------------------------------------
        obs['score'] = float(current_progress)
        obs['score_on_zoomed'] = (
            float(zoomed_progress)
            if is_zoomed
            else 0.0
        )

        # ------------------------------------------------------------
        # Role-Decoupled World-Model heatmap.
        #
        # 到这里之前：
        #   Natural Zoom 已经使用了完整 spatial map。
        #   P85 已经计算完成。
        #   Gaussian/intrinsic shaping 也已经计算完成。
        #
        # 这里只改变进入 replay / World Model 的 heatmap。
        # ------------------------------------------------------------
        current_spatial_heatmap = self.concentration.get_heatmap(
            is_zoomed=False
        )

        obs['heatmap'] = self._to_world_model_heatmap(
            current_spatial_heatmap
        )

        if is_zoomed:
            zoomed_spatial_heatmap = self.concentration.get_heatmap(
                is_zoomed=True
            )

            obs['heatmap_on_zoomed'] = self._to_world_model_heatmap(
                zoomed_spatial_heatmap
            )
        else:
            obs['heatmap_on_zoomed'] = obs['heatmap'].copy()

        return obs

    def step(self, action):
        self.steps += 1
        obs, reward, done, info = self.env.step(action)

        if len(self.prompt) > 0:
            (
                gaussian_score,
                zoom_in_prob,
                check_threshold,
            ) = self.concentration.get_reward(
                obs,
                self.prompt,
                self.episode,
                self.steps,
            )

            current_progress = self.concentration.get_progress_score(
                is_zoomed=False
            )

            zoomed_image, is_check = (
                self.concentration.generate_zoom_in_frame()
            )

            if is_check:
                (
                    zoomed_reward,
                    gaussian_on_zoomed,
                    zoom_in_prob_on_zoomed,
                    is_zoomed,
                    jump,
                ) = self.concentration.compute_reward_on_zoomed_image()

                zoomed_progress = self.concentration.get_progress_score(
                    is_zoomed=True
                )
            else:
                zoomed_reward = 0.0
                gaussian_on_zoomed = 0.0
                zoom_in_prob_on_zoomed = 0.0
                zoomed_progress = 0.0
                is_zoomed = False
                jump = False

            obs['is_zoomed'] = is_zoomed
            obs['jump'] = jump
            obs['jumping_steps'] = self.max_steps
            obs['accumulated_reward'] = 0.0
            obs['is_calculated'] = False
            obs['reward_on_zoomed'] = reward
            obs['intrinsic_on_zoomed'] = 0.0
            obs['zoomed_image'] = zoomed_image

            # --------------------------------------------------------
            # Existing current-map intrinsic shaping.
            # --------------------------------------------------------
            if gaussian_score > self.last_score:
                obs['intrinsic'] += (
                    self.dense_reward
                    * gaussian_score
                    * self.gaussian_reward_weight
                )
                self.last_score = gaussian_score

            # --------------------------------------------------------
            # Existing zoomed intrinsic shaping is retained.
            # It no longer contributes to score_on_zoomed.
            # --------------------------------------------------------
            if is_zoomed:
                if (
                    gaussian_on_zoomed > self.last_score
                    and gaussian_on_zoomed > self.last_zoom_in_gaussian_score
                ):
                    obs['intrinsic_on_zoomed'] += (
                        self.dense_reward
                        * gaussian_on_zoomed
                        * self.gaussian_reward_weight
                    )
                    self.last_zoom_in_gaussian_score = gaussian_on_zoomed

                if (
                    zoomed_reward > info["clip_last_score"]
                    and zoomed_reward > self.last_zoom_in_mineclip_score
                ):
                    self.mineclip_dense_reward = info["clip_dense_reward"]
                    obs['intrinsic_on_zoomed'] += (
                        self.mineclip_dense_reward
                        * zoomed_reward
                    )
                    self.last_zoom_in_mineclip_score = zoomed_reward

            # --------------------------------------------------------
            # ScoreStorage fields: symmetric NACLIP-P85 only.
            # --------------------------------------------------------
            obs['score'] = float(current_progress)
            obs['score_on_zoomed'] = (
                float(zoomed_progress)
                if is_zoomed
                else 0.0
            )

            # --------------------------------------------------------
            # World-model heatmaps: parameter-free raw-shifted map.
            # --------------------------------------------------------
            current_spatial_heatmap = self.concentration.get_heatmap(
                is_zoomed=False
            )

            obs['heatmap'] = self._to_world_model_heatmap(
                current_spatial_heatmap
            )

            if is_zoomed:
                zoomed_spatial_heatmap = self.concentration.get_heatmap(
                    is_zoomed=True
                )

                obs['heatmap_on_zoomed'] = self._to_world_model_heatmap(
                    zoomed_spatial_heatmap
                )
            else:
                obs['heatmap_on_zoomed'] = obs['heatmap'].copy()

        return obs, reward, done, info
