from gym import Wrapper
import torch as th


class ClipWrapper(Wrapper):
    def __init__(self, env, clip, prompts=None, dense_reward=.01, smoothing=1, target_object='log', **kwargs):
        super().__init__(env)
        self.clip = clip  # ClipReward
        self.wrapper_name = "ClipWrapper"

        assert prompts is not None
        self.prompt = prompts
        # self.expl_prompt = [f"Explore the widest possible area to find {target_object}"]
        self.expl_prompt = []
        self.dense_reward = dense_reward
        self.smoothing = smoothing

        self.buffer = None
        self.expl_buffer = None
        # 去掉_clip_state和_expl_clip_state改_image_state
        self._image_state = None
        self.last_score = 0
        self.expl_last_score = 0

    def reset(self, **kwargs):
        self._image_state = None

        self.buffer = None
        self.expl_buffer = None
        self.last_score = 0
        self.expl_last_score = 0

        obs = self.env.reset(**kwargs)
        obs['intrinsic'] = 0.0
        obs['score'] = 0.0

        return obs

    def step(self, action):
        obs, reward, done, info = (
            self.env.step(action)
        )
        task_num = len(self.prompt)
        expl_num = len(self.expl_prompt)

        # 一次 Vision
        if task_num > 0 or expl_num > 0:
            bundle = (self.clip.get_frame_bundle(obs))

            # 一次 Global Temporal
            video_feat, self._image_state = (
                self.clip.get_video_feat_from_global(
                    bundle["global_feat"],
                    self._image_state,
                )
            )

            # Task + Exploration 一起算
            all_prompts = (list(self.prompt) + list(self.expl_prompt))

            all_logits = (
                self.clip.get_logits_from_video(
                    video_feat,
                    all_prompts,
                )
                .detach()
                .cpu()
            )

        # Task reward
        if task_num > 0:
            task_logits = all_logits[:task_num]

            self.buffer = self._insert_buffer(
                self.buffer,
                task_logits[:1],
            )

            score = self._get_score()
            if score > self.last_score:
                obs["intrinsic"] = (self.dense_reward * score)
                self.last_score = score
            else:
                obs["intrinsic"] = 0.0

            obs["score"] = (self.dense_reward * score)

        else:
            obs["intrinsic"] = 0.0
            obs["score"] = 0.0

        # Exploration reward
        if expl_num > 0:

            expl_logits = all_logits[
                task_num:
                task_num + expl_num
            ]

            self.expl_buffer = (
                self._insert_buffer(self.expl_buffer, expl_logits[:1], )
            )

            expl_score = (self._get_expl_score())
            if expl_score > self.expl_last_score:
                info["expl_intrinsic"] = (self.dense_reward * expl_score)
                self.expl_last_score = expl_score

            else:
                info["expl_intrinsic"] = 0.0

        else:
            info["expl_intrinsic"] = 0.0

        info["clip_score"] = obs['intrinsic']
        info["clip_last_score"] = self.last_score
        info["clip_dense_reward"] = self.dense_reward

        return obs, reward, done, info

    def _get_score(self):
        score = th.mean(self.buffer)
        return (1 / (1 + th.exp(1.2 * (21.8 - score)))).item()

    def _get_expl_score(self):
        score = th.mean(self.expl_buffer)
        return (1 / (1 + th.exp(1.2 * (21.8 - score)))).item()

    def _insert_buffer(self, buffer, logits):
        if buffer is None:
            buffer = logits.unsqueeze(0)
        elif buffer.shape[0] < self.smoothing:
            buffer = th.cat([buffer, logits.unsqueeze(0)], dim=0)
        else:
            buffer = th.cat([buffer[1:], logits.unsqueeze(0)], dim=0)
        return buffer
