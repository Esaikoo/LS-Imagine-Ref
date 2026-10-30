from gym import Wrapper
import gym.spaces as spaces
import numpy as np
from abc import ABC
import cv2
from collections import OrderedDict

import copy
import uuid
from search_evidence import SEARCH_LABEL_KEYS, VNEXT_LABEL_KEYS, empty_labels
from search_actuator import CONTROL_KEYS, CONTROL_DIM, STATE_DIM, apply_command, world_group_id

BASIC_ACTIONS = {
    "noop": dict(),
    "attack": dict(attack=np.array(1)),
    "turn_up": dict(camera=np.array([-10.0, 0.])),
    "turn_down": dict(camera=np.array([10.0, 0.])),
    "turn_left": dict(camera=np.array([0., -10.0])),
    "turn_right": dict(camera=np.array([0., 10.0])),
    "forward": dict(forward=np.array(1)),
    "back": dict(back=np.array(1)),
    "left": dict(left=np.array(1)),
    "right": dict(right=np.array(1)),
    "jump": dict(jump=np.array(1), forward=np.array(1)),
    "use": dict(use=np.array(1)),
}

NOOP_ACTION = {
    'camera': np.array([0., 0.]), 
    'smelt': 'none', 
    'craft': 'none', 
    'craft_with_table': 'none', 
    'forward': np.array(0), 
    'back': np.array(0), 
    'left': np.array(0), 
    'right': np.array(0), 
    'jump': np.array(0), 
    'sneak': np.array(0), 
    'sprint': np.array(0), 
    'use': np.array(0), 
    'attack': np.array(0), 
    'drop': 0, 
    'swap_slot': OrderedDict([('source_slot', 0), ('target_slot', 0)]), 
    'pickItem': 0, 
    'hotbar.1': 0, 
    'hotbar.2': 0, 
    'hotbar.3': 0, 
    'hotbar.4': 0, 
    'hotbar.5': 0, 
    'hotbar.6': 0, 
    'hotbar.7': 0, 
    'hotbar.8': 0, 
    'hotbar.9': 0,
}


class LSImagineWrapper(Wrapper, ABC):
    def __init__(self, env, repeat=1, sticky_attack=0, sticky_jump=10, pitch_limit=(-70, 70), vnext=False,
                 vnext_world_seed=None):
        super().__init__(env)
        self.wrapper_name = "LSImagineWrapper"
        self._vnext = bool(vnext)
        self._fixed_world_seed = None if vnext_world_seed is None else str(vnext_world_seed)
        if self._fixed_world_seed is not None:
            try:
                self._fixed_world_seed = str(int(self._fixed_world_seed))
            except ValueError:
                pass  # Minecraft also accepts named seeds; retain exact name.
        self._world_session = uuid.uuid4().hex
        self._world_generation = 0
        self._world_group = 0
        if self._vnext and repeat != 1:
            raise ValueError("vNext actuator protocol currently requires repeat=1")
        self._actuator_log = {
            "log_actuator_before": np.zeros(STATE_DIM, np.float32),
            "log_actuator_after": np.zeros(STATE_DIM, np.float32),
            "log_applied_control": np.zeros(CONTROL_DIM, np.float32),
            "log_requested_action": np.int32(0),
            "log_env_error": np.float32(0),
        }

        self._noop_action = NOOP_ACTION
        actions = self._insert_defaults(BASIC_ACTIONS)
        self._action_names = tuple(actions.keys())
        self._action_values = tuple(actions.values())

        self.observation_space = spaces.Dict(
            {
                'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                'heatmap': spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
                'jump': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                'is_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                'is_calculated': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                'is_first': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                'is_last': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                'is_terminal': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                'reward_on_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'intrinsic': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'intrinsic_on_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'score': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'score_on_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'jumping_steps': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                'accumulated_reward': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            }
        )

        for key in (VNEXT_LABEL_KEYS if self._vnext else SEARCH_LABEL_KEYS):
            high = (np.inf if key in ("search_place_id", "search_surface_coverage") else
                    3 if self._vnext and key == "search_label_version" else 2)
            self.observation_space.spaces[key] = spaces.Box(0, high, shape=(), dtype=np.float32)

        if self._vnext:
            self.observation_space.spaces.update({
                "log_actuator_before": spaces.Box(-np.inf, np.inf, (STATE_DIM,), dtype=np.float32),
                "log_actuator_after": spaces.Box(-np.inf, np.inf, (STATE_DIM,), dtype=np.float32),
                "log_applied_control": spaces.Box(-np.inf, np.inf, (CONTROL_DIM,), dtype=np.float32),
                "log_requested_action": spaces.Box(0, 11, (), dtype=np.int32),
                "log_env_error": spaces.Box(0, 1, (), dtype=np.float32),
                "log_vnext_world_group": spaces.Box(0, (1 << 24)-1, (), dtype=np.int64),
            })
        self.action_space = spaces.discrete.Discrete(len(BASIC_ACTIONS))
        self.action_space.discrete = True
        self._repeat = repeat
        self._sticky_attack_length = sticky_attack
        self._sticky_attack_counter = 0
        self._sticky_jump_length = sticky_jump
        self._sticky_jump_counter = 0
        self._pitch_limit = pitch_limit
        self._pitch = 0

    def reset(self):
        obs = self.env.reset()
        if self._vnext:
            current, seen, group = self.env, set(), None
            while current is not None and id(current) not in seen:
                seen.add(id(current))
                attrs = vars(current)
                if "_vnext_world_group" in attrs:
                    group = attrs["_vnext_world_group"]
                    break
                current = attrs.get("env")
            if group is None:
                # Without semifast reset each reset starts a full mission.
                self._world_generation += 1
                group = world_group_id(self._world_session, self._world_generation)
            if self._fixed_world_seed is not None:
                # Identical fixed seeds can regenerate identical terrain across
                # clients/resume. They must never land in different data folds.
                group = world_group_id("fixed-world-seed:" + self._fixed_world_seed, 0)
            self._world_group = int(group)
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pitch = 0
        for key, value in self._actuator_log.items():
            self._actuator_log[key] = np.zeros_like(value)
        return self._obs(obs)

    def step(self, action):
        requested = int(action)
        action = copy.deepcopy(self._action_values[requested])
        if self._vnext:
            before = np.asarray((self._sticky_attack_counter, self._sticky_jump_counter, self._pitch), np.float32)
            applied, after = apply_command(requested, before, self._sticky_attack_length,
                                           self._sticky_jump_length, self._pitch_limit)
            for index, name in enumerate(CONTROL_KEYS[:7]):
                action[name] = np.array(int(applied[index]))
            action["camera"] = applied[7:9].copy()
            self._sticky_attack_counter, self._sticky_jump_counter, self._pitch = after.tolist()
            self._actuator_log = {
                "log_actuator_before": before,
                "log_actuator_after": after,
                "log_applied_control": applied,
                "log_requested_action": np.int32(requested),
                "log_env_error": np.float32(0),
            }
        else:
            action = self._action(action)
        following = self._noop_action.copy()
        for key in ("attack", "forward", "back", "left", "right"):
            following[key] = action[key]
        for act in [action] + ([following] * (self._repeat - 1)):
            obs, reward, done, info = self.env.step(act)
            if "error" in info:
                done = True
                break
        if self._vnext:
            self._actuator_log["log_env_error"] = np.float32(bool(info.get("error")))
        obs["is_first"] = False
        obs["is_last"] = bool(done)
        obs["is_terminal"] = bool(info.get("is_terminal", info.get("real_done", False)))
        obs = self._obs(obs)

        assert "pov" not in obs, list(obs.keys())

        return obs, reward, done, info

    def _obs(self, obs):
        evidence = {key: np.float32(obs.get(key, value)) for key, value in empty_labels(3 if self._vnext else 2).items()}
        image = obs['rgb'] # 3 * H * W
        image = image.transpose(1, 2, 0).astype(np.uint8) # H * W * 3
        image = cv2.resize(image, (64, 64)) # 64 * 64 * 3

        if 'zoomed_image' in obs:
            zoomed_image = obs['zoomed_image'] # H * W * 3
            zoomed_image = zoomed_image.astype(np.uint8) # H * W * 3
            zoomed_image = cv2.resize(zoomed_image, (64, 64)) # 64 * 64 * 3
        else:
            zoomed_image = np.zeros_like(image)

        heatmap = cv2.resize(obs['heatmap'] if 'heatmap' in obs else np.zeros((64, 64, 1)), (64, 64))
        heatmap_on_zoomed = cv2.resize(obs['heatmap_on_zoomed'] if 'heatmap_on_zoomed' in obs else np.zeros((64, 64, 1)), (64, 64))
        heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
        heatmap_on_zoomed = np.clip(heatmap_on_zoomed * 255, 0, 255).astype(np.uint8)

        obs = {
            'image': image,
            'heatmap': heatmap,
            'jump': obs['jump'] if 'jump' in obs else False,
            'is_zoomed': obs['is_zoomed'] if 'is_zoomed' in obs else False,
            'is_calculated': obs['is_calculated'] if 'is_calculated' in obs else False,
            'is_first': obs['is_first'],
            'is_last': obs['is_last'],
            'is_terminal': obs['is_terminal'],
            'reward_on_zoomed': obs['reward_on_zoomed'] if 'reward_on_zoomed' in obs else 0.0,
            'intrinsic': obs['intrinsic'] if 'intrinsic' in obs else 0.0,
            'intrinsic_on_zoomed': obs['intrinsic_on_zoomed'] if 'intrinsic_on_zoomed' in obs else 0.0,
            'score': obs['score'] if 'score' in obs else 0.0,
            'score_on_zoomed': obs['score_on_zoomed'] if 'score_on_zoomed' in obs else 0.0,
            'jumping_steps': obs['jumping_steps'] if 'jumping_steps' in obs else 1000.0,
            'accumulated_reward': obs['accumulated_reward'] if 'accumulated_reward' in obs else 1000.0,
        }

        obs.update(evidence)
        if self._vnext:
            obs.update({key: np.asarray(value).copy() for key, value in self._actuator_log.items()})
            obs["log_vnext_world_group"] = np.int64(self._world_group)
        if obs["is_zoomed"]:
            obs["zoomed_image"] = zoomed_image
            obs["heatmap_on_zoomed"] = heatmap_on_zoomed
        else:
            obs["zoomed_image"] = None
            obs["heatmap_on_zoomed"] = None

        if self._vnext:
            # The previous tuple assertion was always true. Validate only the
            # new protocol metadata; legacy base-observation scalar quirks stay.
            keys = VNEXT_LABEL_KEYS + tuple(self._actuator_log) + ("log_vnext_world_group",)
            for key in keys:
                if not self.observation_space[key].contains(np.asarray(obs[key])):
                    raise ValueError("Invalid vNext observation metadata: " + key)
        return obs

    def _action(self, action):
        if self._sticky_attack_length:
            if action["attack"]:
                self._sticky_attack_counter = self._sticky_attack_length
            if self._sticky_attack_counter > 0:
                action["attack"] = np.array(1)
                action["jump"] = np.array(0)
                self._sticky_attack_counter -= 1
        if self._sticky_jump_length:
            if action["jump"]:
                self._sticky_jump_counter = self._sticky_jump_length
            if self._sticky_jump_counter > 0:
                action["jump"] = np.array(1)
                action["forward"] = np.array(1)
                self._sticky_jump_counter -= 1
        if self._pitch_limit and action["camera"][0]:
            lo, hi = self._pitch_limit
            if not (lo <= self._pitch + action["camera"][0] <= hi):
                action["camera"] = (0, action["camera"][1])
            self._pitch += action["camera"][0]
        return action


    def _insert_defaults(self, actions):
        actions = {name: action.copy() for name, action in actions.items()}
        for key, default in self._noop_action.items():
            for action in actions.values():
                if key not in action:
                    action[key] = default
        return actions

        