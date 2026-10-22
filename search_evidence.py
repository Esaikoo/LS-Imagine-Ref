"""Training-only ray evidence. No coordinates or simulator observations reach a policy.

"Discovery" means a transition to verified target evidence in the sampled rays,
not exhaustive full-image detection. "New/revisit" describe observed surface cells,
not a learned memory or a complete map.
"""
import numpy as np
import gym

LABEL_VERSION = 2
EVENTS = ("new", "discovery", "revisit")
SEARCH_LABEL_KEYS = tuple(
    key for event in EVENTS for key in ("search_" + event, "search_" + event + "_valid")
) + ("search_visible", "search_visible_valid", "search_label_version")
RAY_PITCHES = np.deg2rad(np.linspace(-25, 25, 5))
RAY_YAWS = np.deg2rad(np.linspace(-40, 40, 9))


def lidar_rays(distance=32.0):
    return [(float(p), float(y), float(distance)) for p in RAY_PITCHES for y in RAY_YAWS]


def empty_labels(version=LABEL_VERSION):
    labels = {key: np.float32(0) for key in SEARCH_LABEL_KEYS}
    labels["search_label_version"] = np.float32(version)
    return labels


def complete_episode_labels(episode):
    """Old replay has unknown labels, never fabricated negative supervision."""
    result = episode.copy()
    length = len(result["reward"])
    for key in SEARCH_LABEL_KEYS:
        if key not in result:
            result[key] = np.zeros(length, np.float32)
    return result


def _name(value):
    return str(value).lower().replace("minecraft:", "").replace(" ", "_")


BLOCK_TARGETS = {
    "log": {"log", "log2", "oak_wood", "spruce_wood", "birch_wood", "jungle_wood",
            "acacia_wood", "dark_oak_wood", "oak_log", "spruce_log", "birch_log",
            "jungle_log", "acacia_log", "dark_oak_log", "wood"},
    "sand": {"sand", "red_sand"},
    "iron_ore": {"iron_ore"},
    "water_bucket": {"water", "flowing_water", "stationary_water"},
}
ENTITY_TARGETS = {"wool": {"sheep"}}


class SearchEvidence:
    def __init__(self, target, distance=32.0, cell_size=2.0, revisit_gap=6,
                 new_fraction=0.25, revisit_fraction=0.8, expected_rays=45):
        if target not in BLOCK_TARGETS and target not in ENTITY_TARGETS:
            raise ValueError("No verified A label mapping for target: " + str(target))
        if distance <= 0 or cell_size <= 0 or revisit_gap < 1:
            raise ValueError("Invalid A evidence geometry")
        self.target, self.distance, self.cell_size = target, distance, cell_size
        self.revisit_gap = revisit_gap
        self.new_fraction, self.revisit_fraction = new_fraction, revisit_fraction
        self.expected_rays = expected_rays
        self.reset()

    def reset(self):
        self.first_seen = {}
        self.previous_visible = None
        self.step = 0

    def observe(self, obs, initial=False):
        if initial:
            self.reset()
        else:
            self.step += 1
        result = empty_labels()
        rays = obs.get("rays", {})
        fields = ("block_name", "block_distance", "entity_name", "entity_distance",
                  "traced_block_x", "traced_block_y", "traced_block_z")
        if not all(k in rays for k in fields):
            self.previous_visible = None
            return result
        arrays = {k: np.asarray(rays[k]).reshape(-1) for k in fields}
        if any(len(v) != self.expected_rays for v in arrays.values()):
            self.previous_visible = None
            return result
        numeric = [arrays[k].astype(float) for k in fields if k not in ("block_name", "entity_name")]
        if not all(np.isfinite(v).all() for v in numeric):
            self.previous_visible = None
            return result
        blocks = np.array([_name(x) for x in arrays["block_name"]])
        entities = np.array([_name(x) for x in arrays["entity_name"]])
        bd, ed = arrays["block_distance"], arrays["entity_distance"]
        # Java reports a block behind an entity as well; mask that block explicitly.
        entity_hit = (~np.isin(entities, ["null", "none", ""])) & (ed >= 0) & (ed < self.distance) & (ed <= bd)
        block_hit = (~np.isin(blocks, ["air", "null", "none", ""])) & (bd >= 0) & (bd < self.distance - 0.1) & ~entity_hit
        visible = bool(np.any(block_hit & np.isin(blocks, list(BLOCK_TARGETS.get(self.target, ()))))
                       or np.any(entity_hit & np.isin(entities, list(ENTITY_TARGETS.get(self.target, ())))))
        result["search_visible"] = np.float32(visible)
        result["search_visible_valid"] = np.float32(1)
        if not initial and (self.previous_visible is not None or not visible):
            result["search_discovery"] = np.float32(visible and not self.previous_visible)
            result["search_discovery_valid"] = np.float32(1)
        self.previous_visible = visible

        coords = np.stack([arrays["traced_block_" + axis] for axis in "xyz"], -1)[block_hit]
        cells = {tuple(cell) for cell in np.floor(coords / self.cell_size).astype(np.int64)}
        if cells:
            new = sum(cell not in self.first_seen for cell in cells) / len(cells)
            old = sum(cell in self.first_seen and self.step - self.first_seen[cell] >= self.revisit_gap
                      for cell in cells) / len(cells)
            result["search_new"] = np.float32(new >= self.new_fraction)
            result["search_revisit"] = np.float32(old >= self.revisit_fraction)
            result["search_new_valid"] = result["search_revisit_valid"] = np.float32(1)
            for cell in cells:
                self.first_seen.setdefault(cell, self.step)
        return result


class SearchEvidenceWrapper(gym.Wrapper):
    """Runs before the final image wrapper. Evaluation produces unknown labels."""
    def __init__(self, env, target="log", enabled=False, **kwargs):
        super().__init__(env)
        self.evidence = SearchEvidence(target, **kwargs) if enabled else None

    def _annotate(self, obs, initial=False):
        obs = obs.copy()
        obs.update(self.evidence.observe(obs, initial) if self.evidence else empty_labels())
        return obs

    def reset(self, **kwargs):
        return self._annotate(self.env.reset(**kwargs), initial=True)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._annotate(obs), reward, done, info
