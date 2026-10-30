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
) + ("search_visible", "search_visible_valid", "search_label_version",
     "search_place_id", "search_place_valid",
     "search_first_discovery", "search_first_discovery_valid",
     "search_surface_coverage", "search_surface_coverage_valid")
VNEXT_LABEL_VERSION = 3
VNEXT_EVENTS = ("center_target", "reachable", "center_interactable", "return", "dwell", "progress", "blocked")
VNEXT_LABEL_KEYS = SEARCH_LABEL_KEYS + tuple(
    key for event in VNEXT_EVENTS for key in ("search_" + event, "search_" + event + "_valid")
)
RAY_PITCHES = np.deg2rad(np.linspace(-25, 25, 5))
RAY_YAWS = np.deg2rad(np.linspace(-40, 40, 9))


def lidar_rays(distance=32.0):
    return [(float(p), float(y), float(distance)) for p in RAY_PITCHES for y in RAY_YAWS]


def empty_labels(version=LABEL_VERSION):
    labels = {key: np.float32(0) for key in (VNEXT_LABEL_KEYS if version >= 3 else SEARCH_LABEL_KEYS)}
    labels["search_label_version"] = np.float32(version)
    return labels


def complete_episode_labels(episode, version=LABEL_VERSION):
    """Old replay has unknown labels, never fabricated negative supervision."""
    result = episode.copy()
    length = len(result["reward"])
    for key in (VNEXT_LABEL_KEYS if version >= 3 else SEARCH_LABEL_KEYS):
        if key not in result:
            result[key] = np.zeros(length, np.float32)
    from search_c import complete_c_fields
    return complete_c_fields(result)


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
                 new_fraction=0.25, revisit_fraction=0.8, expected_rays=45, version=2,
                 reach_distance=3.0):
        if target not in BLOCK_TARGETS and target not in ENTITY_TARGETS:
            raise ValueError("No verified A label mapping for target: " + str(target))
        if distance <= 0 or cell_size <= 0 or revisit_gap < 1:
            raise ValueError("Invalid A evidence geometry")
        if version not in (2, 3) or reach_distance <= 0 or reach_distance > distance:
            raise ValueError("Invalid evidence version or interaction distance")
        self.version, self.reach_distance = int(version), float(reach_distance)
        self.target, self.distance, self.cell_size = target, distance, cell_size
        self.revisit_gap = revisit_gap
        self.new_fraction, self.revisit_fraction = new_fraction, revisit_fraction
        self.expected_rays = expected_rays
        self.reset()

    def reset(self):
        self.first_seen = {}
        self.last_seen = {}
        self.last_epoch = {}
        self.epoch = 0
        self.previous_position = None
        self.place_ids = {}
        self.previous_visible = None
        self.ever_visible = False
        self.history_known = True
        self.step = 0

    def observe(self, obs, initial=False):
        if initial:
            self.reset()
        else:
            self.step += 1
        result = empty_labels(self.version)
        # Opaque within-episode place/view IDs are targets, never policy inputs.
        location = obs.get("location_stats", {})
        position = location.get("pos")
        if position is not None and "yaw" in location and "pitch" in location:
            pose = np.concatenate((np.asarray(position).reshape(-1),
                                   np.asarray([location["yaw"], location["pitch"]]).reshape(-1)))
            if len(pose) == 5 and np.isfinite(pose).all():
                cell = tuple(np.floor(pose[:3] / self.cell_size).astype(np.int64))
                view = (int(np.floor((pose[3] % 360) / 45)), int(np.floor(pose[4] / 45)))
                place = self.place_ids.setdefault(cell + view, len(self.place_ids) + 1)
                result["search_place_id"] = np.float32(place)
                result["search_place_valid"] = np.float32(1)
        rays = obs.get("rays", {})
        fields = ("block_name", "block_distance", "entity_name", "entity_distance",
                  "traced_block_x", "traced_block_y", "traced_block_z")
        if not all(k in rays for k in fields):
            self._unknown_geometry()
            return result
        arrays = {k: np.asarray(rays[k]).reshape(-1) for k in fields}
        if any(len(v) != self.expected_rays for v in arrays.values()):
            self._unknown_geometry()
            return result
        numeric = [arrays[k].astype(float) for k in fields if k not in ("block_name", "entity_name")]
        if not all(np.isfinite(v).all() for v in numeric):
            self._unknown_geometry()
            return result
        blocks = np.array([_name(x) for x in arrays["block_name"]])
        entities = np.array([_name(x) for x in arrays["entity_name"]])
        bd, ed = arrays["block_distance"], arrays["entity_distance"]
        # Java reports a block behind an entity as well; mask that block explicitly.
        unoccluded = ed <= bd
        if self.version >= 3:
            valid_block = (~np.isin(blocks, ["air", "null", "none", ""])) & (bd >= 0)
            unoccluded = ~valid_block | (ed <= bd)
        entity_hit = (~np.isin(entities, ["null", "none", ""])) & (ed >= 0) & (ed < self.distance) & unoccluded
        block_hit = (~np.isin(blocks, ["air", "null", "none", ""])) & (bd >= 0) & (bd < self.distance - 0.1) & ~entity_hit
        visible = bool(np.any(block_hit & np.isin(blocks, list(BLOCK_TARGETS.get(self.target, ()))))
                       or np.any(entity_hit & np.isin(entities, list(ENTITY_TARGETS.get(self.target, ())))))
        result["search_visible"] = np.float32(visible)
        result["search_visible_valid"] = np.float32(1)
        # Distinct from legacy discovery (reacquisition). Never pay repeatedly
        # for looking away and back, nor infer first discovery across unknowns.
        result["search_first_discovery_valid"] = np.float32(self.history_known or self.ever_visible)
        result["search_first_discovery"] = np.float32(visible and not self.ever_visible and self.history_known)
        self.ever_visible = self.ever_visible or visible
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
            result["search_surface_coverage"] = np.float32(len(self.first_seen))
            result["search_surface_coverage_valid"] = np.float32(1)
        if self.version >= 3:
            self._vnext_geometry(obs, result, arrays, block_hit, entity_hit, cells, initial)
        return result

    def _unknown_geometry(self):
        self.previous_visible = None
        self.history_known = False
        self.previous_position = None
        self.epoch += 1

    def _tool_legality(self, obs):
        # These are training labels only. Inventory is never exported to policy.
        if self.target in ("sand", "log"):
            return True, True
        # MineDojo EquipmentObservation lists mainhand first. Merely owning a
        # pickaxe/bucket in another inventory slot does not make this action legal.
        equipment = obs.get("equipment")
        if not isinstance(equipment, dict) or "name" not in equipment:
            return False, False
        names = np.asarray(equipment["name"]).reshape(-1)
        quantity = np.asarray(equipment.get("quantity", np.ones(len(names)))).reshape(-1)
        if not len(names) or len(quantity) != len(names) or not np.isfinite(quantity.astype(float)).all():
            return False, False
        available = {_name(names[0])} if float(quantity[0]) > 0 else set()
        if self.target == "iron_ore":
            legal = any(any(tool in n for tool in ("stone_pickaxe", "iron_pickaxe", "diamond_pickaxe"))
                        for n in available)
        elif self.target == "water_bucket":
            legal = "bucket" in available
        elif self.target == "wool":
            legal = "shears" in available
        else:
            return False, False
        return bool(legal), True

    def _vnext_geometry(self, obs, result, arrays, block_hit, entity_hit, cells, initial):
        if self.expected_rays == len(RAY_PITCHES) * len(RAY_YAWS):
            center = len(RAY_YAWS) * (len(RAY_PITCHES)//2) + len(RAY_YAWS)//2
            target_block = block_hit[center] and _name(arrays["block_name"][center]) in BLOCK_TARGETS.get(self.target, ())
            target_entity = entity_hit[center] and _name(arrays["entity_name"][center]) in ENTITY_TARGETS.get(self.target, ())
            target = bool(target_block or target_entity)
            distance = float(arrays["entity_distance" if target_entity else "block_distance"][center])
            reach = bool(target and distance <= self.reach_distance)
            result["search_center_target"] = np.float32(target)
            result["search_center_target_valid"] = np.float32(1)
            result["search_reachable"] = np.float32(reach)
            result["search_reachable_valid"] = np.float32(1)
            tool, tool_known = self._tool_legality(obs)
            # A proven geometric negative needs no tool assumption. A positive
            # requires known legality for this same center hit, not a nearby ray.
            result["search_center_interactable"] = np.float32(reach and tool)
            result["search_center_interactable_valid"] = np.float32(not reach or tool_known)
        if cells:
            known = all(cell not in self.last_seen or self.last_epoch[cell] == self.epoch for cell in cells)
            returning = sum(cell in self.last_seen and self.last_epoch[cell] == self.epoch
                            and self.step-self.last_seen[cell] >= self.revisit_gap for cell in cells) / len(cells)
            dwell = sum(cell in self.last_seen and self.last_epoch[cell] == self.epoch
                        and self.last_seen[cell] == self.step-1
                        and self.step-self.first_seen[cell] >= self.revisit_gap for cell in cells) / len(cells)
            result["search_return"] = result["search_revisit"] = np.float32(returning >= self.revisit_fraction)
            result["search_return_valid"] = result["search_revisit_valid"] = np.float32(known and not initial)
            result["search_dwell"] = np.float32(dwell >= self.revisit_fraction)
            result["search_dwell_valid"] = np.float32(known and not initial)
            # A missing ray interval cannot prove a new physical observation.
            result["search_new_valid"] = np.float32(self.history_known)
            for cell in cells:
                self.last_seen[cell] = self.step
                self.last_epoch[cell] = self.epoch


class SearchEvidenceWrapper(gym.Wrapper):
    """Runs before the final image wrapper. Evaluation produces unknown labels."""
    def __init__(self, env, target="log", enabled=False, **kwargs):
        super().__init__(env)
        self.version = int(kwargs.get("version", LABEL_VERSION))
        self.evidence = SearchEvidence(target, **kwargs) if enabled else None
        self._last_position = None

    def _annotate(self, obs, initial=False):
        obs = obs.copy()
        obs.update(self.evidence.observe(obs, initial) if self.evidence else empty_labels(self.version))
        return obs

    @staticmethod
    def _position(obs):
        value = obs.get("location_stats", {}).get("pos")
        if value is None:
            return None
        value = np.asarray(value, float).reshape(-1)
        return value.copy() if value.shape == (3,) and np.isfinite(value).all() else None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_position = self._position(obs)
        return self._annotate(obs, initial=True)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.version >= 3 and info.get("error"):
            labels = obs.copy()
            labels.update(empty_labels(self.version))
            if self.evidence is not None:
                self.evidence._unknown_geometry()
            self._last_position = None
            return labels, reward, done, info
        position = self._position(obs)
        labels = self._annotate(obs)
        if self.version >= 3 and self.evidence is not None:
            finite_reward = bool(np.isfinite(reward))
            labels["search_progress"] = np.float32(finite_reward and (float(reward) > 0 or bool(info.get("success", False))))
            labels["search_progress_valid"] = np.float32(finite_reward)
            moving = isinstance(action, dict) and any(bool(np.asarray(action.get(k, 0)).any())
                for k in ("forward", "back", "left", "right"))
            known = moving and position is not None and self._last_position is not None
            labels["search_blocked"] = np.float32(known and np.linalg.norm(position-self._last_position) < .03)
            labels["search_blocked_valid"] = np.float32(known)
        self._last_position = position
        return labels, reward, done, info
