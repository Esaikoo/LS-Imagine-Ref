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
GOAL_LABEL_VERSION = 1
GOAL_GRID = 16
GOAL_IMAGE_SIZE = 96
GOAL_LABEL_SHAPES = {
    "goal_region_gt": (GOAL_GRID, GOAL_GRID, 1),
    "goal_region_valid": (GOAL_GRID, GOAL_GRID, 1),
    "goal_instance_group": (GOAL_GRID, GOAL_GRID, 1),
    "goal_visible_gt": (), "goal_visible_valid": (),
    "goal_interaction_outcome": (), "goal_interaction_valid": (),
    "goal_label_version": (), "goal_holdout": (),
}
GOAL_LABEL_DTYPES = dict.fromkeys(GOAL_LABEL_SHAPES, np.uint8)
GOAL_LABEL_DTYPES["goal_instance_group"] = np.int32


def empty_goal_labels():
    return {key: np.zeros(shape, GOAL_LABEL_DTYPES[key])
            for key, shape in GOAL_LABEL_SHAPES.items()}


SPATIAL_LABEL_SHAPES = {"search_pose_delta": (3,), "search_pose_valid": ()}
for _event in ("new_area", "area_revisit", "new_view", "view_revisit"):
    SPATIAL_LABEL_SHAPES["search_" + _event] = ()
    SPATIAL_LABEL_SHAPES["search_" + _event + "_valid"] = ()
SPATIAL_LABEL_SHAPES.update(search_area_coverage=(), log_spatial_gt_pose=(3,),
                            log_spatial_gt_valid=())


def empty_spatial_labels():
    return {key: np.zeros(shape, np.float32) for key, shape in SPATIAL_LABEL_SHAPES.items()}


class SpatialEvidenceWrapper(gym.Wrapper):
    """Coordinate-derived loss targets and separate offline audit metadata.

    No public position is ever supplied to spatial memory. Evaluation's loss
    targets remain unknown; optional audit coordinates are log_* and are removed
    before the policy call, not used to initialize, correct or update its map.
    """
    def __init__(self, env, supervision=False, audit=False, cell_size=1.5, sectors=12):
        super().__init__(env)
        self.supervision, self.audit = bool(supervision), bool(audit)
        self.cell_size, self.sectors = float(cell_size), int(sectors)
        if self.cell_size <= 0 or self.sectors < 4:
            raise ValueError("Invalid spatial supervision grid")
        self.previous = None
        self.areas, self.views = set(), set()
        self.origin_pose = None

    @staticmethod
    def pose(obs):
        location = obs.get("location_stats", {})
        try:
            pos = np.asarray(location["pos"], float).reshape(3)
            yaw = float(np.asarray(location["yaw"]).reshape(-1)[0])
        except (KeyError, TypeError, ValueError, IndexError):
            return None
        result = np.array([pos[0], pos[2], np.deg2rad(yaw)], np.float64)
        return result if np.isfinite(result).all() else None

    def annotate(self, obs, initial=False, error=False):
        result = obs.copy()
        labels = empty_spatial_labels()
        current = self.pose(obs) if not error and (self.supervision or self.audit) else None
        if current is not None and self.audit:
            labels["log_spatial_gt_pose"] = current.astype(np.float32)
            labels["log_spatial_gt_valid"] = np.float32(1)
        if current is not None and self.supervision:
            if self.origin_pose is None:
                self.origin_pose = current.copy()
            def map_position(pose):
                offset = pose[:2]-self.origin_pose[:2]
                angle = self.origin_pose[2]
                return np.array([np.cos(angle)*offset[0]+np.sin(angle)*offset[1],
                                 -np.sin(angle)*offset[0]+np.cos(angle)*offset[1]])
            area = tuple(np.floor(map_position(current) / self.cell_size+.5).astype(np.int64))
            direction = int(np.floor(((current[2]-self.origin_pose[2]) % (2*np.pi)) / (2*np.pi) * self.sectors+.5)) % self.sectors
            view = (*area, direction)
            known = not initial and self.previous is not None
            if known:
                delta = current[:2] - self.previous[:2]
                angle = self.previous[2]
                local = np.array([np.cos(angle)*delta[0]+np.sin(angle)*delta[1],
                                  -np.sin(angle)*delta[0]+np.cos(angle)*delta[1]])
                turn = (current[2]-angle+np.pi) % (2*np.pi)-np.pi
                # Exclude teleports/reset glitches rather than teaching walking
                # to explain discontinuous world-coordinate jumps.
                valid = np.linalg.norm(local) < 3.0 and abs(turn) < np.pi/2
                labels["search_pose_delta"] = np.r_[local, turn].astype(np.float32)
                labels["search_pose_valid"] = np.float32(valid)
                known = valid
            previous_area = (None if self.previous is None else
                             tuple(np.floor(map_position(self.previous)/self.cell_size+.5).astype(np.int64)))
            previous_direction = (None if self.previous is None else int(np.floor(
                ((self.previous[2]-self.origin_pose[2]) % (2*np.pi))/(2*np.pi)*self.sectors+.5)) % self.sectors)
            previous_view = None if previous_area is None else (*previous_area,previous_direction)
            values = {"new_area": area not in self.areas,
                      "area_revisit": area in self.areas and area != previous_area,
                      "new_view": view not in self.views,
                      "view_revisit": view in self.views and view != previous_view}
            for name, value in values.items():
                labels["search_"+name] = np.float32(value and known)
                labels["search_"+name+"_valid"] = np.float32(known)
            self.areas.add(area); self.views.add(view)
            labels["search_area_coverage"] = np.float32(len(self.areas))
        self.previous = current
        result.update(labels)
        return result

    def reset(self, **kwargs):
        self.previous = None
        self.origin_pose = None
        self.areas.clear(); self.views.clear()
        return self.annotate(self.env.reset(**kwargs), initial=True)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.annotate(obs, error=bool(info.get("error"))), reward, done, info
RAY_PITCHES = np.deg2rad(np.linspace(-25, 25, 5))
RAY_YAWS = np.deg2rad(np.linspace(-40, 40, 9))


# Additional training-only sightlines land in distinct cells over the
# client's FOV range. Keep the original 45 rays first for the A labels.
GOAL_RAY_OFFSETS = tuple(
    (float(np.deg2rad(p)), float(np.deg2rad(y)))
    for pitches, yaws in (((-7, -3, 3, 7), (-12, -2, 2, 12)),
                          ((-9, -5, 5, 9), (-27, -17, 17, 27)))
    for p in pitches for y in yaws
)


def lidar_rays(distance=32.0, *, goal=False):
    rays = [(float(p), float(y), float(distance)) for p in RAY_PITCHES for y in RAY_YAWS]
    if goal:
        rays.extend((p, y, float(distance)) for p, y in GOAL_RAY_OFFSETS)
    return rays


def goal_ray_cells(obs, count):
    return _project_goal_rays(obs, count)[0]


def _project_goal_rays(obs, count):
    """Conservative projection from MineDojo's first-person rich lidar.

    The ray implementation adds elevation, clips at +/-89 degrees, then rotates
    about world up. Its positive yaw offset points to camera-left. Use the
    native framebuffer aspect before the policy's square resize. Keep only rays
    that occupy one grid cell across the default client's FOV variation. A ray
    supervises its sampled point in that cell, never all pixels in a patch.
    """
    unknown = np.full((count, 2), -1, np.int16)
    outside = np.zeros(count, bool)
    rays = obs.get("rays", {})
    try:
        image = np.asarray(obs["rgb"])
        pitch = np.asarray(rays["ray_pitch"], float).reshape(count)
        yaw = np.asarray(rays["ray_yaw"], float).reshape(count)
        elevation = -np.deg2rad(float(np.asarray(obs["location_stats"]["pitch"]).item()))
    except (KeyError, TypeError, ValueError):
        return unknown, outside
    if image.ndim != 3 or image.shape[0] != 3 or min(image.shape[1:]) <= 0:
        return unknown, outside
    if not np.isfinite(elevation) or not np.isfinite(pitch).all() or not np.isfinite(yaw).all():
        return unknown, outside
    ray_elevation = np.clip(elevation + pitch, -np.deg2rad(89), np.deg2rad(89))
    horizontal = np.cos(ray_elevation)
    forward = np.sin(elevation)*np.sin(ray_elevation) + np.cos(elevation)*horizontal*np.cos(yaw)
    right = -horizontal*np.sin(yaw)
    up = np.cos(elevation)*np.sin(ray_elevation) - np.sin(elevation)*horizontal*np.cos(yaw)
    aspect = image.shape[2]/image.shape[1]
    projected = []
    for fov in (58., 80.):
        scale = 2*np.maximum(forward, 1e-6)*np.tan(np.deg2rad(fov/2))
        row = .5 - up/scale
        col = .5 + right/(scale*aspect)
        projected.append(np.floor(GOAL_GRID*np.stack((row, col), axis=-1)).astype(np.int32))
    a, b = projected
    in_view = ((forward > 0) & (a >= 0).all(-1) & (a < GOAL_GRID).all(-1)
               & (b >= 0).all(-1) & (b < GOAL_GRID).all(-1))
    stable = in_view & (a == b).all(-1)
    unknown[stable] = a[stable]
    center = (np.abs(pitch) < 1e-7) & (np.abs(yaw) < 1e-7) & (abs(elevation) <= np.deg2rad(89))
    unknown[center] = GOAL_GRID//2
    return unknown, in_view | center


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
    from apm.legacy.control import complete_c_fields
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
# Rich lidar returns the Java class name on the standard client.
ENTITY_TARGETS = {"wool": {"sheep", "entitysheep"}}


class SearchEvidence:
    def __init__(self, target, distance=32.0, cell_size=2.0, revisit_gap=6,
                 new_fraction=0.25, revisit_fraction=0.8, expected_rays=45, version=2,
                 reach_distance=3.0, goal_supervision=False):
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
        self.goal_supervision = bool(goal_supervision)
        if self.goal_supervision and self.version < 3:
            raise ValueError("D supervision requires vNext first-hit evidence")
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
        self.goal_instances = {}

    def observe(self, obs, initial=False):
        if initial:
            self.reset()
        else:
            self.step += 1
        result = empty_labels(self.version)
        if self.goal_supervision:
            result.update(empty_goal_labels())
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
        count = len(arrays["block_name"])
        legacy = slice(None, self.expected_rays)
        allowed = (self.expected_rays, self.expected_rays + len(GOAL_RAY_OFFSETS)
                   if self.goal_supervision else self.expected_rays)
        if count not in allowed or any(len(v) != count for v in arrays.values()):
            self._unknown_geometry()
            return result
        numeric = [arrays[k].astype(float) for k in fields if k not in ("block_name", "entity_name")]
        if not all(np.isfinite(v[legacy]).all() for v in numeric):
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
        goal_blocks = block_hit & np.isin(blocks, list(BLOCK_TARGETS.get(self.target, ())))
        goal_entities = entity_hit & np.isin(entities, list(ENTITY_TARGETS.get(self.target, ())))
        if self.goal_supervision and all(np.isfinite(v).all() for v in numeric):
            self._goal_regions(obs, result, arrays, blocks, block_hit, goal_blocks, goal_entities)
        # Additional D rays do not alter the scope of existing A, map,
        # center-interaction or discovery supervision.
        arrays = {k: v[legacy] for k, v in arrays.items()}
        block_hit, entity_hit = block_hit[legacy], entity_hit[legacy]
        visible = bool(goal_blocks[legacy].any() or goal_entities[legacy].any())
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

    def _goal_regions(self, obs, result, arrays, blocks, block_hit, goal_blocks, goal_entities):
        # A D cell represents one or more sampled rays, not a filled mask.
        # Rich lidar can miss distant rendered objects. Neither air nor an
        # empty ray scan supplies a negative full-frame label.
        cells, in_view = _project_goal_rays(obs, len(block_hit))
        # MineDojo fills miss coordinates at the end of a ray. Its synthetic
        # miss distance is unreliable, so independently require real hit
        # coordinates to be interior to the ray and consistent with distance.
        location = obs.get("location_stats", {})
        try:
            eye = np.asarray(location["pos"], float).reshape(3) + [0, 1.62, 0]
        except (KeyError, TypeError, ValueError):
            eye = np.full(3, np.nan)
        points = np.stack([arrays["traced_block_" + axis] for axis in "xyz"], axis=-1).astype(float)+.5
        distance = np.linalg.norm(points-eye, axis=-1)
        confirmed_block = (block_hit & (distance < self.distance-1.5)
                           & (np.abs(distance-arrays["block_distance"]) < 1.3))
        rays = obs.get("rays", {})
        solid = np.asarray(rays.get("is_solid", np.zeros(len(block_hit)))).reshape(-1)
        if solid.shape != block_hit.shape:
            solid = np.zeros_like(block_hit)
        transparent = np.array([any(name in block for name in
                            ("glass", "ice", "leaf", "leaves", "fence", "pane", "door", "slime"))
                            for block in blocks])
        targets = (confirmed_block & goal_blocks) | goal_entities
        first_hits = targets | (confirmed_block & ~goal_blocks & (solid == 1) & ~transparent)
        labels = result["goal_region_gt"][..., 0]
        valid = result["goal_region_valid"][..., 0]
        instances = result["goal_instance_group"][..., 0]
        conflicts = set()
        for index in np.flatnonzero(first_hits & (cells[:, 0] >= 0)):
            cell = tuple(cells[index])
            if cell in conflicts:
                continue
            target = bool(targets[index])
            instance = 0
            if confirmed_block[index] and goal_blocks[index]:
                coord = tuple(int(arrays["traced_block_" + axis][index]) for axis in "xyz")
                # Episode-local IDs are opaque. No geometry reaches the model.
                instance = self.goal_instances.setdefault(coord, len(self.goal_instances)+1)
            if valid[cell] and bool(labels[cell]) != target:
                valid[cell] = labels[cell] = instances[cell] = 0
                conflicts.add(cell)
            else:
                instances[cell] = 0 if valid[cell] and instances[cell] != instance else instance
                labels[cell] = target
                valid[cell] = 1
        result["goal_label_version"] = np.uint8(GOAL_LABEL_VERSION)
        # Visibility can use hits that stay in-frame even when their exact
        # 16x16 cell moves with FOV. A sparse miss still cannot prove absence.
        result["goal_visible_gt"] = result["goal_visible_valid"] = np.uint8((targets & in_view).any())

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
        self.goal_supervision = self.evidence is not None and self.evidence.goal_supervision
        self._target = target
        self._last_position = None
        self._last_goal_quantity = None
        self._goal_succeeded = False

    def _annotate(self, obs, initial=False):
        obs = obs.copy()
        for key in GOAL_LABEL_SHAPES:
            obs.pop(key, None)
        obs.update(self.evidence.observe(obs, initial) if self.evidence else empty_labels(self.version))
        return obs

    @staticmethod
    def _position(obs):
        value = obs.get("location_stats", {}).get("pos")
        if value is None:
            return None
        value = np.asarray(value, float).reshape(-1)
        return value.copy() if value.shape == (3,) and np.isfinite(value).all() else None

    def _goal_quantity(self, obs):
        inventory = obs.get("inventory", {})
        try:
            names = np.asarray(inventory["name"]).reshape(-1)
            quantities = np.asarray(inventory["quantity"], float).reshape(names.shape)
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(quantities).all() or (quantities < 0).any():
            return None
        acceptable = BLOCK_TARGETS[self._target] if self._target == "log" else {self._target}
        return float(sum(q for n, q in zip(names, quantities) if _name(n) in acceptable))

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_position = self._position(obs) if self.evidence is not None else None
        self._last_goal_quantity = self._goal_quantity(obs) if self.goal_supervision else None
        self._goal_succeeded = False
        return self._annotate(obs, initial=True)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.version >= 3 and info.get("error"):
            labels = obs.copy()
            for key in GOAL_LABEL_SHAPES:
                labels.pop(key, None)
            labels.update(empty_labels(self.version))
            if self.goal_supervision:
                labels.update(empty_goal_labels())
                self._last_goal_quantity = None
            if self.evidence is not None:
                self.evidence._unknown_geometry()
            self._last_position = None
            return labels, reward, done, info
        position = self._position(obs) if self.evidence is not None else None
        labels = self._annotate(obs)
        if self.goal_supervision:
            quantity = self._goal_quantity(obs)
            known_inventory = quantity is not None and self._last_goal_quantity is not None
            gain = known_inventory and quantity > self._last_goal_quantity
            succeeded = bool(info.get("success", False))
            success_now = succeeded and not self._goal_succeeded
            interact = isinstance(action, dict) and any(bool(np.asarray(action.get(k, 0)).any())
                                                         for k in ("attack", "use"))
            labels["goal_interaction_outcome"] = np.uint8(gain or success_now)
            labels["goal_interaction_valid"] = np.uint8(gain or success_now or (interact and known_inventory))
            if labels["goal_interaction_valid"]:
                labels["goal_label_version"] = np.uint8(GOAL_LABEL_VERSION)
            self._goal_succeeded = succeeded or self._goal_succeeded
            self._last_goal_quantity = quantity
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
