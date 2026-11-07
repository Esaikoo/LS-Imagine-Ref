"""Build a fixed class bank from real training NPZ images.

Use train_eps directories, or explicitly declared archived training NPZ copies.
Optional reviewed crops can add off-center positives or known distractors. The
annotation file is JSON with a list of entries using source (NPZ basename),
frame, normalized ROI xyxy, role (positive/negative), and note. Negative crops
also declare distractor_type. Training-only visual review is required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from apm.replay import identity

TASKS = ("harvest_log_in_plains", "harvest_water_with_bucket", "harvest_sand",
         "mine_iron_ore", "shear_sheep")
TARGETS = dict(zip(TASKS, ("log", "water_bucket", "sand", "iron_ore", "wool")))
CENTER_ROI = (.375, .375, .625, .625)
CAPTURE_COMMANDS = (4,)*5 + (5,)*10 + (4,)*5 + (3,)*3 + (6,)*10 + (2,)*3 + (6,)*10


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sources_from_args(directories, archives, task):
    sources = set()
    for directory in directories:
        directory = directory.resolve(strict=True)
        if directory.name != "train_eps":
            raise ValueError(f"Expected a train_eps directory, got {directory}")
        sources.update(p.resolve(strict=True) for p in directory.glob("*.npz"))
    sources.update(p.resolve(strict=True) for p in archives)
    for source in sources:
        if source.suffix != ".npz" or any(
                name in {"eval_eps", "evaluation_results", "eval", "test_eps"} for name in source.parts):
            raise ValueError(f"Evaluation data cannot supply references: {source}")
        path_tasks = {name.removeprefix("minedojo_") for name in source.parts}
        if path_tasks.intersection(TASKS) - {task}:
            raise ValueError(f"Source path identifies a different task: {source}")
    if not sources or len({p.name for p in sources}) != len(sources):
        raise ValueError("Supply training NPZ files with unique basenames")
    return sorted(sources)


def candidate_rows(path, reviews):
    with np.load(path, allow_pickle=False) as data:
        if "image" not in data or "log_vnext_world_group" not in data:
            return []
        image = data["image"]
        frames = data["goal_image"] if "goal_image" in data else None
        native = data.get("goal_image_valid", np.zeros(len(image), np.uint8))
        groups = np.asarray(data["log_vnext_world_group"]).reshape(len(image))
        group = int(groups[0])
        if group <= 0 or not np.all(groups == group):
            raise ValueError(f"Missing stable training world group in {path.name}")
        holdout = bool(np.asarray(data.get("goal_holdout", False)).any() or identity(str(group)) % 5 == 0)
        if holdout:
            if reviews:
                raise ValueError(f"Reviewed source {path.name} is a held-out training world")
            return []
        # Older replay is eligible as a reference source only with actual
        # train-only pose evidence and a valid center ray.
        training_marker = bool(np.asarray(data.get("search_pose_valid", False)).any() or
                               np.asarray(data.get("goal_region_valid", False)).any())
        if not training_marker:
            return []
        good = np.asarray(data.get("search_center_target_valid", np.zeros(len(image)))).reshape(-1) > .5
        good &= np.asarray(data.get("search_center_target", np.zeros(len(image)))).reshape(-1) > .5
        good &= np.asarray(data.get("log_env_error", np.zeros(len(image)))).reshape(-1) < .5
        indices = np.flatnonzero(good)
        # Bound per-file memory on large replay sets.
        indices = indices[np.linspace(0, len(indices)-1, min(64, len(indices))).astype(int)]
        choices = reviews if reviews else [{"frame": int(i), "roi": CENTER_ROI, "role": "positive"}
                                          for i in indices]
        records = []
        errors = np.asarray(data.get("log_env_error", np.zeros(len(image)))).reshape(-1)
        for choice in choices:
            frame = int(choice["frame"])
            roi = tuple(float(v) for v in choice.get("roi", CENTER_ROI))
            role = choice.get("role", "positive")
            if (not 0 <= frame < len(image) or len(roi) != 4 or
                    not 0 <= roi[0] < roi[2] <= 1 or not 0 <= roi[1] < roi[3] <= 1 or
                    role not in {"positive", "negative"}):
                raise ValueError(f"Invalid annotation for {path.name}, frame {frame}")
            if reviews and not choice.get("note"):
                raise ValueError("Reviewed examples must explain the visual evidence")
            if errors[frame] > .5:
                raise ValueError(f"A transport-error frame cannot supply an annotation: {path.name}:{frame}")
            rgb = frames[frame] if frames is not None and bool(native[frame]) else image[frame]
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"Invalid RGB data in {path.name}")
            h, w = rgb.shape[:2]
            crop = rgb[round(roi[1]*h):round(roi[3]*h), round(roi[0]*w):round(roi[2]*w)]
            if min(crop.shape[:2]) < 4:
                raise ValueError(f"Reference crop is too small in {path.name}, frame {frame}")
            descriptor = np.asarray(Image.fromarray(crop).resize((8, 8)), np.float32).ravel()/255
            metadata = dict(frame=frame, roi=roi, role=role, world_group=group,
                            source_replay=path.name,
                            annotation=choice.get("note", "valid training center target ray"))
            if role == "negative":
                metadata["distractor_type"] = choice["distractor_type"]
            records.append((path, rgb.copy(), metadata, descriptor))
        return records


def select_diverse(records, limit):
    if not records:
        return []
    vectors = np.stack([entry[3] for entry in records])
    groups = [entry[2]["world_group"] for entry in records]
    distinct = {}
    distances = np.ones(len(records), np.float32)
    chosen = []
    for _ in range(min(limit, len(records))):
        balance = np.array([1/(1+distinct.get(g, 0)) for g in groups])
        index = int(np.argmax(distances*balance))
        if distances[index] < 1e-6:
            break
        chosen.append(records[index])
        distinct[groups[index]] = distinct.get(groups[index], 0)+1
        distances = np.minimum(distances, ((vectors-vectors[index])**2).mean(-1))
        distances[index] = -1
    return chosen


def build(args):
    paths = sources_from_args(args.train_replay, args.verified_train_archive, args.task)
    annotations = json.loads(args.annotations.read_text()) if args.annotations else []
    world_seeds = json.loads(args.world_seeds.read_text()) if args.world_seeds else {}
    for parent in {path.parent.parent for path in paths if path.parent.name == "train_eps"}:
        capture_manifest = parent/"capture.json"
        if capture_manifest.is_file():
            capture = json.loads(capture_manifest.read_text())
            if capture["task_id"] != args.task or capture["split"] != "train":
                raise ValueError(f"Incompatible capture provenance: {capture_manifest}")
            for world in capture["worlds"]:
                group, seed = str(world["world_group"]), world["world_seed"]
                if group in world_seeds and int(world_seeds[group]) != seed:
                    raise ValueError(f"Conflicting seed provenance for world group {group}")
                world_seeds[group] = seed
    if {a["source"] for a in annotations} - {p.name for p in paths}:
        raise ValueError("Annotations name a source that was not supplied")
    records = []
    for path in paths:
        records.extend(candidate_rows(path, [a for a in annotations if a["source"] == path.name]))
        if len(records) > 768:
            records = (select_diverse([r for r in records if r[2]["role"] == "positive"], 384)
                       + select_diverse([r for r in records if r[2]["role"] == "negative"], 128))
    positives = select_diverse([r for r in records if r[2]["role"] == "positive"], args.count)
    negatives = select_diverse([r for r in records if r[2]["role"] == "negative"], args.count)
    if len(positives) < args.count:
        raise ValueError(f"Found {len(positives)} distinct confirmed references; requested {args.count}")
    groups = {r[2]["world_group"] for r in positives}
    if len(groups) < args.min_worlds:
        raise ValueError(f"Only {len(groups)} distinct training world groups; requested {args.min_worlds}")
    output = args.output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing reference bank {output}")
    source_sha = {p: file_sha(p) for p in {r[0] for r in positives+negatives}}
    with tempfile.TemporaryDirectory(prefix=".goal-bank-", dir=output.parent) as temporary:
        staging = Path(temporary)/"bank"
        staging.mkdir()
        entries = []
        for i, (source, rgb, metadata, _) in enumerate(positives+negatives):
            name = f"{args.task}_{i:03d}.png"
            image_path = staging/name
            Image.fromarray(rgb).save(image_path)
            entries.append(dict(metadata, task_id=args.task, split="train", path=name,
                                world_seed=world_seeds.get(str(metadata["world_group"])),
                                source_sha256=source_sha[source], sha256=file_sha(image_path)))
        manifest = dict(version=1, task_id=args.task, entries=entries,
                        provenance="MineDojo training frames; no evaluation-derived assets")
        (staging/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n",
                                            encoding="utf-8")
        staging.rename(output)
    print(f"{output}: {len(positives)} positive references across {len(groups)} train world groups, "
          f"{len(negatives)} reviewed negatives")


def collect(args):
    """Bounded reference preparation with the original train env and actuator."""
    from types import SimpleNamespace
    import yaml
    os.environ.setdefault("MINEDOJO_HEADLESS", "1")
    from expr import make_env
    import tools

    seeds = args.collect_world_seeds
    if len(set(seeds)) != len(seeds) or not 1 <= len(seeds) <= 8 or any(s < 0 for s in seeds):
        raise ValueError("Supply 1 to 8 distinct nonnegative training world seeds")
    config_file = Path(__file__).resolve().parents[1]/"configs.yaml"
    configs = yaml.safe_load(config_file.read_text())
    config = {}
    for group in ("defaults", "minedojo", "apm_vnext", "apm_vnext4"):
        config.update(configs[group])
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    train_eps = output/"train_eps"
    train_eps.mkdir()
    manifest = dict(version=1, split="train", task_id=args.task,
                    commands=list(args.capture_commands or CAPTURE_COMMANDS),
                    max_steps_per_world=args.capture_steps, worlds=[])
    config.update(task="minedojo_"+args.task, target_item=TARGETS[args.task],
                  logdir=str(output), results_dir=str(output/"environment"), name="goal_reference_capture",
                  goal_ref_enabled=True, goal_ref_use_raw_image=True, vnext_visual_audit=False,
                  minedojo_gradle_offline=args.minedojo_gradle_offline,
                  episode_max_steps_override=args.capture_steps)
    commands = manifest["commands"]
    actions = np.eye(12, dtype=np.float32)
    for seed in seeds:
        env = make_env(SimpleNamespace(**dict(config, env_seed=seed)), "train", 0)
        try:
            cache = {}
            episode_index, completed = 0, 0
            observation = env.reset()
            group = int(observation["log_vnext_world_group"])
            name = f"capture-{seed}-{episode_index}"
            tools.add_to_cache(cache, name, dict(observation, action=np.zeros(12,np.float32),
                                                reward=np.float32(0), discount=np.float32(1)))
            for step in range(args.capture_steps):
                action = actions[commands[step % len(commands)]]
                observation, reward, done, info = env.step({"action": action})
                if info.get("error", False):
                    break
                if int(observation["log_vnext_world_group"]) != group:
                    raise RuntimeError("Training world changed without a reset")
                tools.add_to_cache(cache, name, dict(observation, action=action,
                    reward=np.float32(reward), discount=np.float32(not observation["is_terminal"])))
                completed += 1
                if done and step+1 < args.capture_steps:
                    observation = env.reset()
                    if int(observation["log_vnext_world_group"]) != group:
                        raise RuntimeError("A fixed capture world seed changed group across resets")
                    episode_index += 1
                    name = f"capture-{seed}-{episode_index}"
                    tools.add_to_cache(cache, name, dict(observation, action=np.zeros(12,np.float32),
                                                        reward=np.float32(0), discount=np.float32(1)))
            tools.save_episodes(train_eps, {key: episode for key, episode in cache.items()
                                           if len(episode["reward"]) > 1})
            manifest["worlds"].append(dict(world_seed=seed, world_group=group,
                heldout=identity(str(group)) % 5 == 0, completed_steps=completed,
                requested_steps_completed=completed == args.capture_steps))
            pending = output/"capture.json.tmp"
            pending.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
            pending.replace(output/"capture.json")
            print(f"Training world {seed}: {completed} real actions, group {group}, "
                  f"held out {manifest['worlds'][-1]['heldout']}")
        finally:
            env.close()
    if not all(world["requested_steps_completed"] for world in manifest["worlds"]):
        raise RuntimeError(f"Capture was interrupted. Real recorded prefixes are in {train_eps}.")
    print(f"Training frames collected in {train_eps}. Build a bank or add reviewed crop annotations next.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--train-replay", type=Path, action="append", default=[])
    parser.add_argument("--verified-train-archive", type=Path, action="append", default=[],
                        help="An explicitly verified copy of a real training NPZ outside train_eps")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--world-seeds", type=Path, help="JSON mapping opaque training world group to recorded seed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, choices=range(1, 33), default=16)
    parser.add_argument("--min-worlds", type=int, choices=range(1, 33), default=2)
    parser.add_argument("--collect-world-seeds", type=int, nargs="+", metavar="SEED",
                        help="Collect training-world samples instead of building a bank")
    parser.add_argument("--capture-steps", type=int, choices=range(1, 2001), default=128, metavar="N")
    parser.add_argument("--capture-commands", type=int, nargs="+", choices=range(12),
                        metavar="ACTION_ID", help="Repeat the supplied 12-action command sequence")
    parser.add_argument("--minedojo-gradle-offline", action="store_true")
    args = parser.parse_args()
    if args.collect_world_seeds:
        if args.train_replay or args.verified_train_archive or args.annotations or args.world_seeds:
            parser.error("Capture creates a new training source; run the bank builder separately")
        collect(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
