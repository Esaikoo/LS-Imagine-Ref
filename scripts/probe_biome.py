#!/usr/bin/env python3
"""Record fresh MineDojo worlds with a stationary 360-degree camera sweep.

No policy, checkpoint, training, movement, teleport or spawn correction is used.
The specified-biome generator is the same one selected by HarvestMeta when
specified_biome is supplied. Outputs default to an explicitly supplied directory.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_checkpoints import Mp4Writer, write_json


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value.tolist() if hasattr(value, "tolist") else value


def camera_yaw(obs):
    import numpy as np
    return float(np.asarray(obs["location_stats"]["yaw"]).reshape(-1)[0])


def describe(obs, info):
    import numpy as np
    voxels = np.asarray(obs["voxels"]["block_name"]).astype(str)
    # Voxel offsets: x,z=-1..1; y=-1..2. Feet y=0, head voxel y=1.
    feet, head = str(voxels[1, 1, 1]), str(voxels[1, 2, 1])
    is_water = lambda name: name.removeprefix("minecraft:") in ("water", "flowing_water")
    return dict(location=plain(obs["location_stats"]), life=plain(obs["life_stats"]),
                feet_block=feet, head_block=head,
                feet_in_water=is_water(feet), head_in_water=is_water(head),
                vertical_blocks=voxels[1, :, 1].tolist(),
                raw_water_flags={k: plain(v) for k, v in info.items()
                                 if "water" in k.lower() and not isinstance(v, dict)},
                world_time=plain(info.get("world_time")))


def worker(args):
    os.environ["MINEDOJO_HEADLESS"] = "1"
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image
    from minedojo.sim import MineDojoSim
    from minedojo.sim.mc_meta import mc

    seed = args.worker_seed
    folder = args.output / ("seed_%d" % seed)
    folder.mkdir(parents=True, exist_ok=False)
    env = writer = None
    result = dict(seed=seed, biome=args.biome, status="running")
    trace = []
    started = time.monotonic()
    try:
        env = MineDojoSim(
            image_size=(args.height, args.width), seed=seed, world_seed=str(seed),
            generate_world_type="default" if args.biome == "default" else "specified_biome",
            specified_biome=None if args.biome == "default" else args.biome,
            regenerate_world_after_reset=True, event_level_control=False,
            start_time=6000, allow_time_passage=False, initial_weather="clear",
            break_speed_multiplier=100., use_voxel=True,
            voxel_size=dict(xmin=-1, xmax=1, ymin=-1, ymax=2, zmin=-1, zmax=1))
        (folder / "configured_mission.xml").write_text(
            env._sim_spec.to_xml("biome-probe-%d" % seed), encoding="utf-8")
        writer = Mp4Writer(folder / "lookaround.mp4", args.fps,
                           imageio_ffmpeg.get_ffmpeg_exe())

        def capture(obs, info, phase, commanded_yaw=0., snapshot=None):
            rgb = np.asarray(obs["rgb"])
            if rgb.shape[0] == 3:
                rgb = np.moveaxis(rgb, 0, -1)
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
            writer.append_data(rgb)
            row = dict(frame=len(trace), phase=phase, commanded_yaw=commanded_yaw,
                       **describe(obs, info))
            trace.append(row)
            if snapshot:
                Image.fromarray(rgb).save(folder / snapshot)
            return row

        print("RESET seed=%d biome=%s" % (seed, args.biome), flush=True)
        obs = env.reset()
        info = env.prev_info
        result["reset"] = capture(obs, info, "reset", snapshot="reset.png")

        def step(yaw, phase, snapshot=None):
            action = env.action_space.no_op()
            action["camera"] = np.array([0., yaw], dtype=np.float32)
            obs, _, done, info = env.step(action)
            if done:
                raise RuntimeError("Simulator terminated during %s" % phase)
            capture(obs, info, phase, yaw, snapshot)
            return obs, info

        # Preserve the raw reset frame; only gravity/render synchronization can
        # change position during these neutral steps. No water escape is added.
        for _ in range(args.settle_steps):
            obs, info = step(0., "settle")
        result["turn_start"] = describe(obs, info)
        Image.fromarray(np.moveaxis(obs["rgb"], 0, -1)).save(folder / "turn_start.png")
        previous_yaw = camera_yaw(obs)
        actual_rotation = 0.
        delta = 360. / args.turn_steps
        for index in range(args.turn_steps):
            snapshot = ("view_%03d.png" % round((index+1)*delta)
                        if (index+1)*4 % args.turn_steps == 0 else None)
            obs, info = step(delta, "turn", snapshot)
            yaw = camera_yaw(obs)
            actual_rotation += (yaw - previous_yaw + 180.) % 360. - 180.
            previous_yaw = yaw
        result["turn_end"] = describe(obs, info)
        for _ in range(10):
            obs, info = step(0., "finish")
        expected_biome = None if args.biome == "default" else int(mc.BIOMES_MAP[args.biome])
        observed_biome = int(np.asarray(obs["location_stats"]["biome_id"]).item())
        result.update(status="completed", expected_biome_id=expected_biome,
                      observed_biome_id=observed_biome,
                      actual_yaw_degrees=actual_rotation,
                      commanded_yaw_degrees=360., frames=len(trace),
                      horizontal_drift=float(np.linalg.norm(
                          np.asarray(result["turn_end"]["location"]["pos"])[[0, 2]] -
                          np.asarray(result["turn_start"]["location"]["pos"])[[0, 2]])))
        if abs(actual_rotation-360.) > 2.:
            raise RuntimeError("Observed camera sweep was not a full turn: %s" % actual_rotation)
        if expected_biome is not None and observed_biome != expected_biome:
            raise RuntimeError("Observed biome differs from the requested generator")
    except BaseException:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                result["wall_seconds"] = time.monotonic()-started
                with (folder / "trace.jsonl").open("w", encoding="utf-8") as stream:
                    for row in trace:
                        stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                write_json(folder / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--biome", default="beach", help="Biome name, or default for natural generation")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--turn-steps", type=int, default=72)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.count < 1 or args.turn_steps < 4 or args.settle_steps < 0 or args.fps < 1
            or min(args.height, args.width) < 2 or args.height % 2 or args.width % 2):
        parser.error("Positive counts/FPS, >=4 turn steps and even image dimensions are required")
    args.output = args.output.expanduser().resolve()
    if args.worker_seed is not None:
        worker(args)
        return
    seeds = args.seeds or secrets.SystemRandom().sample(range(1, 2**31-1), args.count)
    if len(set(seeds)) != len(seeds):
        parser.error("Duplicate seeds would reuse output directories")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        parser.error("Use a fresh output directory to preserve the previous probe")
    manifest = dict(biome=args.biome, seeds=seeds, image_size=[args.height, args.width],
                    fps=args.fps, settle_steps=args.settle_steps, turn_steps=args.turn_steps,
                    rotation_degrees=360, start_time=6000, initial_weather="clear",
                    allow_time_passage=False, fast_reset=False, models_loaded=False,
                    movement_commands=False, passive_voxel_audit=True)
    write_json(args.output / "manifest.json", manifest)
    results = []
    for seed in seeds:
        write_json(args.output / "status.json", dict(status="running", seed=seed,
                                                      completed=len(results), total=len(seeds)))
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--output", str(args.output), "--worker-seed", str(seed),
                   "--biome", args.biome, "--height", str(args.height),
                   "--width", str(args.width), "--fps", str(args.fps),
                   "--turn-steps", str(args.turn_steps), "--settle-steps", str(args.settle_steps)]
        with (args.output / ("seed_%d.log" % seed)).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, MINEDOJO_HEADLESS="1"))
            try:
                code = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                import psutil
                owned = psutil.Process(process.pid).children(recursive=True)
                for child in reversed(owned):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait()
                code = -1
        result_path = args.output / ("seed_%d" % seed) / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else dict(seed=seed)
        result["exit_code"] = code
        results.append(result)
        write_json(args.output / "summary.json", dict(seeds=seeds, results=results))
        print("SEED %d: %s (exit %d)" % (seed, result.get("status", "failed"), code), flush=True)
    okay = all(r.get("status") == "completed" and r["exit_code"] == 0 for r in results)
    write_json(args.output / "status.json", dict(status="completed" if okay else "failed",
                                                  completed=len(results), total=len(seeds)))
    if not okay:
        raise SystemExit("One or more biome probes failed; see per-seed logs")


if __name__ == "__main__":
    main()
