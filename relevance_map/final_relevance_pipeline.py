#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Smoke/integration test for the final LS-Imagine relevance pipeline.

Expected production design
--------------------------
MAP:
    Value + Temporal(L=1)
    -> raw cosine
    -> (cos+1)/2
    -> Natural Zoom
    -> obs['heatmap']

SCORESTORAGE:
    NACLIP + Temporal(L=1)
    -> raw P85
    -> obs['score'] / obs['score_on_zoomed']
    -> unchanged tools.ScoreStorage

Run from LS-Imagine project root in the real training environment:

    python relevance_map/test_final_relevance_pipeline.py

No argparse: edit globals below if needed.
"""

from pathlib import Path
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# =====================================================================
# GLOBAL CONFIG
# =====================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

# If the script is elsewhere, set manually:
# PROJECT_ROOT = Path(r"/home/user1/dl/projects/LS-Imagine-Ref")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

TASK_NAME = "harvest_log_in_plains"
NUM_STEPS = 80
SEED = 0

# Run the expensive independent formula check on reset and first accepted zoom.
CHECK_FORMULA_EXACTLY = True

# Do not fail if random rollout produces no accepted zoom; print a warning.
REQUIRE_ACCEPTED_ZOOM = False

# A real log_dir is required because task_specs.yaml enables the screenshot
# wrapper entry even when reset_flag/step_flag are False.
TEST_LOG_DIR = (
    PROJECT_ROOT
    / "relevance_map"
    / "_final_pipeline_smoke_logs"
)


# =====================================================================
# HELPERS
# =====================================================================

def scalar(x):
    a = np.asarray(x)
    return float(a.reshape(-1)[0])


def flag(x):
    return bool(np.asarray(x).reshape(-1)[0])


def find_wrapper(env, wrapper_name):
    cur = env
    while cur is not None:
        explicit_name = getattr(cur, "wrapper_name", None)
        class_name = cur.__class__.__name__
        if explicit_name == wrapper_name or class_name == wrapper_name:
            return cur
        cur = getattr(cur, "env", None)
    return None


def assert_finite(name, x):
    value = scalar(x)
    assert np.isfinite(value), f"{name} is not finite: {value}"
    return value


def get_vision_count(clip):
    """
    The modified ClipReward may expose vision_forward_count for reuse checks.
    Return None instead of crashing if an older local file does not.
    """
    value = getattr(clip, "vision_forward_count", None)
    if value is None:
        return None
    return int(value)


def vision_delta(before, after):
    if before is None or after is None:
        return None
    return int(after - before)


def expected_outer_heatmap(concentration, is_zoomed=False):
    """Reproduce LSImagineWrapper's heatmap resize + uint8 path."""
    hm = concentration.get_heatmap(is_zoomed=is_zoomed)
    hm64 = cv2.resize(hm, (64, 64))
    hm64 = np.clip(hm64 * 255.0, 0, 255).astype(np.uint8)
    return hm64


@torch.no_grad()
def recompute_current_formula(clip, concentration, prompts):
    """
    Independently recompute BOTH final formulas from the current 160x256 RGB.

    Returns:
        map_max_abs_error
        p85_abs_error
        diagnostics
    """
    frame = torch.from_numpy(
        concentration.curr_frame
    ).permute(2, 0, 1).contiguous()

    bundle = clip.make_bundle_from_frame(
        frame,
        need_global=False,
    )

    # ------------------------------------------------------------
    # A) Value + Temporal(L=1) -> cosine -> (cos+1)/2
    # ------------------------------------------------------------
    sim = clip.get_patch_similarity(
        bundle,
        prompts,
    )

    relevance = torch.clamp(
        (sim + 1.0) / 2.0,
        0.0,
        1.0,
    )

    B, P, N = relevance.shape
    gh, gw = bundle["grid_size"]
    assert N == gh * gw

    relevance = relevance.reshape(
        B * P,
        1,
        gh,
        gw,
    )

    relevance = F.interpolate(
        relevance,
        size=concentration.resolution,
        mode="bilinear",
        align_corners=False,
    )

    relevance = relevance.reshape(
        B,
        P,
        concentration.resolution[0],
        concentration.resolution[1],
    )[0]

    expected_map = (
        relevance.max(dim=0).values
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    actual_map = (
        concentration.mask
        / 255.0
    ).astype(np.float32)

    map_error = float(
        np.max(
            np.abs(
                expected_map
                - actual_map
            )
        )
    )

    # ------------------------------------------------------------
    # B) NACLIP + Temporal(L=1) -> P85
    # ------------------------------------------------------------
    expected_p85 = float(
        clip.get_naclip_progress_score(
            bundle,
            prompts,
            percentile=concentration.progress_percentile,
        )[0].item()
    )

    actual_p85 = concentration.get_progress_score(
        is_zoomed=False
    )

    p85_error = abs(
        expected_p85
        - actual_p85
    )

    value_raw = (
        sim.max(dim=1).values[0]
        .detach()
        .cpu()
        .numpy()
    )

    diagnostics = {
        "map_min": float(expected_map.min()),
        "map_max": float(expected_map.max()),
        "map_mean": float(expected_map.mean()),
        "map_std": float(expected_map.std()),
        "value_raw_p85": float(
            np.percentile(value_raw, 85)
        ),
        "naclip_p85": expected_p85,
    }

    return map_error, p85_error, diagnostics


@torch.no_grad()
def recompute_zoomed_p85(clip, concentration, prompts):
    frame = torch.from_numpy(
        concentration.zoomed_frame
    ).permute(2, 0, 1).contiguous()

    bundle = clip.make_bundle_from_frame(
        frame,
        need_global=False,
    )

    expected = float(
        clip.get_naclip_progress_score(
            bundle,
            prompts,
            percentile=concentration.progress_percentile,
        )[0].item()
    )

    actual = concentration.get_progress_score(
        is_zoomed=True
    )

    return expected, actual, abs(expected - actual)


def check_public_obs(obs):
    required = [
        "image",
        "heatmap",
        "jump",
        "is_zoomed",
        "is_calculated",
        "is_first",
        "is_last",
        "is_terminal",
        "reward_on_zoomed",
        "intrinsic",
        "intrinsic_on_zoomed",
        "score",
        "score_on_zoomed",
        "jumping_steps",
        "accumulated_reward",
    ]

    missing = [
        k
        for k in required
        if k not in obs
    ]

    assert not missing, f"Missing public obs keys: {missing}"

    # Internal MineCLIP cache must not enter replay/world-model obs.
    assert "_mineclip_bundle" not in obs

    assert np.asarray(obs["image"]).shape == (64, 64, 3)

    hm = np.asarray(obs["heatmap"])
    assert hm.shape in [
        (64, 64),
        (64, 64, 1),
    ], hm.shape
    assert hm.dtype == np.uint8

    cur = assert_finite("score", obs["score"])
    assert -1.0001 <= cur <= 1.0001, (
        "obs['score'] should now be raw cosine percentile, "
        f"got {cur}"
    )

    z = assert_finite(
        "score_on_zoomed",
        obs["score_on_zoomed"],
    )

    if flag(obs["is_zoomed"]):
        assert -1.0001 <= z <= 1.0001
    else:
        assert abs(z) < 1e-8, (
            "Non-zoomed transition should keep score_on_zoomed=0"
        )


# =====================================================================
# MAIN
# =====================================================================

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    from envs.tasks import make
    from tools import ScoreStorage

    print("=" * 100)
    print("FINAL RELEVANCE PIPELINE SMOKE TEST")
    print("=" * 100)
    print("Task:", TASK_NAME)

    # task_specs.yaml contains screenshot_specs, so the existing
    # MinedojoScreenshotWrapper requires a non-None log_dir even when
    # reset_flag/step_flag are both False.
    TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = make(
        TASK_NAME,
        log_dir=str(TEST_LOG_DIR),
    )

    concentration_wrapper = find_wrapper(
        env,
        "ConcentrationWrapper",
    )
    clip_wrapper = find_wrapper(
        env,
        "ClipWrapper",
    )

    assert concentration_wrapper is not None, (
        "ConcentrationWrapper not found"
    )
    assert clip_wrapper is not None, (
        "ClipWrapper not found"
    )

    concentration = concentration_wrapper.concentration
    clip = clip_wrapper.clip

    # One shared MineCLIP instance is essential.
    assert concentration.clip is clip
    assert concentration.model is clip.model

    required_clip_methods = [
        "make_bundle_from_frame",
        "get_patch_similarity",
        "get_naclip_progress_score",
        "get_naclip_patch_similarity",
    ]
    missing_clip_methods = [
        name
        for name in required_clip_methods
        if not hasattr(clip, name)
    ]
    assert not missing_clip_methods, (
        "Modified ClipReward is missing required methods: "
        f"{missing_clip_methods}"
    )

    required_concentration_attrs = [
        "progress_percentile",
        "get_progress_score",
        "get_heatmap",
        "curr_frame",
        "mask",
    ]
    missing_concentration_attrs = [
        name
        for name in required_concentration_attrs
        if not hasattr(concentration, name)
    ]
    assert not missing_concentration_attrs, (
        "Modified ConcentrationReward is missing required API: "
        f"{missing_concentration_attrs}"
    )

    print("[PASS] ClipWrapper and ConcentrationWrapper share one MineCLIP")
    print(
        "progress_percentile =",
        concentration.progress_percentile,
    )
    print(
        "NACLIP Gaussian =",
        getattr(clip, "naclip_gaussian_std", "<missing>"),
        getattr(clip, "naclip_gaussian_weight", "<missing>"),
        "include_cls=",
        getattr(clip, "naclip_include_cls", "<missing>"),
    )
    print(
        "Natural Zoom config: cooldown=",
        concentration.zoom_cooldown_steps,
        "max_zoom_factor=",
        concentration.max_zoom_factor,
    )

    # ------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------
    before = get_vision_count(clip)
    obs = env.reset()
    after = get_vision_count(clip)
    reset_vision_delta = vision_delta(before, after)

    check_public_obs(obs)

    print()
    print("RESET")
    print(
        "  vision passes =",
        reset_vision_delta
        if reset_vision_delta is not None
        else "<counter unavailable>",
    )
    print("  score/P85     =", scalar(obs["score"]))
    print("  is_zoomed     =", flag(obs["is_zoomed"]))
    print(
        "  heatmap uint8 =",
        int(np.min(obs["heatmap"])),
        "..",
        int(np.max(obs["heatmap"])),
    )

    # Normal current frame must require only one ViT encoding.
    # A second pass is allowed only if a Natural-Zoom candidate was actually
    # evaluated on new pixels.
    if reset_vision_delta is not None:
        assert 1 <= reset_vision_delta <= 2, (
            "Unexpected reset vision passes: "
            f"{reset_vision_delta}. "
            "Expected current RGB once, plus at most one zoomed RGB."
        )
    else:
        print(
            "[WARN] clip.vision_forward_count is unavailable; "
            "skip the vision-reuse counter check."
        )

    # Outer LSImagineWrapper must contain exactly the map produced by
    # ConcentrationReward, only resized/quantized.
    expected_hm = expected_outer_heatmap(
        concentration,
        is_zoomed=False,
    )
    actual_hm = np.asarray(obs["heatmap"])
    actual_hm = np.squeeze(actual_hm)
    expected_hm = np.squeeze(expected_hm)

    hm_uint8_error = int(
        np.max(
            np.abs(
                actual_hm.astype(np.int16)
                - expected_hm.astype(np.int16)
            )
        )
    )

    print("  outer heatmap exact uint8 error =", hm_uint8_error)
    assert hm_uint8_error == 0

    if CHECK_FORMULA_EXACTLY:
        map_err, p85_err, diag = recompute_current_formula(
            clip,
            concentration,
            concentration_wrapper.prompt,
        )

        print()
        print("INDEPENDENT FORMULA CHECK")
        print("  map max abs error =", map_err)
        print("  P85 abs error     =", p85_err)
        print("  diagnostics       =", diag)

        assert map_err < 1e-5, (
            "Production heatmap is not exactly Value+Temporal -> (cos+1)/2"
        )
        assert p85_err < 1e-6, (
            "Production score is not exactly NACLIP+Temporal raw P85"
        )

        assert abs(
            scalar(obs["score"])
            - concentration.get_progress_score(False)
        ) < 1e-6

        print("[PASS] Map formula is exactly raw-shifted Value cosine")
        print("[PASS] obs['score'] is exactly current NACLIP-P85")

    # ------------------------------------------------------------
    # Reuse the UNCHANGED official ScoreStorage implementation.
    # ------------------------------------------------------------
    storage = ScoreStorage(
        max_steps=concentration_wrapper.max_steps
    )
    env_id = "smoke_test"

    current_step = 0
    if flag(obs["is_zoomed"]):
        storage.add(
            env_id,
            current_step,
            scalar(obs["score_on_zoomed"]),
        )

    accepted_zooms = int(
        flag(obs["is_zoomed"])
    )
    resolved_pairs = 0
    checked_zoom_formula = False

    # ------------------------------------------------------------
    # STEP LOOP
    # ------------------------------------------------------------
    print()
    print("RUNNING ENV STEPS")

    for t in range(1, NUM_STEPS + 1):
        action = env.action_space.sample()

        before = get_vision_count(clip)
        obs, reward, done, info = env.step(action)
        after = get_vision_count(clip)
        step_vision_delta = vision_delta(before, after)

        check_public_obs(obs)

        # Current normal frame once; optional new zoomed RGB once.
        if step_vision_delta is not None:
            assert 1 <= step_vision_delta <= 2, (
                f"Unexpected vision passes at step {t}: "
                f"{step_vision_delta}. "
                "Expected current RGB once, plus at most one zoomed RGB."
            )

        cur_score = scalar(obs["score"])
        zoomed = flag(obs["is_zoomed"])

        if zoomed:
            accepted_zooms += 1
            zscore = scalar(
                obs["score_on_zoomed"]
            )

            # Most important symmetry property:
            # both are raw NACLIP percentile values, not composite scores.
            assert -1.0001 <= zscore <= 1.0001

            if not done:
                storage.add(
                    env_id,
                    t,
                    zscore,
                )

            if (
                CHECK_FORMULA_EXACTLY
                and not checked_zoom_formula
            ):
                expected, actual, err = recompute_zoomed_p85(
                    clip,
                    concentration,
                    concentration_wrapper.prompt,
                )

                print()
                print("FIRST ACCEPTED ZOOM FORMULA CHECK")
                print("  step              =", t)
                print("  current P85       =", cur_score)
                print("  zoomed P85 obs    =", zscore)
                print("  zoomed P85 direct =", expected)
                print("  abs error         =", err)
                print(
                    "  zoom factor       =",
                    getattr(
                        concentration,
                        "actual_zoom_factor",
                        "<missing>",
                    ),
                )

                assert err < 1e-6
                assert abs(zscore - actual) < 1e-6

                expected_zoom_hm = expected_outer_heatmap(
                    concentration,
                    is_zoomed=True,
                )
                actual_zoom_hm = np.squeeze(
                    np.asarray(
                        obs["heatmap_on_zoomed"]
                    )
                )
                expected_zoom_hm = np.squeeze(
                    expected_zoom_hm
                )

                zoom_hm_error = int(
                    np.max(
                        np.abs(
                            actual_zoom_hm.astype(np.int16)
                            - expected_zoom_hm.astype(np.int16)
                        )
                    )
                )
                assert zoom_hm_error == 0

                print("[PASS] score_on_zoomed is exactly zoomed NACLIP-P85")
                print("[PASS] heatmap_on_zoomed follows raw-shifted map")
                checked_zoom_formula = True

        resolved = storage.get_and_remove_less_than(
            env_id,
            t,
            cur_score,
        )

        if resolved:
            resolved_pairs += len(resolved)
            print(
                f"  ScoreStorage resolve @ t={t}: "
                f"sources={resolved}, current_P85={cur_score:.4f}"
            )

        if (
            t == 1
            or t % 10 == 0
            or zoomed
            or done
        ):
            print(
                f"  t={t:03d} "
                f"vision={step_vision_delta if step_vision_delta is not None else 'NA'} "
                f"P85={cur_score:.4f} "
                f"zoom={int(zoomed)} "
                f"map=[{int(np.min(obs['heatmap']))},"
                f"{int(np.max(obs['heatmap']))}] "
                f"gate={getattr(concentration, 'zoom_gate_reason', 'NA')}"
            )

        if done:
            print("  episode finished at", t)
            break

    # ------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------
    print()
    print("=" * 100)
    print("RESULT")
    print("=" * 100)
    print("accepted zooms             =", accepted_zooms)
    print("ScoreStorage resolved pairs=", resolved_pairs)
    print(
        "ScoreStorage pending        =",
        storage.count_data_pairs(env_id),
    )
    print(
        "total MineCLIP vision passes=",
        get_vision_count(clip),
    )

    if accepted_zooms == 0:
        msg = (
            "No accepted Natural-Zoom event occurred in this short random "
            "rollout. Core current-map/current-P85 checks still passed. "
            "Increase NUM_STEPS or manually control the agent toward a tree "
            "to test score_on_zoomed."
        )
        if REQUIRE_ACCEPTED_ZOOM:
            raise AssertionError(msg)
        print("[WARN]", msg)

    print()
    print("[PASS] Public obs dictionary names are unchanged")
    print("[PASS] Existing tools.ScoreStorage accepts the new scores unchanged")
    if get_vision_count(clip) is not None:
        print("[PASS] Current frame uses one shared MineCLIP vision encoding")
    else:
        print(
            "[WARN] Vision-pass reuse was not counted because "
            "vision_forward_count is unavailable."
        )
    print("[PASS] Final pipeline smoke test completed")

    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
