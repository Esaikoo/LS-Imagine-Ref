"""T05 CPU sampler check: small replay tables only, no model or environment."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import t00_baseline as baseline
import goal_sampling as sampling


class Report(baseline.Report):
    def finish(self):
        self.data["status"] = "failed" if any(row["level"] == "FAIL" for row in self.data["checks"]) else "passed"
        self.data["elapsed_seconds"] = round(time.perf_counter() - self.started, 3)
        self.save()
        failed = self.data["status"] == "failed"
        print(f"[{'FAIL' if failed else 'PASS'}] T05_SAMPLING_CHECK; report={self.directory / 'report.json'}", flush=True)
        return 2 if failed else 0


def same_rng(left, right):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def rejection(report, name, operation):
    try:
        operation()
    except ValueError as error:
        report.check(name, "PASS", str(error))
        return
    raise ValueError(f"{name} 未拒绝不合法输入")


def check(args, report, cache_dir):
    rng_before = np.random.get_state()
    manifest = json.loads((cache_dir / "cache_manifest.json").read_text(encoding="utf-8"))
    prepared = json.loads((cache_dir / "report.json").read_text(encoding="utf-8"))
    sampling.require(manifest.get("format") == "ls_imagine_bc_cache_v1" and
                     prepared.get("command") == "prepare" and prepared.get("status") in ("passed", "passed_with_warnings") and
                     manifest.get("cache_id") == prepared.get("cache_id"), "需要已经通过 T04 prepare 的训练表")
    with np.load(cache_dir / "tables.npz", allow_pickle=False) as tables:
        remaining, mask = tables["remaining"].copy(), tables["worker_train"].copy()
        candidate_mask = tables["candidate_train"].copy()
        segments = tables["worker_segment"].copy()
    horizon = manifest["source_metadata"]["horizon"]
    sampling.require(segments.shape == mask.shape and np.issubdtype(segments.dtype, np.integer) and
                     candidate_mask.ndim == 1 and candidate_mask.dtype == bool and
                     np.all((segments >= 0) & (segments < len(candidate_mask))) and
                     np.array_equal(mask, candidate_mask[segments]), "训练条目与片段 episode 划分不一致")
    sampling.require(np.any(~mask) and np.any(candidate_mask), "需要真实训练与留出数据")
    candidate_rows = np.flatnonzero(candidate_mask)
    samplers = {mode: sampling.WorkerRowSampler(remaining, mask, horizon, mode) for mode in sampling.MODES}
    report.data["cache_id"] = manifest["cache_id"]
    report.data["tables_sha256"] = hashlib.sha256((cache_dir / "tables.npz").read_bytes()).hexdigest()
    report.check("real_training_tables", "PASS", f"只读训练/留出表；{len(samplers['uniform_rows'].rows)} 个训练动作标签；horizon={horizon}；不加载 states.npy 或冻结权重")

    legacy, current = np.random.RandomState(args.seed), np.random.RandomState(args.seed)
    for _ in range(4):
        old_worker = legacy.choice(np.flatnonzero(mask), size=256, replace=True)
        new_worker = samplers["uniform_rows"].draw(current, 256)
        old_candidate = legacy.choice(candidate_rows, size=256, replace=True)
        new_candidate = current.choice(candidate_rows, size=256, replace=True)
        sampling.require(np.array_equal(old_worker, new_worker) and np.array_equal(old_candidate, new_candidate) and
                         same_rng(legacy.get_state(), current.get_state()), "默认采样改变旧批次或候选 RNG 顺序")
    report.check("legacy_rng_equivalence", "PASS", "默认 uniform_rows 的真实动作/候选批次及 RNG 与旧实现逐项相同")

    results = {}
    for mode, sampler in samplers.items():
        rows = sampler.draw(np.random.RandomState(args.seed), args.samples)
        sampling.require(len(rows) == args.samples and np.all(mask[rows]), "采样混入留出条目")
        counts = sampler.histogram(rows)
        plan = sampler.plan()
        expected = np.asarray(plan["expected_fractions"])
        # Six-sigma marginal bounds avoid an exact-equality requirement for IID draws.
        tolerance = 6 * np.sqrt(args.samples * expected * (1 - expected)) + 1
        sampling.require(np.all(np.abs(counts - args.samples * expected) <= tolerance), f"{mode} 采样比例偏离预声明分布")
        results[mode] = dict(plan, sampled_labels=args.samples, observed_counts=counts.tolist(),
                             observed_fractions=(counts / args.samples).tolist(),
                             diagnostic_count_tolerance=tolerance.tolist())
        print(f"[SAMPLING] {mode} remaining=1 fraction={counts[0] / args.samples:.6f} remaining={horizon} fraction={counts[-1] / args.samples:.6f}", flush=True)

        stream = np.random.RandomState(args.seed)
        sampler.draw(stream, 256)
        saved = stream.get_state()
        expected_worker = sampler.draw(stream, 256)
        expected_candidate = stream.choice(candidate_rows, size=256, replace=True)
        restored = np.random.RandomState(0)
        restored.set_state(saved)
        sampling.require(np.array_equal(expected_worker, sampler.draw(restored, 256)) and
                         np.array_equal(expected_candidate, restored.choice(candidate_rows, size=256, replace=True)) and
                         same_rng(stream.get_state(), restored.get_state()), "采样恢复后的下一批次/RNG 不同")
    baseline.write_json(report.directory / "sampling.json", results)
    report.check("real_sampling_distribution", "PASS", "两种采样均仅来自真实训练条目；remaining 分布符合各自预声明规则")
    report.check("sampler_resume_equivalence", "PASS", "两种方式保存/恢复采样 RNG 后，下一批动作/候选与 RNG 一致；真实模型更新另由 T04 verify 检查")

    missing = mask.copy()
    missing[remaining == horizon] = False
    rejection(report, "empty_budget_guard", lambda: sampling.WorkerRowSampler(remaining, missing, horizon, "uniform_remaining"))
    rejection(report, "budget_range_guard", lambda: sampling.WorkerRowSampler(np.zeros_like(remaining), mask, horizon))
    rejection(report, "split_type_guard", lambda: sampling.WorkerRowSampler(remaining, mask.astype(np.int64), horizon))
    rejection(report, "sampling_mode_guard", lambda: sampling.normalize_training_options({"worker_sampling": "unknown"}))
    legacy_options = {"conditioning": "goal"}
    sampling.require(sampling.normalize_training_options(legacy_options) ==
                     sampling.normalize_training_options(dict(legacy_options, worker_sampling="uniform_rows")) and
                     "worker_sampling" not in legacy_options, "旧选项归一化改变源配置或缺省语义")
    sampling.validate_checkpoint_sampling({"options": legacy_options})
    rejection(report, "balanced_protocol_guard", lambda: sampling.validate_checkpoint_sampling(
        {"options": dict(legacy_options, worker_sampling="uniform_remaining")}))
    rejection(report, "protocol_version_guard", lambda: sampling.validate_checkpoint_sampling(
        {"options": legacy_options, "worker_sampling_protocol": "unknown"}))
    report.check("legacy_options_compatibility", "PASS", "旧缺省只映射 uniform_rows；公平比较核对实际采样模式；不改旧模型内容身份")
    sampling.require(same_rng(rng_before, np.random.get_state()), "采样检查推进调用方 NumPy RNG")
    report.check("scope", "PASS", "采样与兼容性验收；不加载模型或优化器，不启动 MineCLIP/MineDojo；不证明目标控制有效；训练前仍需 T04 完整缓存校验")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.command = "check"
    if args.samples < 16384 or not 0 <= args.seed < 2**32:
        parser.error("samples 至少 16384，seed 需合法非负整数")
    cache_dir = baseline.project_path(args.cache_dir)
    files = [cache_dir / name for name in ("cache_manifest.json", "tables.npz", "report.json")]
    directory = baseline.project_path(args.output_dir) if args.output_dir else ROOT / "relevance_map/t05_outputs" / ("sampling_check_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f"))
    if directory == cache_dir or cache_dir in directory.parents or directory in cache_dir.parents:
        parser.error("输出目录必须独立于训练缓存")
    try:
        before = {str(path): baseline.file_signature(path) for path in files}
        directory.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        print(f"[FAIL] input_or_output_directory: {error}", file=sys.stderr, flush=True)
        return 2
    report = Report(directory, args)
    print(f"OUTPUT_DIR={directory}", flush=True)
    report.data["source_inputs_before"] = before
    try:
        report.save()
        report.require_writable()
        check(args, report, cache_dir)
    except (Exception, KeyboardInterrupt) as error:
        baseline.record_exception(report, error)
    finally:
        try:
            after = {str(path): baseline.file_signature(path) for path in files}
            report.data["source_inputs_after"] = after
            report.check("source_inputs_unchanged", "PASS" if before == after else "FAIL", "源训练表大小/修改时间未变；只写独立诊断目录")
        except OSError as error:
            report.check("source_inputs_unchanged", "FAIL", str(error))
    return report.finish()


if __name__ == "__main__":
    sys.exit(main())
