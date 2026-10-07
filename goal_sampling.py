"""Replay-row sampling policies; NumPy only, without model/environment imports."""

import copy

import numpy as np


MODES = ("uniform_rows", "uniform_remaining")
PROTOCOLS = {"uniform_rows": "expanded_row_uniform_v1",
             "uniform_remaining": "remaining_uniform_then_row_uniform_v1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_training_options(options):
    """Compare legacy options semantically without changing inference identities."""
    result = copy.deepcopy(options)
    result.setdefault("worker_sampling", "uniform_rows")
    require(result["worker_sampling"] in MODES, "未知低层采样方式")
    return result


def checkpoint_sampling_protocol(options):
    return PROTOCOLS[normalize_training_options(options)["worker_sampling"]]


def validate_checkpoint_sampling(payload):
    mode = normalize_training_options(payload["options"])["worker_sampling"]
    protocol = payload.get("worker_sampling_protocol")
    # Before this option existed, every checkpoint used expanded-row sampling.
    require(protocol == PROTOCOLS[mode] or (protocol is None and mode == "uniform_rows"),
            "checkpoint 采样协议缺失或不兼容")


class WorkerRowSampler:
    def __init__(self, remaining, training_mask, horizon, mode="uniform_rows"):
        require(mode in MODES, "未知低层采样方式")
        require(type(horizon) is int and horizon >= 1, "采样 horizon 必须为正整数")
        remaining, training_mask = np.asarray(remaining), np.asarray(training_mask)
        require(remaining.ndim == 1 and np.issubdtype(remaining.dtype, np.integer) and
                training_mask.shape == remaining.shape and training_mask.dtype == bool and
                np.all((remaining >= 1) & (remaining <= horizon)), "采样标签或训练划分错误")
        self.remaining = remaining.copy()
        self.horizon, self.mode = horizon, mode
        self.rows = np.flatnonzero(training_mask)
        require(len(self.rows) > 0, "低层训练采样池为空")
        self.bins = [self.rows[self.remaining[self.rows] == h] for h in range(1, horizon + 1)]
        self.bin_sizes = np.array([len(pool) for pool in self.bins], dtype=np.int64)
        if mode == "uniform_remaining":
            require(np.all(self.bin_sizes > 0), "均衡采样需要每个 remaining 都有真实训练样本")

    def draw(self, rng, size):
        require(type(size) is int and size > 0, "采样批次必须为正整数")
        if self.mode == "uniform_rows":
            # Keep the exact legacy call and RNG consumption order.
            return rng.choice(self.rows, size=size, replace=True)
        budgets = rng.randint(1, self.horizon + 1, size=size)
        within_bin = rng.random_sample(size)
        result = np.empty(size, dtype=np.int64)
        for h, pool in enumerate(self.bins, 1):
            chosen = budgets == h
            offsets = (within_bin[chosen] * len(pool)).astype(np.int64)
            result[chosen] = pool[offsets]
        return result

    def histogram(self, rows):
        return np.bincount(self.remaining[rows], minlength=self.horizon + 1)[1:]

    def plan(self):
        source = self.bin_sizes / len(self.rows)
        expected = source if self.mode == "uniform_rows" else np.full(self.horizon, 1 / self.horizon)
        return {"mode": self.mode, "protocol": PROTOCOLS[self.mode],
                "remaining_values": list(range(1, self.horizon + 1)),
                "training_labels": len(self.rows), "available_labels": self.bin_sizes.tolist(),
                "source_fractions": source.tolist(), "expected_fractions": expected.tolist(),
                "relative_label_weights": [float(p / q) if q else None for p, q in zip(expected, source)],
                "replacement": True, "split": "train",
                "candidate_sampling": "uniform segment-start rows with replacement",
                "validation_weighting": "original expanded rows; no resampling",
                "environment_steps_added": 0}
