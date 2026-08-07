"""Unit tests for compressor and reward (no SeQUeNCe)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learning.reward import build_context, compute_reward
from semantic.compressor import SemanticCompressor


def test_compression_levels_change_num_states():
    rng = np.random.default_rng(0)
    k = 10
    centroids = rng.normal(size=(k, 8))
    probs = np.ones(k) / k
    compressor = SemanticCompressor(k)

    states = [compressor.target_states(c) for c in [1.0, 0.8, 0.6, 0.4, 0.2]]
    assert states == [10, 8, 6, 4, 2]


def test_fs_is_geometric_not_granularity():
    rng = np.random.default_rng(1)
    k = 10
    probs = np.ones(k) / k

    centroids = rng.normal(size=(k, 16))
    comp = SemanticCompressor(k)
    full = comp.compress(centroids, probs, 1.0)
    small = comp.compress(centroids, probs, 0.2)

    assert full.num_states == 10
    assert small.num_states == 2
    assert full.information_fidelity > 0.999
    assert small.information_fidelity < 0.55
    # Geometric Fs is used for reward; it should drop under aggressive merge
    # of well-separated random centroids.
    assert full.semantic_fidelity > small.semantic_fidelity


def test_reward_harmonic_mean():
    assert abs(compute_reward(1.0, 1.0) - 1.0) < 1e-9
    assert abs(compute_reward(1.0, 0.0) - 0.0) < 1e-9
    r = compute_reward(0.8, 0.4)
    assert 0.5 < r < 0.6


def test_context_dim():
    ctx = build_context(
        probe_success_rate=0.5,
        probe_average_fidelity=0.9,
        normalized_distance=0.4,
        normalized_probe_delay=0.2,
        offered_load=0.3,
    )
    assert len(ctx) == 9
    assert ctx[0] == 1.0
    assert abs(ctx[6] - 0.3 * 0.5) < 1e-9
    assert abs(ctx[8] - 0.7 * 0.5) < 1e-9


if __name__ == "__main__":
    test_compression_levels_change_num_states()
    test_fs_is_geometric_not_granularity()
    test_reward_harmonic_mean()
    test_context_dim()
    print("All unit tests passed.")
