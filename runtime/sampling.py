"""Physical network condition sampling for dynamic episodes."""

from __future__ import annotations

import math
import random
from typing import Any

from runtime.tiers import TIER_GOOD, TIER_MODERATE, TIER_ORDER, TIER_POOR


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    if lo <= 0 or hi <= 0 or lo > hi:
        raise ValueError("invalid log-uniform bounds")
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def sample_physical_condition(rng: random.Random) -> dict[str, float | int | str]:
    """
    Stratified multi-factor network conditions for adaptive QSC.

    Each condition jointly varies fidelity, entanglement availability,
    decoherence, and traffic — matching the dynamic limitations of static QSC.

    Target oracle policy (SERVICE_WINDOW=12, K=10, background-first):
      good     → leftover ≈ 12  → c ≈ 1.0
      moderate → leftover ≈ 5–6 → c ≈ 0.4–0.6
      poor     → leftover ≈ 1–2 → c ≈ 0.2
    """
    tier = rng.choice(list(TIER_ORDER))

    if tier == TIER_GOOD:
        # Favorable physics so Fc (and thus FSQ) can reach ~0.7–0.8.
        condition = {
            "distance": rng.randint(250, 500),
            "attenuation": rng.uniform(0.5e-4, 0.9e-4),
            "memory_fidelity": rng.uniform(0.985, 0.999),
            "memory_efficiency": rng.uniform(0.98, 1.00),
            "memory_coherence_time": log_uniform(rng, 8.0, 20.0),
            "detector_efficiency": rng.uniform(0.98, 1.00),
            "offered_load": rng.uniform(0.0, 0.05),
        }
    elif tier == TIER_MODERATE:
        condition = {
            "distance": rng.randint(700, 1100),
            "attenuation": rng.uniform(1.2e-4, 1.8e-4),
            "memory_fidelity": rng.uniform(0.94, 0.98),
            "memory_efficiency": rng.uniform(0.92, 0.97),
            "memory_coherence_time": log_uniform(rng, 2.0, 5.0),
            "detector_efficiency": rng.uniform(0.92, 0.97),
            "offered_load": rng.uniform(0.38, 0.52),
        }
    else:  # poor — keep hard so Adaptive must compress
        condition = {
            "distance": rng.randint(1200, 1800),
            "attenuation": rng.uniform(2.2e-4, 3.0e-4),
            "memory_fidelity": rng.uniform(0.88, 0.94),
            "memory_efficiency": rng.uniform(0.82, 0.92),
            "memory_coherence_time": log_uniform(rng, 0.4, 1.5),
            "detector_efficiency": rng.uniform(0.82, 0.92),
            "offered_load": rng.uniform(0.78, 0.92),
        }

    condition["tier"] = tier
    return condition


def make_episode_schedule(
    num_episodes: int,
    seed: int,
    *,
    balanced: bool = False,
    poor_weight: int = 2,
    adverse_weight: int | None = None,
    congested_weight: int | None = None,
    hard_weight: int | None = None,
) -> list[dict[str, Any]]:
    """
    Build an episode schedule.

    ``poor_weight`` oversamples poor network conditions during training.
    Older weight argument names are kept as aliases.
    """
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive")
    for alias in (hard_weight, congested_weight, adverse_weight):
        if alias is not None:
            poor_weight = alias

    rng = random.Random(seed)
    episodes = []
    if balanced:
        tier_cycle = [TIER_GOOD, TIER_MODERATE] + [TIER_POOR] * max(1, int(poor_weight))
    else:
        tier_cycle = None

    for i in range(num_episodes):
        if balanced:
            tier = tier_cycle[i % len(tier_cycle)]
            for _ in range(50):
                cond = sample_physical_condition(rng)
                if cond["tier"] == tier:
                    break
            else:
                cond = sample_physical_condition(rng)
                cond["tier"] = tier
        else:
            cond = sample_physical_condition(rng)
        episodes.append(
            {
                "episode": i + 1,
                **cond,
                "probe_seed_offset": i * 100,
                "transmit_seed_offset": 1_000_000 + i * 100,
            }
        )
    return episodes
