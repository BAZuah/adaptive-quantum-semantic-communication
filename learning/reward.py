"""Reward and LinUCB context helpers."""

from __future__ import annotations


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def semantic_quantum_fidelity(
    semantic_fidelity: float,
    communication_fidelity: float,
    eps: float = 1e-12,
) -> float:
    """
    F_SQ = harmonic mean of Fs and Fc.

    High reward requires BOTH semantic preservation and successful
    quantum delivery.
    """
    fs = clamp(float(semantic_fidelity))
    fc = clamp(float(communication_fidelity))
    return 2.0 * fs * fc / (fs + fc + eps)


def compute_reward(semantic_fidelity: float, communication_fidelity: float) -> float:
    return semantic_quantum_fidelity(semantic_fidelity, communication_fidelity)


def normalize_delay(delay_seconds: float, reference_seconds: float = 1.0) -> float:
    d = max(0.0, float(delay_seconds))
    if reference_seconds <= 0:
        raise ValueError("reference_seconds must be positive")
    return d / (d + reference_seconds)


def build_context(
    *,
    probe_success_rate: float,
    probe_average_fidelity: float,
    normalized_distance: float,
    normalized_probe_delay: float,
    offered_load: float,
    num_concepts: int,
    service_window: int,
    max_concepts: int = 40,
) -> list[float]:
    """
    Fourteen-dimensional LinUCB context.

    The last three terms make the policy valid across dataset sizes and
    entanglement budgets:
      demand_n       = K / K_max
      capacity_n     = leftover semantic slots / K_max
      capacity_ratio = leftover slots / K
    """
    success = clamp(probe_success_rate)
    fidelity = clamp(probe_average_fidelity)
    distance = clamp(normalized_distance)
    delay = clamp(normalized_probe_delay)
    load = clamp(offered_load)
    k = max(1, int(num_concepts))
    window = max(1, int(service_window))
    semantic_slots = max(0, window - int(round(load * window)))
    demand_n = clamp(k / max(1.0, float(max_concepts)))
    capacity_n = clamp(semantic_slots / max(1.0, float(max_concepts)))
    capacity_ratio = clamp(semantic_slots / float(k))
    network_quality = (1.0 - load) * success * fidelity

    ctx = [
        1.0,
        success,
        fidelity,
        distance,
        delay,
        load,
        demand_n,
        capacity_n,
        capacity_ratio,
        capacity_ratio**2,
        network_quality,
        network_quality * capacity_ratio,
        load * (1.0 - capacity_ratio),
        load**2,
    ]
    if len(ctx) != 14:
        raise ValueError(f"expected 14-D context, got {len(ctx)}")
    return ctx


def observed_stress(
    *,
    probe_success_rate: float,
    probe_average_fidelity: float,
    normalized_probe_delay: float,
    offered_load: float,
) -> float:
    """Reporting-only stress score for analysis plots (not fed to LinUCB)."""
    fc_proxy = clamp(probe_success_rate) * clamp(probe_average_fidelity)
    return clamp(
        0.35 * (1.0 - fc_proxy)
        + 0.25 * (1.0 - clamp(probe_success_rate))
        + 0.15 * clamp(normalized_probe_delay)
        + 0.25 * clamp(offered_load)
    )
