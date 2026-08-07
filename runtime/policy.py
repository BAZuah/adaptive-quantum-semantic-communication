"""Shared deployment policy for adaptive semantic compression."""

from __future__ import annotations

from typing import Any

from config import SERVICE_WINDOW
from learning.linucb import LinUCB


def select_adaptive_level(
    agent: LinUCB,
    context: list[float],
    episode: dict[str, Any],
    *,
    num_concepts: int,
    service_window: int | None = None,
) -> float:
    """
    Select one compression level using the same policy in every experiment.

    In this simulator, if all K semantic states fit after background traffic,
    c=1.0 weakly dominates compression: it preserves maximum semantic fidelity
    without causing resource drops. Otherwise LinUCB handles the trade-off.
    """
    window = int(
        service_window
        if service_window is not None
        else episode.get("service_window", SERVICE_WINDOW)
    )
    load = max(0.0, min(1.0, float(episode["offered_load"])))
    semantic_slots = max(0, window - int(round(load * window)))

    full_level = next(
        (float(level) for level in agent.arms if float(level) == 1.0),
        None,
    )
    if full_level is not None and semantic_slots >= int(num_concepts):
        return full_level

    _, level = agent.select_greedy_arm(context)
    return float(level)
