"""Canonical network-condition names for adaptive QSC.

Tiers are multi-factor regimes (not traffic-only). Each combines:
  - channel / pair fidelity
  - entanglement availability
  - decoherence (memory coherence / efficiency)
  - traffic / offered load

Names used in figures and papers:
  Good Network Condition
  Moderate Network Condition
  Poor Network Condition
"""

from __future__ import annotations

TIER_GOOD = "good"
TIER_MODERATE = "moderate"
TIER_POOR = "poor"

TIER_ORDER = [TIER_GOOD, TIER_MODERATE, TIER_POOR]

TIER_LABELS = {
    TIER_GOOD: "Good Network Condition",
    TIER_MODERATE: "Moderate Network Condition",
    TIER_POOR: "Poor Network Condition",
}

TIER_LABELS_SHORT = {
    TIER_GOOD: "Good",
    TIER_MODERATE: "Moderate",
    TIER_POOR: "Poor",
}

# Backward-compatible aliases from earlier naming schemes
TIER_ALIASES = {
    "easy": TIER_GOOD,
    "medium": TIER_MODERATE,
    "hard": TIER_POOR,
    "favorable": TIER_GOOD,
    "constrained": TIER_MODERATE,
    "congested": TIER_POOR,
    "adverse": TIER_POOR,
    "good": TIER_GOOD,
    "moderate": TIER_MODERATE,
    "poor": TIER_POOR,
    TIER_GOOD: TIER_GOOD,
    TIER_MODERATE: TIER_MODERATE,
    TIER_POOR: TIER_POOR,
}


def normalize_tier(tier: str) -> str:
    key = str(tier).strip().lower()
    if key not in TIER_ALIASES:
        raise KeyError(f"Unknown network condition: {tier!r}")
    return TIER_ALIASES[key]


def tier_label(tier: str, *, short: bool = False) -> str:
    tid = normalize_tier(tier)
    return TIER_LABELS_SHORT[tid] if short else TIER_LABELS[tid]
