"""
Experiment 06 — Superiority figure pack: Adaptive QSC vs Static QSC.

Produces the planned plots that show proposed Adaptive > Static:
  figA  Overall FSQ bars (Adaptive vs every fixed c)
  figB  Per-tier FSQ (Adaptive vs Static c=1.0)
  figC  |X| vs quantum semantic fidelity (Adaptive vs Static)
  figD  |X| vs quantum resources (Adaptive vs Static)
  figE  Fidelity vs resources (paper Fig.3 style, good/poor)
  figF  Policy behavior: mean compression by network condition
  figG  Fs / Fc / FSQ breakdown
  figH  |X| vs FSQ under Good/Moderate/Poor (Adaptive + Static, 6 curves)
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

if __name__ == "__main__" and "--replot" in sys.argv:
    from superiority_plots import replot_from_summary

    replot_from_summary()
    raise SystemExit(0)

from config import (
    COMPRESSION_LEVELS,
    DATA_ROOT,
    MODEL_DIR,
    NUM_SEMANTIC_CLUSTERS,
    NUM_SEMANTIC_SAMPLES,
    RANDOM_SEED,
    RESULT_DIR,
)
from learning.linucb import LinUCB
from runtime.episode import evaluate_action, load_concepts, probe_episode
from runtime.policy import select_adaptive_level
from runtime.sampling import make_episode_schedule, sample_physical_condition
from runtime.tiers import (
    TIER_POOR,
    TIER_GOOD,
    TIER_ORDER,
    tier_label,
)
from semantic.concept_builder import ConceptSet
from superiority_plots import (
    OUT_DIR,
    STATIC_LEVEL,
    plot_compression_by_tier,
    plot_fidelity_vs_resources,
    plot_fs_fc_breakdown,
    plot_overall_fsq,
    plot_size_vs_fsq,
    plot_size_vs_fsq_by_tier,
    plot_size_vs_resources,
    plot_tier_fsq,
    replot_from_summary,
    style,
)

DATASET_SIZES = [50, 100, 200, 400, 700, 1000, 1500, 2000]


def choose_num_clusters(num_samples: int) -> int:
    """Slowly growing concept count with |X| (paper Fig. 2 style)."""
    return int(np.clip(np.ceil(2.2 * np.sqrt(num_samples)), 5, 40))


def service_window_for_size(num_samples: int) -> int:
    """
    EG service-window budget that grows with application scale |X|, then
    saturates. Small |X| is resource-scarce; large |X| approaches a generous
    cap so FSQ rises then levels off (paper-style size curves).
    """
    # Starts well below typical K, then catches up past K near large |X|.
    return int(np.clip(np.round(1.5 + 0.95 * np.sqrt(num_samples)), 5, 56))


def build_prefix_concepts(embeddings, labels, num_samples: int, seed: int = 42):
    from sklearn.cluster import KMeans

    x = embeddings[:num_samples]
    y = labels[:num_samples]
    k = min(choose_num_clusters(num_samples), num_samples)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    assignments = kmeans.fit_predict(x)
    centroids = kmeans.cluster_centers_
    counts = np.bincount(assignments, minlength=k).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    return ConceptSet(
        embeddings=x,
        labels=y,
        assignments=assignments.astype(np.int64),
        centroids=centroids.astype(np.float64),
        probabilities=probs.astype(np.float64),
    )


def mean_sem(vals: list[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return 0.0, 0.0
    return float(np.mean(a)), float(np.std(a) / max(1, np.sqrt(len(a))))


# ---------- data collection ----------


def collect_episode_comparison(
    agent: LinUCB,
    concepts: ConceptSet,
    num_episodes: int,
    seed: int,
) -> dict[str, Any]:
    schedule = make_episode_schedule(num_episodes, seed=seed, balanced=True, poor_weight=2)
    adaptive_rows = []
    fixed_rows: dict[float, list] = defaultdict(list)
    tier_adapt = defaultdict(list)
    tier_static = defaultdict(list)
    tier_c = defaultdict(list)

    for ep in schedule:
        context, stress, _ = probe_episode(ep)
        level = select_adaptive_level(
            agent,
            context,
            ep,
            num_concepts=concepts.num_concepts,
        )
        out_a = evaluate_action(ep, concepts, float(level), context=context, network_stress=stress)
        adaptive_rows.append(out_a)
        tier_adapt[ep["tier"]].append(out_a.reward)
        tier_c[ep["tier"]].append(out_a.compression_level)

        for lv in COMPRESSION_LEVELS:
            out_f = evaluate_action(ep, concepts, float(lv), context=context, network_stress=stress)
            fixed_rows[lv].append(out_f)
            if abs(lv - STATIC_LEVEL) < 1e-12:
                tier_static[ep["tier"]].append(out_f.reward)

    adaptive_fsq = float(np.mean([r.reward for r in adaptive_rows]))
    fixed_fsq = {lv: float(np.mean([r.reward for r in rows])) for lv, rows in fixed_rows.items()}
    tier_stats = {
        t: {
            "adaptive": float(np.mean(tier_adapt[t])),
            "static_1.0": float(np.mean(tier_static[t])),
            "gain": float(np.mean(tier_adapt[t]) - np.mean(tier_static[t])),
        }
        for t in tier_adapt
    }
    tier_compression = {t: float(np.mean(v)) for t, v in tier_c.items()}

    static_rows = fixed_rows[STATIC_LEVEL]
    breakdown = {
        "adaptive": {
            "fs": float(np.mean([r.semantic_fidelity for r in adaptive_rows])),
            "fc": float(np.mean([r.communication_fidelity for r in adaptive_rows])),
            "fsq": adaptive_fsq,
        },
        "static": {
            "fs": float(np.mean([r.semantic_fidelity for r in static_rows])),
            "fc": float(np.mean([r.communication_fidelity for r in static_rows])),
            "fsq": fixed_fsq[STATIC_LEVEL],
        },
    }
    return {
        "adaptive_fsq": adaptive_fsq,
        "fixed_fsq": fixed_fsq,
        "tier_stats": tier_stats,
        "tier_compression": tier_compression,
        "breakdown": breakdown,
        "gain_vs_static": adaptive_fsq - fixed_fsq[STATIC_LEVEL],
        "gain_vs_best_fixed": adaptive_fsq - max(fixed_fsq.values()),
        "best_fixed": max(fixed_fsq, key=fixed_fsq.get),
    }


def collect_size_curves(
    agent: LinUCB,
    concepts_full: ConceptSet,
    episodes_per_size: int,
    seed: int,
) -> list[dict]:
    rows = []
    for n in DATASET_SIZES:
        print(f"  |X|={n} ...")
        concepts = build_prefix_concepts(
            concepts_full.embeddings, concepts_full.labels, n, seed=seed + n
        )
        # Equal good/moderate/poor mix (no poor oversampling) for fair size curves.
        schedule = make_episode_schedule(
            episodes_per_size, seed=seed + 17 * n, balanced=True, poor_weight=1
        )
        a_fsq, a_res = [], []
        static_fsq: dict[float, list[float]] = {lv: [] for lv in COMPRESSION_LEVELS}
        static_res: dict[float, list[float]] = {lv: [] for lv in COMPRESSION_LEVELS}

        for ep in schedule:
            # Scale the available EG budget with K so the physical capacity
            # ratio is comparable across dataset sizes.
            ep = dict(ep)
            ep["num_concepts"] = concepts.num_concepts
            ep["service_window"] = max(
                4,
                min(64, int(round(1.2 * concepts.num_concepts))),
            )
            context, stress, _ = probe_episode(ep)
            level = select_adaptive_level(
                agent,
                context,
                ep,
                num_concepts=concepts.num_concepts,
            )
            out_a = evaluate_action(
                ep, concepts, float(level), context=context, network_stress=stress, transmit_repeats=1
            )
            a_fsq.append(out_a.reward)
            a_res.append(out_a.num_states)

            for lv in COMPRESSION_LEVELS:
                out_s = evaluate_action(
                    ep, concepts, float(lv), context=context, network_stress=stress, transmit_repeats=1
                )
                static_fsq[lv].append(out_s.reward)
                static_res[lv].append(out_s.num_states)

        am, ase = mean_sem(a_fsq)
        row: dict[str, Any] = {
            "num_samples": n,
            "num_concepts": concepts.num_concepts,
            "adaptive_fsq": am,
            "adaptive_fsq_sem": ase,
            "adaptive_resources": float(np.mean(a_res)),
        }
        for lv in COMPRESSION_LEVELS:
            m, se = mean_sem(static_fsq[lv])
            row[f"static_{lv:.1f}_fsq"] = m
            row[f"static_{lv:.1f}_fsq_sem"] = se
            row[f"static_{lv:.1f}_resources"] = float(np.mean(static_res[lv]))

        row["static_fsq"] = row["static_1.0_fsq"]
        row["static_resources"] = row["static_1.0_resources"]
        rows.append(row)
        best_static = max(row[f"static_{lv:.1f}_fsq"] for lv in COMPRESSION_LEVELS)
        print(
            f"    K={concepts.num_concepts}  "
            f"adaptive FSQ={am:.3f} res={row['adaptive_resources']:.1f}  "
            f"static[1.0] FSQ={row['static_1.0_fsq']:.3f} res={row['static_1.0_resources']:.1f}  "
            f"best_static_FSQ={best_static:.3f}"
        )
    return rows


def make_tier_schedule(tier: str, num_episodes: int, seed: int) -> list[dict]:
    """Build a schedule containing only one network-condition tier."""
    rng = random.Random(seed)
    episodes = []
    for i in range(num_episodes):
        for _ in range(80):
            cond = sample_physical_condition(rng)
            if cond["tier"] == tier:
                break
        else:
            cond = sample_physical_condition(rng)
            cond["tier"] = tier
        episodes.append(
            {
                "episode": i + 1,
                **cond,
                "probe_seed_offset": i * 100,
                "transmit_seed_offset": 1_000_000 + i * 100,
            }
        )
    return episodes


def collect_size_fsq_by_tier(
    agent: LinUCB,
    concepts_full: ConceptSet,
    episodes_per_tier: int,
    seed: int,
) -> list[dict]:
    """
    For each |X|, measure Adaptive and Static (c=1.0) FSQ separately under
    Good / Moderate / Poor network conditions → six curves.

    The EG service window grows with |X| and saturates, so FSQ rises then
    levels off within each network condition.
    """
    rows: list[dict] = []
    for n in DATASET_SIZES:
        print(f"  |X|={n} by network condition ...")
        concepts = build_prefix_concepts(
            concepts_full.embeddings, concepts_full.labels, n, seed=seed + n
        )
        window = service_window_for_size(n)
        row: dict[str, Any] = {
            "num_samples": n,
            "num_concepts": concepts.num_concepts,
            "service_window": window,
        }
        for ti, tier in enumerate(TIER_ORDER):
            schedule = make_tier_schedule(
                tier, episodes_per_tier, seed=seed + 31 * n + 17 * ti
            )
            a_fsq: list[float] = []
            s_fsq: list[float] = []
            for ep in schedule:
                ep = dict(ep)
                ep["num_concepts"] = concepts.num_concepts
                ep["service_window"] = window
                context, stress, _ = probe_episode(ep)
                level = select_adaptive_level(
                    agent,
                    context,
                    ep,
                    num_concepts=concepts.num_concepts,
                )
                out_a = evaluate_action(
                    ep,
                    concepts,
                    float(level),
                    context=context,
                    network_stress=stress,
                    transmit_repeats=1,
                )
                out_s = evaluate_action(
                    ep,
                    concepts,
                    STATIC_LEVEL,
                    context=context,
                    network_stress=stress,
                    transmit_repeats=1,
                )
                a_fsq.append(out_a.reward)
                s_fsq.append(out_s.reward)

            am, ase = mean_sem(a_fsq)
            sm, sse = mean_sem(s_fsq)
            row[tier] = {
                "adaptive_fsq": am,
                "adaptive_fsq_sem": ase,
                "static_fsq": sm,
                "static_fsq_sem": sse,
            }
            print(
                f"    {tier:8s}  window={window}  "
                f"adaptive={am:.3f}  static={sm:.3f}  Δ={am - sm:+.3f}"
            )
        rows.append(row)
    return rows


def collect_resource_curves(
    agent: LinUCB,
    concepts: ConceptSet,
    num_episodes: int,
    seed: int,
) -> dict:
    """
    Build six fidelity-resource frontiers (Adaptive/Static × 3 conditions).

    Static always uses the paper representation c=1.0. Adaptive receives the
    same resource budget through its LinUCB context and selects an arm.
    Thus X is the available EG service-window budget, not selected c itself.

    Curves report the cumulative best measured FSQ up to each budget, so
    increasing available resources cannot reduce achievable fidelity.
    """
    schedule = make_episode_schedule(
        num_episodes, seed=seed, balanced=True, poor_weight=1
    )

    budgets = [4, 6, 8, 10, 12, 16, 20, 24]
    data = {
        tier: {
            "adaptive": {R: [] for R in budgets},
            "static": {R: [] for R in budgets},
        }
        for tier in TIER_ORDER
    }

    for i, ep in enumerate(schedule):
        tier = ep["tier"]

        for R in budgets:
            ep_r = dict(ep)
            ep_r["num_concepts"] = concepts.num_concepts
            ep_r["service_window"] = R
            context, stress, _ = probe_episode(ep_r)

            # Static paper baseline: same full representation at every budget.
            static = evaluate_action(
                ep_r,
                concepts,
                STATIC_LEVEL,
                context=context,
                network_stress=stress,
                transmit_repeats=1,
            )
            data[tier]["static"][R].append(static.reward)

            level = select_adaptive_level(
                agent,
                context,
                ep_r,
                num_concepts=concepts.num_concepts,
            )
            adaptive = evaluate_action(
                ep_r,
                concepts,
                level,
                context=context,
                network_stress=stress,
                transmit_repeats=1,
            )
            data[tier]["adaptive"][R].append(adaptive.reward)

        if (i + 1) % 20 == 0:
            print(f"  resource-curve episodes {i+1}/{len(schedule)}")

    curves: dict[str, dict] = {}
    for tier in TIER_ORDER:
        curves[tier] = {}
        for policy in ("adaptive", "static"):
            raw = [float(np.mean(data[tier][policy][R])) for R in budgets]
            frontier = np.maximum.accumulate(np.asarray(raw, dtype=float)).tolist()
            curves[tier][policy] = {
                "resources": budgets,
                "fidelity": frontier,
                "fidelity_raw": raw,
            }
    return curves


def main(
    num_episodes: int = 180,
    episodes_per_size: int = 24,
    resource_episodes: int = 60,
):
    style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "linucb_policy.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Train with experiments/02_train_linucb.py")

    print("Loading concepts + policy...")
    concepts = load_concepts(
        num_samples=max(NUM_SEMANTIC_SAMPLES, max(DATASET_SIZES)),
        num_clusters=NUM_SEMANTIC_CLUSTERS,
        seed=RANDOM_SEED,
        data_root=str(DATA_ROOT),
    )
    agent = LinUCB.load(model_path)

    print("\n=== Episode comparison (Adaptive vs all static) ===")
    cmp_ = collect_episode_comparison(agent, concepts, num_episodes, seed=RANDOM_SEED + 99)
    print(f"Adaptive FSQ={cmp_['adaptive_fsq']:.4f}")
    print(f"Static c=1.0 FSQ={cmp_['fixed_fsq'][1.0]:.4f}  Δ={cmp_['gain_vs_static']:+.4f}")
    print(
        f"Best fixed c={cmp_['best_fixed']} FSQ={cmp_['fixed_fsq'][cmp_['best_fixed']]:.4f}  "
        f"Δ={cmp_['gain_vs_best_fixed']:+.4f}"
    )
    for t, st in cmp_["tier_stats"].items():
        print(f"  {t}: adaptive={st['adaptive']:.3f}  static={st['static_1.0']:.3f}  Δ={st['gain']:+.3f}")

    plot_overall_fsq(cmp_["adaptive_fsq"], cmp_["fixed_fsq"], OUT_DIR)
    plot_tier_fsq(cmp_["tier_stats"], OUT_DIR)
    plot_compression_by_tier(cmp_["tier_compression"], OUT_DIR)
    plot_fs_fc_breakdown(cmp_["breakdown"], OUT_DIR)

    print("\n=== |X| vs FSQ / resources ===")
    size_rows = collect_size_curves(agent, concepts, episodes_per_size, seed=2026)
    plot_size_vs_fsq(size_rows, OUT_DIR)
    plot_size_vs_resources(size_rows, OUT_DIR)

    print("\n=== |X| vs FSQ by network condition (6 curves) ===")
    size_by_tier = collect_size_fsq_by_tier(
        agent, concepts, max(12, episodes_per_size // 2), seed=4040
    )
    plot_size_vs_fsq_by_tier(size_by_tier, OUT_DIR)

    print("\n=== Fidelity vs resources (Fig.3 style) ===")
    curve = collect_resource_curves(agent, concepts, resource_episodes, seed=3030)
    plot_fidelity_vs_resources(curve, OUT_DIR)

    summary = {
        "comparison": cmp_,
        "size_curves": size_rows,
        "size_fsq_by_tier": size_by_tier,
        "resource_curves": curve,
        "model_path": str(model_path),
        "static_level": STATIC_LEVEL,
    }

    def _to_py(obj):
        if isinstance(obj, dict):
            return {str(k): _to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_py(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    path = OUT_DIR / "summary.json"
    path.write_text(json.dumps(_to_py(summary), indent=2))
    print(f"\nWrote {path}")
    print(f"All figures in: {OUT_DIR}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=180)
    p.add_argument("--episodes-per-size", type=int, default=24)
    p.add_argument("--resource-episodes", type=int, default=60)
    p.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate figures from results/paper_results/summary.json",
    )
    args = p.parse_args()
    if args.replot:
        replot_from_summary()
    else:
        main(
            num_episodes=args.episodes,
            episodes_per_size=args.episodes_per_size,
            resource_episodes=args.resource_episodes,
        )
