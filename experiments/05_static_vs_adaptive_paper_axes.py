"""
Experiment 05 — Paper axes, our comparison: Static QSC vs Adaptive QSC.

Same axes as Chehimi et al.:
  Fig. 2 style:  X = |X|,  Y = quantum communication resources
  Fig. 3 style:  X = quantum communication resources,  Y = quantum semantic fidelity

Curves compare what WE proposed:
  - Static QSC (fixed compression, paper-style static setting)
  - Adaptive QSC (LinUCB chooses compression from network state)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
from runtime.sampling import make_episode_schedule
from runtime.tiers import TIER_POOR, TIER_GOOD, tier_label
from semantic.compressor import SemanticCompressor
from semantic.concept_builder import ConceptSet


OUT_DIR = RESULT_DIR / "static_vs_adaptive_paper_axes"
DATASET_SIZES = [50, 100, 200, 400, 700, 1000, 1500, 2000]
STATIC_LEVEL = 1.0  # paper-style static: keep full concept set (|S|=K)
STATIC_COMPROMISE = 0.6  # best overall fixed from our experiments


def paper_style_rc():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
        }
    )


def choose_num_clusters(num_samples: int) -> int:
    return int(np.clip(np.ceil(2.2 * np.sqrt(num_samples)), 5, 40))


def resources_for_level(num_concepts: int, compression_level: float) -> int:
    return max(1, int(np.ceil(num_concepts * compression_level)))


def build_prefix_concepts(embeddings: np.ndarray, num_samples: int, seed: int = 42):
    from sklearn.cluster import KMeans

    x = embeddings[:num_samples]
    k = min(choose_num_clusters(num_samples), num_samples)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    assignments = kmeans.fit_predict(x)
    centroids = kmeans.cluster_centers_
    counts = np.bincount(assignments, minlength=k).astype(np.float64)
    probs = counts / max(counts.sum(), 1.0)
    return k, centroids, probs


def estimate_adaptive_resources_for_size(
    agent: LinUCB,
    concepts_full,
    num_samples: int,
    episodes_per_size: int = 36,
    seed: int = 123,
) -> dict:
    """
    For a given |X|, rebuild K concepts and estimate mean |S| used by
    static vs adaptive under a mixed network schedule.
    """
    k, centroids, probs = build_prefix_concepts(
        concepts_full.embeddings, num_samples, seed=seed
    )
    concepts = ConceptSet(
        embeddings=concepts_full.embeddings[:num_samples],
        labels=concepts_full.labels[:num_samples],
        assignments=np.zeros(num_samples, dtype=np.int64),
        centroids=centroids.astype(np.float64),
        probabilities=probs.astype(np.float64),
    )

    schedule = make_episode_schedule(episodes_per_size, seed=seed + num_samples)
    static_states = []
    adaptive_states = []
    static_fsq = []
    adaptive_fsq = []

    for ep in schedule:
        ep = dict(ep)
        ep["num_concepts"] = concepts.num_concepts
        ep["service_window"] = max(
            4,
            min(64, int(round(1.2 * concepts.num_concepts))),
        )
        context, stress, _ = probe_episode(ep)

        # Static paper-style
        out_s = evaluate_action(
            ep,
            concepts,
            STATIC_LEVEL,
            context=context,
            network_stress=stress,
            transmit_repeats=1,
        )
        static_states.append(out_s.num_states)
        static_fsq.append(out_s.reward)

        # Adaptive
        level = select_adaptive_level(
            agent,
            context,
            ep,
            num_concepts=concepts.num_concepts,
        )
        # Map arm to this K (arms are compression fractions)
        out_a = evaluate_action(
            ep,
            concepts,
            float(level),
            context=context,
            network_stress=stress,
            transmit_repeats=1,
        )
        adaptive_states.append(out_a.num_states)
        adaptive_fsq.append(out_a.reward)

    return {
        "num_samples": num_samples,
        "num_concepts": k,
        "static_resources": float(np.mean(static_states)),
        "adaptive_resources": float(np.mean(adaptive_states)),
        "static_compromise_resources": float(
            resources_for_level(k, STATIC_COMPROMISE)
        ),
        "static_fsq": float(np.mean(static_fsq)),
        "adaptive_fsq": float(np.mean(adaptive_fsq)),
        "static_full_resources": float(resources_for_level(k, STATIC_LEVEL)),
    }


def plot_fig2(rows: list[dict], out_dir: Path):
    """Same axes as paper Fig. 2: resources vs |X|, Static vs Adaptive."""
    xs = [r["num_samples"] for r in rows]
    static_y = [r["static_resources"] for r in rows]
    adaptive_y = [r["adaptive_resources"] for r in rows]
    compromise_y = [r["static_compromise_resources"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.plot(
        xs,
        static_y,
        "r--",
        linewidth=2.2,
        label=f"Static QSC (fixed c={STATIC_LEVEL:g}, paper-style)",
    )
    ax.plot(
        xs,
        compromise_y,
        color="0.45",
        linestyle=":",
        linewidth=2.0,
        label=f"Static QSC (fixed c={STATIC_COMPROMISE:g})",
    )
    ax.plot(
        xs,
        adaptive_y,
        "b-",
        linewidth=2.4,
        marker="o",
        markersize=4,
        label="Adaptive QSC (LinUCB, proposed)",
    )
    ax.set_xlabel(r"$|X|$ (dataset size)")
    ax.set_ylabel("Quantum communication resources")
    ax.set_title("Communication resources: static vs adaptive QSC")
    ax.legend(loc="best", framealpha=0.95)
    ax.set_xlim(min(xs), max(xs))
    ymax = max(max(static_y), max(adaptive_y), max(compromise_y))
    ax.set_ylim(0, ymax * 1.25)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"fig2_resources_vs_X_static_vs_adaptive.{ext}"
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def plot_fig3(curve: dict, out_dir: Path):
    """
    Same style as Chehimi Fig. 3:
      X = quantum communication resources
      Y = quantum semantic fidelity
      solid/dashed = good/poor
      diamonds on Adaptive curves (like paper's proposed framework)
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.7))

    # Static = paper-style fixed c=1.0 (flat once |S|=K is affordable)
    ax.plot(
        curve["static"][TIER_GOOD]["resources"],
        curve["static"][TIER_GOOD]["fidelity"],
        color="#d62728",
        linestyle="-",
        linewidth=2.2,
        label=f"Static QSC (fixed c={STATIC_LEVEL:g}), {tier_label(TIER_GOOD)}",
    )
    ax.plot(
        curve["static"][TIER_POOR]["resources"],
        curve["static"][TIER_POOR]["fidelity"],
        color="#d62728",
        linestyle="--",
        linewidth=2.2,
        label=f"Static QSC (fixed c={STATIC_LEVEL:g}), {tier_label(TIER_POOR)}",
    )

    # Adaptive = curves WITH diamond markers (like paper's blue proposed lines)
    ax.plot(
        curve["adaptive"][TIER_GOOD]["resources"],
        curve["adaptive"][TIER_GOOD]["fidelity"],
        color="#1f77b4",
        linestyle="-",
        linewidth=2.2,
        marker="D",
        markersize=7,
        markerfacecolor="#1f77b4",
        markeredgecolor="#1f77b4",
        label=f"Adaptive QSC, {tier_label(TIER_GOOD)}",
    )
    ax.plot(
        curve["adaptive"][TIER_POOR]["resources"],
        curve["adaptive"][TIER_POOR]["fidelity"],
        color="#1f77b4",
        linestyle="--",
        linewidth=2.2,
        marker="D",
        markersize=7,
        markerfacecolor="#1f77b4",
        markeredgecolor="#1f77b4",
        label=f"Adaptive QSC, {tier_label(TIER_POOR)}",
    )

    ax.set_xlabel("Quantum Communication Resources")
    ax.set_ylabel("Quantum semantic fidelity")
    ax.set_title(
        r"Fig. 3 style: Semantic fidelity vs quantum communication resources"
        "\n"
        r"(static vs adaptive QSC, fixed $|X|$)"
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=8)

    for ext in ("png", "pdf"):
        path = out_dir / f"fig3_fidelity_vs_resources_static_vs_adaptive.{ext}"
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def bin_xy(resources: list[float], fidelity: list[float]) -> tuple[list[float], list[float]]:
    """Average fidelity for each distinct resource value; sort by resources."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for r, f in zip(resources, fidelity):
        buckets[int(round(r))].append(f)
    xs = sorted(buckets)
    ys = [float(np.mean(buckets[x])) for x in xs]
    return xs, ys


def smooth_curve(xs: list[float], ys: list[float], n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Dense interpolation so static lines look continuous like the paper."""
    if len(xs) < 2:
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    # drop duplicate x
    uniq_x, idx = np.unique(x, return_index=True)
    uniq_y = y[idx]
    if len(uniq_x) < 2:
        return uniq_x, uniq_y
    xi = np.linspace(uniq_x.min(), uniq_x.max(), n)
    yi = np.interp(xi, uniq_x, uniq_y)
    return xi, yi


def levels_for_resource_sweep(num_concepts: int) -> list[float]:
    """Finer compression grid so |S| steps look curve-like (paper style)."""
    levels = []
    for s in range(1, num_concepts + 1):
        levels.append(min(1.0, s / float(num_concepts)))
    # unique, sorted
    return sorted(set(round(lv, 6) for lv in levels))


def main(
    episodes_fig2: int = 30,
    episodes_fig3: int = 80,
    skip_fig2: bool = False,
):
    paper_style_rc()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "linucb_policy.npz"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run experiments/02_train_linucb.py first."
        )
    agent = LinUCB.load(model_path)

    print("Loading semantic concepts...")
    concepts = load_concepts(
        num_samples=max(NUM_SEMANTIC_SAMPLES, max(DATASET_SIZES)),
        num_clusters=NUM_SEMANTIC_CLUSTERS,
        seed=RANDOM_SEED,
        data_root=str(DATA_ROOT),
    )

    fig2_rows: list[dict] = []
    if not skip_fig2:
        print("\n=== Fig. 2: resources vs |X| (static vs adaptive) ===")
        for n in DATASET_SIZES:
            print(f"  |X|={n} ...")
            row = estimate_adaptive_resources_for_size(
                agent,
                concepts,
                n,
                episodes_per_size=episodes_fig2,
                seed=1000 + n,
            )
            fig2_rows.append(row)
            print(
                f"    static|S|={row['static_resources']:.1f}  "
                f"adaptive|S|={row['adaptive_resources']:.1f}  "
                f"K={row['num_concepts']}"
            )
        plot_fig2(fig2_rows, OUT_DIR)
    else:
        print("Skipping Fig. 2 (--skip-fig2)")

    # ---------- Fig. 3 (paper-style curves) ----------
    print("\n=== Fig. 3: fidelity vs resources (static vs adaptive curves) ===")
    schedule = make_episode_schedule(episodes_fig3, seed=2026)
    schedule = [ep for ep in schedule if ep.get("tier") in (TIER_GOOD, TIER_POOR)]
    while sum(1 for e in schedule if e["tier"] == TIER_GOOD) < 25 or sum(
        1 for e in schedule if e["tier"] == TIER_POOR
    ) < 25:
        extra = make_episode_schedule(episodes_fig3, seed=len(schedule) + 99)
        schedule.extend([e for e in extra if e.get("tier") in (TIER_GOOD, TIER_POOR)])
        if len(schedule) > episodes_fig3 * 3:
            break

    # Precompute |S| for each level
    sweep_levels = levels_for_resource_sweep(concepts.num_concepts)
    compressor = SemanticCompressor(concepts.num_concepts)
    level_to_states = {
        lv: compressor.target_states(lv) for lv in sweep_levels
    }
    # Map trained arms to |S|
    arm_levels = [float(a) for a in agent.arms]
    arm_states = {lv: compressor.target_states(lv) for lv in arm_levels}

    static_data = {
        TIER_GOOD: {"resources": [], "fidelity": []},
        TIER_POOR: {"resources": [], "fidelity": []},
    }
    adaptive_data = {
        TIER_GOOD: {"resources": [], "fidelity": []},
        TIER_POOR: {"resources": [], "fidelity": []},
    }

    for i, ep in enumerate(schedule):
        tier = ep["tier"]
        context, stress, _ = probe_episode(ep)
        values = agent.get_estimated_values(context)

        # Cache outcomes for each compression level used this episode
        cache: dict[float, Any] = {}

        def eval_level(level: float):
            key = float(level)
            if key not in cache:
                cache[key] = evaluate_action(
                    ep,
                    concepts,
                    key,
                    context=context,
                    network_stress=stress,
                    transmit_repeats=1,
                )
            return cache[key]

        # Static paper baseline: fixed c=1.0 (cannot use fewer than |S|=K resources).
        out_static = eval_level(STATIC_LEVEL)
        full_states = int(out_static.num_states)
        full_fidelity = float(out_static.reward)

        # Adaptive curve: at each resource budget R, pick best LinUCB arm
        # among arms with |S| <= R (paper-style resource sweep).
        budgets = sorted(set(level_to_states.values()))
        for budget in budgets:
            if budget >= full_states:
                static_data[tier]["resources"].append(budget)
                static_data[tier]["fidelity"].append(full_fidelity)

            eligible = [
                (idx, lv)
                for idx, lv in enumerate(arm_levels)
                if arm_states[lv] <= budget
            ]
            if not eligible:
                idx = int(np.argmin([arm_states[lv] for lv in arm_levels]))
                level = arm_levels[idx]
            else:
                _, level = max(eligible, key=lambda t: values[t[0]])
            out = eval_level(level)
            adaptive_data[tier]["resources"].append(budget)
            adaptive_data[tier]["fidelity"].append(out.reward)

        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(schedule)} episodes")

    curve = {"static": {}, "adaptive": {}}
    for tier in (TIER_GOOD, TIER_POOR):
        sx, sy = bin_xy(static_data[tier]["resources"], static_data[tier]["fidelity"])
        ax_r, ay = bin_xy(
            adaptive_data[tier]["resources"], adaptive_data[tier]["fidelity"]
        )
        # Smooth static for continuous look; keep adaptive markers on discrete budgets
        sxi, syi = smooth_curve(sx, sy, n=100)
        curve["static"][tier] = {
            "resources": sxi.tolist(),
            "fidelity": syi.tolist(),
            "raw_resources": sx,
            "raw_fidelity": sy,
        }
        curve["adaptive"][tier] = {"resources": ax_r, "fidelity": ay}
        print(
            f"  {tier}: static |S|={sx}  adaptive budgets={ax_r}  "
            f"adaptive F range=[{min(ay):.3f},{max(ay):.3f}]"
        )

    plot_fig3(curve, OUT_DIR)

    summary = {
        "fig2": fig2_rows,
        "fig3": curve,
        "static_level": STATIC_LEVEL,
        "static_compromise_level": STATIC_COMPROMISE,
        "model_path": str(model_path),
        "note": (
            "Paper-style Fig. 3: Static = fixed c=1.0 (paper baseline); "
            "Adaptive = LinUCB best arm with |S|<=budget."
        ),
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
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    path = OUT_DIR / "summary.json"
    path.write_text(json.dumps(_to_py(summary), indent=2))
    print(f"Wrote {path}")
    print(f"\nSaved figures in: {OUT_DIR}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episodes-fig2", type=int, default=30)
    p.add_argument("--episodes-fig3", type=int, default=80)
    p.add_argument(
        "--skip-fig2",
        action="store_true",
        help="Only regenerate Fig. 3",
    )
    args = p.parse_args()
    main(
        episodes_fig2=args.episodes_fig2,
        episodes_fig3=args.episodes_fig3,
        skip_fig2=args.skip_fig2,
    )
