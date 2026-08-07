"""
Experiment 07 — Contextual MAB learning convergence.

Trains LinUCB while periodically evaluating the greedy policy on a fixed
validation set (same contexts every checkpoint, no exploration bonus).

Outputs
-------
  results/07_mab_convergence/validation_curve.json
  results/07_mab_convergence/fig_mab_validation_convergence.{png,pdf}
"""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from config import (
    COMPRESSION_LEVELS,
    CONTEXT_DIM,
    DATA_ROOT,
    LINUCB_ALPHA,
    MODEL_DIR,
    NUM_SEMANTIC_CLUSTERS,
    NUM_SEMANTIC_SAMPLES,
    RANDOM_SEED,
    RESULT_DIR,
    TRAIN_EPISODES,
)
from learning.linucb import LinUCB
from runtime.episode import evaluate_action, load_concepts, probe_episode, recluster_concepts
from runtime.policy import select_adaptive_level
from runtime.sampling import make_episode_schedule
from superiority_plots import FIGSIZE, save, style

OUT_DIR = RESULT_DIR / "07_mab_convergence"
VALIDATION_SEED = 4242
DEFAULT_VAL_EPISODES = 36
DEFAULT_EVAL_INTERVAL = 50


def build_concept_bank(source, concept_sizes: list[int], seed: int) -> dict[int, Any]:
    return {
        k: (
            source
            if k == max(concept_sizes)
            else recluster_concepts(source, num_clusters=k, seed=seed + k)
        )
        for k in concept_sizes
    }


def build_validation_cases(
    concept_bank: dict[int, Any],
    concept_sizes: list[int],
    window_factors: list[float],
    num_val_episodes: int,
    seed: int,
) -> list[tuple[dict[str, Any], Any]]:
    """Fixed validation episodes reused at every training checkpoint."""
    schedule = make_episode_schedule(
        num_val_episodes, seed=seed, balanced=True, poor_weight=1
    )
    cases: list[tuple[dict[str, Any], Any]] = []
    for i, ep in enumerate(schedule):
        k = concept_sizes[i % len(concept_sizes)]
        factor = window_factors[i % len(window_factors)]
        ep = copy.deepcopy(ep)
        ep["num_concepts"] = k
        ep["service_window"] = max(4, min(64, int(round(k * factor))))
        cases.append((ep, concept_bank[k]))
    return cases


def evaluate_validation(
    agent: LinUCB,
    cases: list[tuple[dict[str, Any], Any]],
) -> dict[str, float]:
    """Greedy policy only (no exploration term, no parameter updates)."""
    rewards: list[float] = []
    for ep, concepts in cases:
        context, stress, _ = probe_episode(ep)
        level = select_adaptive_level(
            agent,
            context,
            ep,
            num_concepts=concepts.num_concepts,
        )
        outcome = evaluate_action(
            ep,
            concepts,
            level,
            context=context,
            network_stress=stress,
            transmit_repeats=2,
        )
        rewards.append(float(outcome.reward))

    arr = np.asarray(rewards, dtype=float)
    return {
        "mean_fsq": float(np.mean(arr)),
        "std_fsq": float(np.std(arr)),
        "n_val": int(len(arr)),
    }


def plot_convergence(curve: list[dict[str, Any]], out_dir: Path):
    style()
    episodes = [row["episode"] for row in curve]
    mean = np.asarray([row["mean_fsq"] for row in curve], dtype=float)
    std = np.asarray([row["std_fsq"] for row in curve], dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(episodes, mean, "-o", color="#1f77b4", linewidth=2.2, markersize=5, label="Validation FSQ")
    ax.fill_between(
        episodes,
        mean - std,
        mean + std,
        color="#1f77b4",
        alpha=0.18,
        label="±1 std. dev.",
    )
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Average validation FSQ")
    ax.set_xlim(left=0)
    ymax = min(1.05, float(np.max(mean + std)) * 1.12 + 1e-6)
    ax.set_ylim(0, ymax)
    ax.legend(loc="lower right", fontsize=9)
    save(fig, out_dir / "fig_mab_validation_convergence")


def train_with_validation(
    num_episodes: int = TRAIN_EPISODES,
    seed: int = RANDOM_SEED,
    alpha: float = LINUCB_ALPHA,
    eval_interval: int = DEFAULT_EVAL_INTERVAL,
    num_val_episodes: int = DEFAULT_VAL_EPISODES,
    online: bool = False,
    balanced: bool = True,
) -> dict[str, Any]:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mode = "online" if online else "full_information"
    concept_sizes = [NUM_SEMANTIC_CLUSTERS, 16, 22, 32, 40]
    window_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    print("Building shared embeddings and concept bank...")
    source = load_concepts(
        num_samples=NUM_SEMANTIC_SAMPLES,
        num_clusters=40,
        seed=seed,
        data_root=str(DATA_ROOT),
    )
    concept_bank = build_concept_bank(source, concept_sizes, seed)

    val_cases = build_validation_cases(
        concept_bank,
        concept_sizes,
        window_factors,
        num_val_episodes,
        seed=VALIDATION_SEED,
    )
    print(f"Fixed validation set: {len(val_cases)} episodes (seed={VALIDATION_SEED})")

    agent = LinUCB(arms=COMPRESSION_LEVELS, context_dim=CONTEXT_DIM, alpha=alpha)
    schedule = make_episode_schedule(
        num_episodes, seed=seed + 11, balanced=balanced, poor_weight=1
    )
    rng = random.Random(seed + 37)

    curve: list[dict[str, Any]] = []
    checkpoints = {0}
    for ep_num in range(eval_interval, num_episodes + 1, eval_interval):
        checkpoints.add(ep_num)

    def maybe_validate(episode_num: int):
        if episode_num not in checkpoints:
            return
        stats = evaluate_validation(agent, val_cases)
        row = {"episode": episode_num, **stats}
        curve.append(row)
        print(
            f"  [val @ ep {episode_num:4d}]  "
            f"mean FSQ={stats['mean_fsq']:.4f}  std={stats['std_fsq']:.4f}"
        )

    maybe_validate(0)

    for ep in schedule:
        k = rng.choice(concept_sizes)
        concepts = concept_bank[k]
        factor = rng.choice(window_factors)
        ep["num_concepts"] = k
        ep["service_window"] = max(4, min(64, int(round(k * factor))))
        context, stress, _ = probe_episode(ep)

        if online:
            arm_idx, level = agent.select_arm(context)
            outcome = evaluate_action(
                ep,
                concepts,
                level,
                context=context,
                network_stress=stress,
                transmit_repeats=2,
            )
            agent.update(arm_idx, context, outcome.reward)
        else:
            for arm_idx, level in enumerate(COMPRESSION_LEVELS):
                outcome = evaluate_action(
                    ep,
                    concepts,
                    level,
                    context=context,
                    network_stress=stress,
                    transmit_repeats=2,
                )
                agent.update(arm_idx, context, outcome.reward)

        if ep["episode"] % 100 == 0:
            print(f"training ep {ep['episode']:4d}/{num_episodes}")
        maybe_validate(int(ep["episode"]))

    model_path = MODEL_DIR / "linucb_policy_convergence.npz"
    agent.save(model_path)

    summary = {
        "num_episodes": num_episodes,
        "training_mode": mode,
        "alpha": alpha,
        "eval_interval": eval_interval,
        "num_val_episodes": num_val_episodes,
        "validation_seed": VALIDATION_SEED,
        "concept_sizes": concept_sizes,
        "service_window_factors": window_factors,
        "model_path": str(model_path),
        "validation_curve": curve,
    }

    json_path = out_dir / "validation_curve.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {json_path}")

    plot_convergence(curve, out_dir)
    return summary


def replot_from_json(json_path: Path | None = None, out_dir: Path | None = None):
    json_path = json_path or (OUT_DIR / "validation_curve.json")
    out_dir = out_dir or OUT_DIR
    summary = json.loads(json_path.read_text())
    plot_convergence(summary["validation_curve"], out_dir)
    print(f"Replot complete: {out_dir}")


def main():
    import argparse

    p = argparse.ArgumentParser(description="LinUCB validation convergence curve")
    p.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--alpha", type=float, default=LINUCB_ALPHA)
    p.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    p.add_argument("--val-episodes", type=int, default=DEFAULT_VAL_EPISODES)
    p.add_argument("--online", action="store_true")
    p.add_argument("--no-balanced", action="store_true")
    p.add_argument(
        "--replot",
        action="store_true",
        help="Regenerate figure from saved validation_curve.json",
    )
    args = p.parse_args()

    if args.replot:
        replot_from_json()
        return

    train_with_validation(
        num_episodes=args.episodes,
        seed=args.seed,
        alpha=args.alpha,
        eval_interval=args.eval_interval,
        num_val_episodes=args.val_episodes,
        online=args.online,
        balanced=not args.no_balanced,
    )


if __name__ == "__main__":
    main()
