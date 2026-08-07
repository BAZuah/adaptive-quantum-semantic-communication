"""
Experiment 02 — Train LinUCB under dynamic network conditions.

Training mode (default): full-information / simulator-offline
    Probe once → evaluate EVERY compression arm on that episode →
    update LinUCB for every arm. This is the right sample-efficient
    protocol when the environment is a controllable simulator.

Optional --online:
    Classic online bandit (select one arm, update only that arm).
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
from runtime.episode import (
    evaluate_action,
    load_concepts,
    probe_episode,
    recluster_concepts,
)
from runtime.policy import select_adaptive_level
from runtime.sampling import make_episode_schedule


def main(
    num_episodes: int = TRAIN_EPISODES,
    seed: int = RANDOM_SEED,
    alpha: float = LINUCB_ALPHA,
    online: bool = False,
    balanced: bool = True,
):
    out_dir = RESULT_DIR / "02_train_linucb"
    out_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mode = "online" if online else "full_information"
    print(f"Training mode: {mode}")
    print("Building shared embeddings and concept bank...")
    source = load_concepts(
        num_samples=NUM_SEMANTIC_SAMPLES,
        num_clusters=40,
        seed=seed,
        data_root=str(DATA_ROOT),
    )
    concept_sizes = [NUM_SEMANTIC_CLUSTERS, 16, 22, 32, 40]
    concept_bank = {
        k: (
            source
            if k == 40
            else recluster_concepts(source, num_clusters=k, seed=seed + k)
        )
        for k in concept_sizes
    }

    agent = LinUCB(
        arms=COMPRESSION_LEVELS,
        context_dim=CONTEXT_DIM,
        alpha=alpha,
    )
    schedule = make_episode_schedule(
        num_episodes, seed=seed + 11, balanced=balanced, poor_weight=1
    )
    rng = random.Random(seed + 37)
    window_factors = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    rows = []
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
            chosen = [outcome]
        else:
            # Full-information: evaluate every arm with the SAME SeQUeNCe
            # seeds so arm differences come from |S| demand, not RNG noise.
            chosen = []
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
                chosen.append(outcome)

            # Log the greedy action under the current model for monitoring.
            greedy_level = select_adaptive_level(
                agent,
                context,
                ep,
                num_concepts=concepts.num_concepts,
            )
            outcome = next(o for o in chosen if o.compression_level == greedy_level)

        row = outcome.to_dict()
        row["arm_index"] = COMPRESSION_LEVELS.index(outcome.compression_level)
        row["tier"] = str(ep.get("tier", "?"))
        row["training_mode"] = mode
        row["context"] = json.dumps(row["context"])
        rows.append(row)

        if ep["episode"] % 25 == 0 or ep["episode"] == 1:
            recent = rows[-25:]
            print(
                f"ep {ep['episode']:4d}/{num_episodes}  "
                f"c={outcome.compression_level:.1f}  FSQ={outcome.reward:.3f}  "
                f"tier={ep.get('tier', '?'):6s}  stress={stress:.3f}  "
                f"recent_FSQ={mean(r['reward'] for r in recent):.3f}"
            )

    model_path = MODEL_DIR / "linucb_policy.npz"
    agent.save(model_path)

    csv_path = out_dir / "training_episodes.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    action_by_tier: dict[str, dict[str, int]] = {}
    for r in rows:
        tier = r["tier"]
        action_by_tier.setdefault(tier, {})
        key = str(r["compression_level"])
        action_by_tier[tier][key] = action_by_tier[tier].get(key, 0) + 1

    report = {
        "num_episodes": num_episodes,
        "training_mode": mode,
        "alpha": alpha,
        "mean_fsq": mean(r["reward"] for r in rows),
        "mean_fs": mean(r["semantic_fidelity"] for r in rows),
        "mean_fc": mean(r["communication_fidelity"] for r in rows),
        "mean_compression": mean(r["compression_level"] for r in rows),
        "action_counts": {
            str(lv): sum(1 for r in rows if r["compression_level"] == lv)
            for lv in COMPRESSION_LEVELS
        },
        "action_by_tier": action_by_tier,
        "concept_sizes": concept_sizes,
        "service_window_factors": window_factors,
        "model_path": str(model_path),
    }
    report_path = out_dir / "training_summary.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n=== LinUCB training done ===")
    print(f"Mode: {mode}")
    print(f"Mean logged FSQ: {report['mean_fsq']:.4f}")
    print(f"Mean compression: {report['mean_compression']:.3f}")
    print(f"Actions: {report['action_counts']}")
    print(f"By tier: {action_by_tier}")
    print(f"Saved model → {model_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--alpha", type=float, default=LINUCB_ALPHA)
    p.add_argument(
        "--online",
        action="store_true",
        help="Use classic online bandit updates (one arm per episode).",
    )
    p.add_argument(
        "--no-balanced",
        action="store_true",
        help="Use random tier mix instead of equal easy/medium/hard.",
    )
    args = p.parse_args()
    main(
        num_episodes=args.episodes,
        seed=args.seed,
        alpha=args.alpha,
        online=args.online,
        balanced=not args.no_balanced,
    )
