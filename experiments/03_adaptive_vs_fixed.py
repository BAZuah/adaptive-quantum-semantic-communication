"""
Experiment 03 — Adaptive (frozen LinUCB) vs fixed compression.

Shared probes: every policy sees the same network context per episode,
so the comparison is fair.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

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


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "mean_fsq": mean(r["reward"] for r in rows),
        "mean_fs": mean(r["semantic_fidelity"] for r in rows),
        "mean_fc": mean(r["communication_fidelity"] for r in rows),
        "mean_compression": mean(r["compression_level"] for r in rows),
        "mean_drop_ratio": mean(r["drop_ratio"] for r in rows),
        "mean_num_states": mean(r["num_states"] for r in rows),
    }


def main(
    num_episodes: int = 120,
    seed: int = RANDOM_SEED,
    model_path: Path | None = None,
):
    out_dir = RESULT_DIR / "03_adaptive_vs_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if model_path is None:
        model_path = MODEL_DIR / "linucb_policy.npz"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing trained policy at {model_path}. "
            "Run experiments/02_train_linucb.py first."
        )

    print("Building semantic concepts...")
    concepts = load_concepts(
        num_samples=NUM_SEMANTIC_SAMPLES,
        num_clusters=NUM_SEMANTIC_CLUSTERS,
        seed=seed,
        data_root=str(DATA_ROOT),
    )
    agent = LinUCB.load(model_path)
    schedule = make_episode_schedule(num_episodes, seed=seed + 99)

    adaptive_rows: list[dict] = []
    fixed_rows: dict[float, list[dict]] = defaultdict(list)
    oracle_rewards: list[float] = []
    oracle_levels: list[float] = []

    for ep in schedule:
        context, stress, _ = probe_episode(ep)
        level = select_adaptive_level(
            agent,
            context,
            ep,
            num_concepts=concepts.num_concepts,
        )

        adaptive = evaluate_action(
            ep, concepts, level, context=context, network_stress=stress
        )
        arow = adaptive.to_dict()
        arow["policy"] = "adaptive"
        arow["tier"] = str(ep.get("tier", "?"))
        arow["context"] = json.dumps(arow["context"])
        adaptive_rows.append(arow)

        episode_fixed: dict[float, float] = {}
        for fixed_level in COMPRESSION_LEVELS:
            fixed = evaluate_action(
                ep,
                concepts,
                fixed_level,
                context=context,
                network_stress=stress,
            )
            frow = fixed.to_dict()
            frow["policy"] = f"fixed_{fixed_level}"
            frow["tier"] = str(ep.get("tier", "?"))
            frow["context"] = json.dumps(frow["context"])
            fixed_rows[fixed_level].append(frow)
            episode_fixed[fixed_level] = fixed.reward

        oracle_level = max(episode_fixed, key=episode_fixed.get)
        oracle_rewards.append(episode_fixed[oracle_level])
        oracle_levels.append(oracle_level)

        if ep["episode"] % 20 == 0 or ep["episode"] == 1:
            print(
                f"ep {ep['episode']:3d}/{num_episodes}  "
                f"tier={ep.get('tier', '?'):6s}  "
                f"adaptive_c={level:.1f}  FSQ={adaptive.reward:.3f}  "
                f"oracle_c={oracle_level:.1f}  "
                f"stress={stress:.3f}"
            )

    adaptive_summary = summarize(adaptive_rows)
    fixed_summary = {
        lv: summarize(rows) for lv, rows in sorted(fixed_rows.items(), reverse=True)
    }
    best_fixed = max(fixed_summary, key=lambda lv: fixed_summary[lv]["mean_fsq"])
    gain = adaptive_summary["mean_fsq"] - fixed_summary[best_fixed]["mean_fsq"]
    gain_pct = (
        100.0 * gain / fixed_summary[best_fixed]["mean_fsq"]
        if fixed_summary[best_fixed]["mean_fsq"] > 1e-12
        else 0.0
    )
    oracle_mean = mean(oracle_rewards)
    oracle_gap = oracle_mean - adaptive_summary["mean_fsq"]
    oracle_vs_best_fixed = oracle_mean - fixed_summary[best_fixed]["mean_fsq"]

    # stress-bin adaptation check
    stresses = [r["network_stress"] for r in adaptive_rows]
    q1 = sorted(stresses)[len(stresses) // 3]
    q2 = sorted(stresses)[(2 * len(stresses)) // 3]

    def bin_name(s: float) -> str:
        if s <= q1:
            return "low"
        if s <= q2:
            return "mid"
        return "high"

    adaptive_by_bin: dict[str, list[dict]] = defaultdict(list)
    for r in adaptive_rows:
        adaptive_by_bin[bin_name(r["network_stress"])].append(r)

    adaptive_by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in adaptive_rows:
        adaptive_by_tier[r["tier"]].append(r)

    fixed_by_tier: dict[str, dict[float, list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for lv, rows in fixed_rows.items():
        for r in rows:
            fixed_by_tier[r["tier"]][lv].append(r)

    tier_comparison = {}
    for tier, a_rows in adaptive_by_tier.items():
        a_fsq = mean(r["reward"] for r in a_rows)
        a_c = mean(r["compression_level"] for r in a_rows)
        fixed_fsq = {
            lv: mean(r["reward"] for r in rows)
            for lv, rows in fixed_by_tier[tier].items()
        }
        best_lv = max(fixed_fsq, key=fixed_fsq.get)
        tier_comparison[tier] = {
            "adaptive_fsq": a_fsq,
            "adaptive_compression": a_c,
            "best_fixed_level": best_lv,
            "best_fixed_fsq": fixed_fsq[best_lv],
            "fixed_1.0_fsq": fixed_fsq.get(1.0, 0.0),
            "gain_vs_best_fixed": a_fsq - fixed_fsq[best_lv],
            "gain_vs_fixed_1.0": a_fsq - fixed_fsq.get(1.0, 0.0),
        }

    all_rows = adaptive_rows + [r for rows in fixed_rows.values() for r in rows]
    csv_path = out_dir / "comparison_episodes.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "num_episodes": num_episodes,
        "model_path": str(model_path),
        "adaptive": adaptive_summary,
        "fixed": {str(k): v for k, v in fixed_summary.items()},
        "best_fixed_level": best_fixed,
        "adaptive_gain_over_best_fixed": gain,
        "adaptive_gain_percent": gain_pct,
        "oracle_mean_fsq": oracle_mean,
        "oracle_minus_adaptive": oracle_gap,
        "oracle_minus_best_fixed": oracle_vs_best_fixed,
        "oracle_action_counts": {
            str(lv): sum(1 for x in oracle_levels if abs(x - lv) < 1e-12)
            for lv in COMPRESSION_LEVELS
        },
        "tier_comparison": tier_comparison,
        "adaptive_mean_compression_by_stress_bin": {
            b: mean(r["compression_level"] for r in rows)
            for b, rows in adaptive_by_bin.items()
        },
        "adaptive_fsq_by_stress_bin": {
            b: mean(r["reward"] for r in rows) for b, rows in adaptive_by_bin.items()
        },
        "adaptive_mean_compression_by_tier": {
            t: mean(r["compression_level"] for r in rows)
            for t, rows in adaptive_by_tier.items()
        },
        "adaptive_fsq_by_tier": {
            t: mean(r["reward"] for r in rows) for t, rows in adaptive_by_tier.items()
        },
    }
    report_path = out_dir / "comparison_summary.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n=== Adaptive vs Fixed ===")
    print(
        f"Adaptive: FSQ={adaptive_summary['mean_fsq']:.4f}  "
        f"avg_c={adaptive_summary['mean_compression']:.3f}"
    )
    for lv, stats in fixed_summary.items():
        marker = " ← best fixed" if lv == best_fixed else ""
        print(f"Fixed {lv:.1f}: FSQ={stats['mean_fsq']:.4f}{marker}")
    print(f"Oracle (per-episode best arm): FSQ={oracle_mean:.4f}")
    print(f"Gain over best fixed: {gain:+.4f} ({gain_pct:+.1f}%)")
    print(f"Oracle − best fixed: {oracle_vs_best_fixed:+.4f}")
    print(f"Oracle − adaptive: {oracle_gap:+.4f}")
    print("\nPer-tier comparison (core research result):")
    for t, stats in tier_comparison.items():
        print(
            f"  {t:6s}: adaptive_c={stats['adaptive_compression']:.2f}  "
            f"adaptive_FSQ={stats['adaptive_fsq']:.3f}  "
            f"best_fixed={stats['best_fixed_level']} "
            f"(FSQ={stats['best_fixed_fsq']:.3f})  "
            f"Δvs1.0={stats['gain_vs_fixed_1.0']:+.3f}"
        )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--model", type=str, default="")
    args = p.parse_args()
    main(
        num_episodes=args.episodes,
        seed=args.seed,
        model_path=Path(args.model) if args.model else None,
    )
