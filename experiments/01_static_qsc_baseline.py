"""
Experiment 01 — Static QSC baseline (paper setting).

Fixed compression levels, no MAB. Evaluates each fixed |S| across
continuously varying physical SeQUeNCe conditions.

This is the static semantic-representation baseline that the professor
wants to move beyond.
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
    NUM_SEMANTIC_CLUSTERS,
    NUM_SEMANTIC_SAMPLES,
    RANDOM_SEED,
    RESULT_DIR,
)
from runtime.episode import evaluate_action, load_concepts, probe_episode
from runtime.sampling import make_episode_schedule


def summarize(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "mean_fsq": mean(r["reward"] for r in rows),
        "mean_fs": mean(r["semantic_fidelity"] for r in rows),
        "mean_fc": mean(r["communication_fidelity"] for r in rows),
        "mean_drop_ratio": mean(r["drop_ratio"] for r in rows),
        "mean_num_states": mean(r["num_states"] for r in rows),
        "mean_stress": mean(r["network_stress"] for r in rows),
    }


def main(
    num_episodes: int = 60,
    seed: int = RANDOM_SEED,
):
    out_dir = RESULT_DIR / "01_static_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building semantic concepts...")
    concepts = load_concepts(
        num_samples=NUM_SEMANTIC_SAMPLES,
        num_clusters=NUM_SEMANTIC_CLUSTERS,
        seed=seed,
        data_root=str(DATA_ROOT),
    )
    print(f"  K = {concepts.num_concepts} concepts")

    schedule = make_episode_schedule(num_episodes, seed=seed + 1)
    all_rows: list[dict] = []
    by_level: dict[float, list[dict]] = defaultdict(list)

    for ep in schedule:
        context, stress, _ = probe_episode(ep)
        for level in COMPRESSION_LEVELS:
            outcome = evaluate_action(
                ep,
                concepts,
                level,
                context=context,
                network_stress=stress,
            )
            row = outcome.to_dict()
            row["tier"] = str(ep.get("tier", "?"))
            row["context"] = json.dumps(row["context"])
            all_rows.append(row)
            by_level[level].append(row)
        print(
            f"episode {ep['episode']:3d}/{num_episodes}  "
            f"tier={ep.get('tier', '?'):6s}  "
            f"stress={stress:.3f}  dist={ep['distance']}  load={ep['offered_load']:.2f}"
        )

    summary = {
        level: summarize(rows) for level, rows in sorted(by_level.items(), reverse=True)
    }
    best_level = max(summary, key=lambda lv: summary[lv]["mean_fsq"])

    tier_rows: dict[str, dict[float, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        tier_rows[row["tier"]][float(row["compression_level"])].append(row)

    best_by_tier = {
        tier: max(levels, key=lambda lv: summarize(levels[lv])["mean_fsq"])
        for tier, levels in tier_rows.items()
    }
    tier_summaries = {
        tier: {str(lv): summarize(rows) for lv, rows in levels.items()}
        for tier, levels in tier_rows.items()
    }

    csv_path = out_dir / "static_episode_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "num_episodes": num_episodes,
        "compression_levels": COMPRESSION_LEVELS,
        "per_level": {str(k): v for k, v in summary.items()},
        "best_fixed_level": best_level,
        "best_fixed_fsq": summary[best_level]["mean_fsq"],
        "best_fixed_by_tier": best_by_tier,
        "per_tier_per_level": tier_summaries,
    }
    report_path = out_dir / "static_summary.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\n=== Static QSC baseline ===")
    for level, stats in summary.items():
        print(
            f"  c={level:.1f}  |S|≈{stats['mean_num_states']:.1f}  "
            f"FSQ={stats['mean_fsq']:.4f}  Fs={stats['mean_fs']:.4f}  "
            f"Fc={stats['mean_fc']:.4f}  drop={stats['mean_drop_ratio']:.3f}"
        )
    print(f"Best fixed overall: {best_level} (FSQ={summary[best_level]['mean_fsq']:.4f})")
    print("Best fixed by tier:", best_by_tier)
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = p.parse_args()
    main(num_episodes=args.episodes, seed=args.seed)
