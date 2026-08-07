"""Plotting helpers for superiority figure pack (no SeQUeNCe runtime deps)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import COMPRESSION_LEVELS, RESULT_DIR
from runtime.tiers import TIER_ORDER, tier_label

OUT_DIR = RESULT_DIR / "paper_results"
STATIC_LEVEL = 1.0

# One size for every figure so PDFs scale consistently in the paper.
# At width=\linewidth in LaTeX, all figures will have the same height.
FIG_WIDTH = 7.0
FIG_HEIGHT = 4.5
FIGSIZE = (FIG_WIDTH, FIG_HEIGHT)


def style():
    """Paper-ready style: white background, full box border."""
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


def save(fig, stem: Path):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        print(f"Wrote {path}")
    plt.close(fig)


def plot_overall_fsq(adaptive: float, fixed: dict[float, float], out_dir: Path):
    labels = ["Adaptive\n(proposed)"] + [f"Static\nc={lv:g}" for lv in COMPRESSION_LEVELS]
    values = [adaptive] + [fixed[lv] for lv in COMPRESSION_LEVELS]
    colors = ["#1f77b4"] + ["#d62728" if lv == STATIC_LEVEL else "#aaaaaa" for lv in COMPRESSION_LEVELS]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(np.arange(len(labels)), values, color=colors, edgecolor="white", width=0.72)
    bars[0].set_edgecolor("#0b3d66")
    bars[0].set_linewidth(1.5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Quantum semantic fidelity")
    ax.set_ylim(0, max(values) * 1.22)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    save(fig, out_dir / "figA_overall_fsq_adaptive_vs_static")


def plot_tier_fsq(tier_stats: dict, out_dir: Path):
    tiers = [t for t in TIER_ORDER if t in tier_stats]
    x = np.arange(len(tiers))
    w = 0.35
    adapt = [tier_stats[t]["adaptive"] for t in tiers]
    static = [tier_stats[t]["static_1.0"] for t in tiers]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    b1 = ax.bar(x - w / 2, adapt, w, label="Adaptive QSC (proposed)", color="#1f77b4")
    b2 = ax.bar(x + w / 2, static, w, label="Static QSC (c=1.0)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [tier_label(t).replace(" Network Condition", "\nNetwork Condition") for t in tiers]
    )
    ax.set_ylabel("Quantum semantic fidelity")
    ax.set_xlabel("Network condition")
    ax.legend(loc="upper right")
    ymax = max(adapt + static) * 1.25
    ax.set_ylim(0, ymax)
    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.008,
                f"{b.get_height():.3f}",
                ha="center",
                fontsize=8,
            )
    save(fig, out_dir / "figB_tier_fsq_adaptive_vs_static")


def plot_size_vs_fsq(rows: list[dict], out_dir: Path):
    xs = [r["num_samples"] for r in rows]
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.errorbar(
        xs,
        [r["adaptive_fsq"] for r in rows],
        yerr=[r["adaptive_fsq_sem"] for r in rows],
        fmt="-o",
        color="#1f77b4",
        linewidth=2.4,
        markersize=6,
        capsize=3,
        zorder=5,
        label="Adaptive QSC (proposed)",
    )

    cmap = plt.cm.Reds(np.linspace(0.35, 0.95, len(COMPRESSION_LEVELS)))
    for color, lv in zip(cmap, sorted(COMPRESSION_LEVELS)):
        key = f"static_{lv:.1f}_fsq"
        sem_key = f"static_{lv:.1f}_fsq_sem"
        line_style = "-" if abs(lv - 1.0) < 1e-12 else "--"
        ax.errorbar(
            xs,
            [r[key] for r in rows],
            yerr=[r[sem_key] for r in rows],
            fmt=line_style + "s",
            color=color,
            linewidth=1.8,
            markersize=4,
            capsize=2,
            alpha=0.95,
            label=f"Static QSC (c={lv:g})",
        )

    ax.set_xlabel(r"$|X|$ (input / dataset size)")
    ax.set_ylabel("Quantum semantic fidelity")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, 1.05)
    save(fig, out_dir / "figC_size_vs_fsq_adaptive_vs_static")


def plot_size_vs_resources(rows: list[dict], out_dir: Path):
    xs = [r["num_samples"] for r in rows]
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(
        xs,
        [r["adaptive_resources"] for r in rows],
        "-o",
        color="#1f77b4",
        linewidth=2.4,
        markersize=6,
        zorder=5,
        label="Adaptive QSC (proposed)",
    )

    cmap = plt.cm.Reds(np.linspace(0.35, 0.95, len(COMPRESSION_LEVELS)))
    ymax = 0.0
    for color, lv in zip(cmap, sorted(COMPRESSION_LEVELS)):
        key = f"static_{lv:.1f}_resources"
        ys = [r[key] for r in rows]
        ymax = max(ymax, max(ys))
        line_style = "-" if abs(lv - 1.0) < 1e-12 else "--"
        ax.plot(
            xs,
            ys,
            line_style + "s",
            color=color,
            linewidth=1.8,
            markersize=4,
            label=f"Static QSC (c={lv:g})",
        )

    ax.set_xlabel(r"$|X|$ (input / dataset size)")
    ax.set_ylabel("Quantum communication resources")
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(0, ymax * 1.15)
    save(fig, out_dir / "figD_size_vs_resources_adaptive_vs_static")


def plot_fidelity_vs_resources(curve: dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = {
        TIER_ORDER[0]: "#1f77b4",
        TIER_ORDER[1]: "#ff8c00",
        TIER_ORDER[2]: "#7b2cbf",
    }

    for tier in TIER_ORDER:
        label = tier_label(tier)
        adaptive = curve[tier]["adaptive"]
        static = curve[tier]["static"]

        ax.plot(
            adaptive["resources"],
            adaptive["fidelity"],
            "-D",
            color=colors[tier],
            linewidth=2.4,
            markersize=6,
            zorder=5,
            label=f"Adaptive QSC — {label}",
        )
        ax.plot(
            static["resources"],
            static["fidelity"],
            "--",
            color=colors[tier],
            linewidth=2.0,
            alpha=0.9,
            label=f"Static QSC — {label}",
        )

    ax.set_xlabel("Quantum communication resources")
    ax.set_ylabel("Quantum semantic fidelity")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.95, borderpad=0.4)
    save(fig, out_dir / "figE_fidelity_vs_resources")


def plot_size_vs_fsq_by_tier(rows: list[dict], out_dir: Path):
    xs = [0] + [r["num_samples"] for r in rows]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    colors = {
        TIER_ORDER[0]: "#1f77b4",
        TIER_ORDER[1]: "#ff8c00",
        TIER_ORDER[2]: "#7b2cbf",
    }

    for tier in TIER_ORDER:
        label = tier_label(tier)
        a_y = np.concatenate(
            [[0.0], np.maximum.accumulate([r[tier]["adaptive_fsq"] for r in rows])]
        )
        s_y = np.concatenate(
            [[0.0], np.maximum.accumulate([r[tier]["static_fsq"] for r in rows])]
        )
        ax.plot(
            xs,
            a_y,
            "-D",
            color=colors[tier],
            linewidth=2.4,
            markersize=6,
            zorder=5,
            label=f"Adaptive QSC — {label}",
        )
        ax.plot(
            xs,
            s_y,
            "--s",
            color=colors[tier],
            linewidth=2.0,
            markersize=4,
            alpha=0.9,
            label=f"Static QSC — {label}",
        )

    ax.set_xlabel(r"$|X|$ (input / dataset size)")
    ax.set_ylabel("Quantum semantic fidelity")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=8, ncol=2)
    save(fig, out_dir / "figH_size_vs_fsq_by_network_condition")


def plot_compression_by_tier(tier_c: dict, out_dir: Path):
    tiers = [t for t in TIER_ORDER if t in tier_c]
    vals = [tier_c[t] for t in tiers]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(
        [tier_label(t).replace(" Network Condition", "\nNetwork Condition") for t in tiers],
        vals,
        color="#1f77b4",
        width=0.55,
    )
    ax.axhline(STATIC_LEVEL, color="#d62728", linestyle="--", linewidth=1.8, label="Static c=1.0")
    ax.set_ylabel("Mean selected compression level")
    ax.set_xlabel("Network condition")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center")
    save(fig, out_dir / "figF_compression_by_condition")


def plot_fs_fc_breakdown(stats: dict, out_dir: Path):
    labels = ["Fs\n(semantic)", "Fc\n(communication)", "FSQ\n(joint)"]
    adapt = [stats["adaptive"]["fs"], stats["adaptive"]["fc"], stats["adaptive"]["fsq"]]
    static = [stats["static"]["fs"], stats["static"]["fc"], stats["static"]["fsq"]]
    x = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - w / 2, adapt, w, label="Adaptive QSC", color="#1f77b4")
    ax.bar(x + w / 2, static, w, label="Static QSC (c=1.0)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean value")
    ax.set_ylim(0, 1.15)
    ax.legend()
    save(fig, out_dir / "figG_fs_fc_fsq_breakdown")


def replot_from_summary(summary_path: Path | None = None, out_dir: Path | None = None):
    """Regenerate figures from a saved summary.json (no re-simulation)."""
    style()
    summary_path = summary_path or (OUT_DIR / "summary.json")
    out_dir = out_dir or OUT_DIR
    summary = json.loads(summary_path.read_text())

    cmp_ = summary["comparison"]
    fixed_fsq = {float(k): v for k, v in cmp_["fixed_fsq"].items()}
    plot_overall_fsq(cmp_["adaptive_fsq"], fixed_fsq, out_dir)
    plot_tier_fsq(cmp_["tier_stats"], out_dir)
    plot_compression_by_tier(cmp_["tier_compression"], out_dir)
    plot_fs_fc_breakdown(cmp_["breakdown"], out_dir)
    plot_size_vs_fsq(summary["size_curves"], out_dir)
    plot_size_vs_resources(summary["size_curves"], out_dir)
    plot_size_vs_fsq_by_tier(summary["size_fsq_by_tier"], out_dir)
    plot_fidelity_vs_resources(summary["resource_curves"], out_dir)
    print(f"Replot complete: {out_dir}")


if __name__ == "__main__":
    replot_from_summary()
