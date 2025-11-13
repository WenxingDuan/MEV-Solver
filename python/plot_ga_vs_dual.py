#!/usr/bin/env python3
from __future__ import annotations

"""Plotting utility that visualises GA vs. deterministic trade-offs."""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def load_summary(path: Path) -> List[Dict[str, Any]]:
    """Parse the JSON summary emitted by evaluate_engines.py."""
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Summary file must contain a list of instance records")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GA vs DualDecomposition trade-offs.")
    parser.add_argument("--summary", required=True, help="JSON output from evaluate_engines.py")
    parser.add_argument("--output", required=True, help="Path to save the figure (PNG/SVG).")
    args = parser.parse_args()

    summary = load_summary(Path(args.summary))
    if not summary:
        raise ValueError("Summary is empty")

    frag = [entry["fragmentationScore"] for entry in summary]
    ga_surplus = [entry["ga"]["surplus"] for entry in summary]
    det_surplus = [entry["deterministic"]["surplus"] for entry in summary]

    market_count = [entry["marketCount"] for entry in summary]
    ga_runtime = [entry["ga"]["runtimeMs"] for entry in summary]
    det_runtime = [entry["deterministic"]["runtimeMs"] for entry in summary]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax0 = axes[0, 0]
    ax0.scatter(frag, det_surplus, label="DualDecomposition", marker="x", color="#a855f7")
    ax0.scatter(frag, ga_surplus, label="GeneticAlgorithm", marker="o", facecolors="none", edgecolors="#2563eb")
    for x, y0, y1 in zip(frag, det_surplus, ga_surplus):
        ax0.plot([x, x], [y0, y1], color="#94a3b8", linewidth=0.8)
    ax0.set_xlabel("Fragmentation Score (complexity proxy)")
    ax0.set_ylabel("Quote-Token Output")
    ax0.set_title("Optimization Effect vs. Fragmentation")
    ax0.legend()

    ax1 = axes[0, 1]
    ax1.scatter(
        frag,
        [entry["deterministic"].get("slippage", 0.0) for entry in summary],
        label="DualDecomposition",
        marker="x",
        color="#a855f7",
    )
    ax1.scatter(
        frag,
        [entry["ga"]["slippage"] for entry in summary],
        label="GeneticAlgorithm",
        marker="o",
        facecolors="none",
        edgecolors="#2563eb",
    )
    ax1.set_xlabel("Fragmentation Score")
    ax1.set_ylabel("Slippage (quote-token units)")
    ax1.set_title("Slippage vs. Fragmentation")
    ax1.legend()

    ax2 = axes[1, 0]
    ax2.scatter(market_count, det_runtime, label="DualDecomposition", marker="x", color="#a855f7")
    ax2.scatter(market_count, ga_runtime, label="GeneticAlgorithm", marker="o", facecolors="none", edgecolors="#2563eb")
    for x, y0, y1 in zip(market_count, det_runtime, ga_runtime):
        ax2.plot([x, x], [y0, y1], color="#94a3b8", linewidth=0.8)
    ax2.set_xlabel("Market Count (complexity proxy)")
    ax2.set_ylabel("Runtime (ms)")
    ax2.set_title("Computational Cost vs. Market Graph Size")
    ax2.legend()

    ax3 = axes[1, 1]
    ga_opg = [entry["ga"].get("outputPerGas", 0.0) for entry in summary]
    det_opg = [entry["deterministic"].get("outputPerGas", 0.0) for entry in summary]
    sizes = [max(30.0, entry["ga"].get("pathCount", 0) * 40.0) for entry in summary]
    ax3.scatter(market_count, det_opg, label="DualDecomposition", marker="x", color="#a855f7")
    ax3.scatter(
        market_count,
        ga_opg,
        label="GeneticAlgorithm",
        marker="o",
        facecolors="none",
        edgecolors="#2563eb",
        s=sizes,
        alpha=0.9,
    )
    for x, y0, y1 in zip(market_count, det_opg, ga_opg):
        ax3.plot([x, x], [y0, y1], color="#94a3b8", linewidth=0.8, alpha=0.7)
    ax3.set_xlabel("Market Count (complexity proxy)")
    ax3.set_ylabel("Output per Gas Unit")
    ax3.set_title("Capital Efficiency vs. Market Graph Size")
    ax3.legend(title="Marker size ∝ GA path count")

    fig.tight_layout()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    main()
