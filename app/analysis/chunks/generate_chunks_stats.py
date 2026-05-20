import json
import statistics
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from app.core.utils import write_json, reset_json

results_dir = Path(__file__).resolve().parent / "results"

def generate_chunks_stats(path: Path, chunk_size: int, chunk_overlap_pct: float):
    token_counts = []
    overlap_counts = []
    chapter_counts = defaultdict(int)

    max_overlap = round(chunk_size * chunk_overlap_pct / 100)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            meta = row.get("metadata", {})

            tokens = meta.get("total_tokens", 0)
            overlap = meta.get("total_overlap_tokens", 0)
            chapter = meta.get("chapter", "unknown")

            token_counts.append(tokens)
            overlap_counts.append(overlap)
            chapter_counts[chapter] += 1

    # -----------------------
    # Stats
    # -----------------------
    stats = {
        "total_chunks": len(token_counts),
        "total_tokens": sum(token_counts),
        "avg_tokens": statistics.mean(token_counts) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "avg_overlap": statistics.mean(overlap_counts) if overlap_counts else 0,
        "chunk_size_limit": chunk_size,
        "overlap_limit": max_overlap,
        "chapters": dict(chapter_counts),
    }

    # Saving Stats
    reset_json(json_path=results_dir / "chunks_stats.json")
    write_json(data=stats, output_path=results_dir / "chunks_stats.json")

    # -----------------------
    # Console report
    # -----------------------
    print("\n" + "=" * 60)
    print("📊 RAG CHUNK STATISTICS REPORT")
    print("=" * 60)

    print(f"Total chunks       : {stats['total_chunks']}")
    print(f"Total tokens       : {stats['total_tokens']}")
    print(f"Avg tokens         : {stats['avg_tokens']:.2f}")
    print(f"Min / Max tokens   : {stats['min_tokens']} / {stats['max_tokens']}")
    print(f"Avg overlap tokens : {stats['avg_overlap']:.2f}")

    print("\n📁 Chapters distribution:")
    for k, v in stats["chapters"].items():
        print(f"  - {k}: {v}")

    print("\n⚙️ Limits:")
    print(f"  - Chunk size max     : {chunk_size}")
    print(f"  - Overlap max (%)    : {chunk_overlap_pct}")
    print(f"  - Overlap max (abs)  : {max_overlap}")

    print("=" * 60 + "\n")

    # -----------------------
    # Histogram plotting
    # -----------------------
    def plot(values, limit, title, output_path, bins=15):

        plt.figure()

        # Build histogram
        counts, bin_edges = np.histogram(values, bins=bins)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        # Create readable range labels like "0-128"
        bin_labels = [
            f"{int(bin_edges[i])}-{int(bin_edges[i+1])}"
            for i in range(len(bin_edges) - 1)
        ]

        # Color bins based on threshold
        colors = ["green" if x <= limit else "orange" for x in bin_centers]

        bars = plt.bar(
            bin_centers,
            counts,
            width=bin_width,
            color=colors,
            edgecolor="black"
        )

        # Value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    str(int(height)),
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        # Replace x-axis ticks with bin ranges
        plt.xticks(bin_centers, bin_labels, rotation=45, ha="right")

        # Threshold line
        plt.axvline(limit, color="black", linestyle="--", linewidth=1)

        plt.title(title)
        plt.xlabel("Value Range")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    # Token histogram
    plot(
        token_counts,
        chunk_size,
        "Chunk Token Distribution",
        results_dir / "chunk_tokens_hist.png",
    )

    # Overlap histogram
    plot(
        overlap_counts,
        max_overlap,
        "Chunk Overlap Distribution",
        results_dir / "chunk_overlap_hist.png",
    )

    print("📈 Saved plots:")
    print("  - chunk_tokens_hist.png")
    print("  - chunk_overlap_hist.png")

    return stats