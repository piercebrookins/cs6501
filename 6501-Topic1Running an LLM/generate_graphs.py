#!/usr/bin/env python3
"""
Generate Graphs from MMLU Evaluation Results

Creates visualizations:
1. Model accuracy comparison (bar chart)
2. Subject-wise accuracy heatmap
3. Timing comparison (real, CPU, GPU)
4. Error pattern analysis

Usage:
    python generate_graphs.py [results_file.json]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f39c12"]


def load_results(filepath: str = "mmlu_results.json") -> dict:
    """Load evaluation results from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)


def plot_accuracy_comparison(results: dict, output_dir: Path):
    """Bar chart comparing model accuracies."""
    models = []
    accuracies = []

    for model_name, data in results["models"].items():
        if isinstance(data, dict) and "overall_accuracy" in data:
            models.append(model_name)
            accuracies.append(data["overall_accuracy"])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=COLORS[: len(models)], edgecolor="black")

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("MMLU Accuracy Comparison by Model", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=150)
    plt.savefig(output_dir / "accuracy_comparison.pdf")
    plt.close()
    print("✓ Saved accuracy_comparison.png/pdf")


def plot_timing_comparison(results: dict, output_dir: Path):
    """Grouped bar chart comparing timing metrics."""
    models = []
    real_times = []
    cpu_times = []
    gpu_times = []

    for model_name, data in results["models"].items():
        if isinstance(data, dict) and "timing" in data:
            models.append(model_name)
            timing = data["timing"]
            real_times.append(timing["total_real_time_sec"])
            cpu_times.append(timing["total_cpu_time_sec"])
            gpu_times.append(timing["total_gpu_time_sec"])

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, real_times, width, label="Real Time", color="#3498db")
    bars2 = ax.bar(x, cpu_times, width, label="CPU Time", color="#2ecc71")
    bars3 = ax.bar(x + width, gpu_times, width, label="GPU Time", color="#e74c3c")

    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Timing Comparison by Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "timing_comparison.png", dpi=150)
    plt.savefig(output_dir / "timing_comparison.pdf")
    plt.close()
    print("✓ Saved timing_comparison.png/pdf")


def plot_subject_heatmap(results: dict, output_dir: Path):
    """Heatmap of accuracy by model and subject."""
    # Build data matrix
    models = []
    subjects = set()
    data = {}

    for model_name, model_data in results["models"].items():
        if isinstance(model_data, dict) and "subjects" in model_data:
            models.append(model_name)
            data[model_name] = {}
            for subj in model_data["subjects"]:
                subjects.add(subj["subject"])
                data[model_name][subj["subject"]] = subj["accuracy"]

    subjects = sorted(subjects)

    # Create matrix
    matrix = np.zeros((len(models), len(subjects)))
    for i, model in enumerate(models):
        for j, subj in enumerate(subjects):
            matrix[i, j] = data.get(model, {}).get(subj, 0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(subjects)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([s.replace("_", " ").title()[:15] for s in subjects], rotation=45, ha="right")
    ax.set_yticklabels(models)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(subjects)):
            text = ax.text(
                j, i, f"{matrix[i, j]:.0f}",
                ha="center", va="center", color="black", fontsize=8
            )

    ax.set_title("Accuracy by Model and Subject (%)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Accuracy (%)")

    plt.tight_layout()
    plt.savefig(output_dir / "subject_heatmap.png", dpi=150)
    plt.savefig(output_dir / "subject_heatmap.pdf")
    plt.close()
    print("✓ Saved subject_heatmap.png/pdf")


def analyze_error_patterns(results: dict, output_dir: Path):
    """Analyze and visualize error patterns across models."""
    # Collect all question results
    question_results = {}  # question -> {model: correct/incorrect}

    for model_name, model_data in results["models"].items():
        if not isinstance(model_data, dict) or "subjects" not in model_data:
            continue
        for subj in model_data["subjects"]:
            for q in subj.get("questions", []):
                q_id = q["question"][:50]  # Use truncated question as ID
                if q_id not in question_results:
                    question_results[q_id] = {}
                question_results[q_id][model_name] = q["is_correct"]

    if not question_results:
        print("⚠️  No question-level data for error analysis")
        return

    # Count how many models got each question wrong
    models = list(results["models"].keys())
    n_models = len(models)

    all_wrong = 0  # All models wrong
    all_right = 0  # All models right
    mixed = 0      # Some right, some wrong

    for q_id, model_results in question_results.items():
        correct_count = sum(1 for m in models if model_results.get(m, False))
        if correct_count == 0:
            all_wrong += 1
        elif correct_count == n_models:
            all_right += 1
        else:
            mixed += 1

    # Pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    sizes = [all_right, mixed, all_wrong]
    labels = [
        f"All Correct\n({all_right})",
        f"Mixed Results\n({mixed})",
        f"All Wrong\n({all_wrong})",
    ]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    explode = (0.02, 0.02, 0.05)

    ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11}
    )
    ax.set_title("Error Pattern Analysis\n(Do models make the same mistakes?)", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "error_patterns.png", dpi=150)
    plt.savefig(output_dir / "error_patterns.pdf")
    plt.close()
    print("✓ Saved error_patterns.png/pdf")

    # Write analysis summary
    total = all_wrong + all_right + mixed
    analysis = f"""
# Error Pattern Analysis

Total Questions: {total}

## Agreement Patterns:
- **All models correct**: {all_right} ({all_right/total*100:.1f}%)
- **Mixed results**: {mixed} ({mixed/total*100:.1f}%)
- **All models wrong**: {all_wrong} ({all_wrong/total*100:.1f}%)

## Interpretation:
- If "All Wrong" is high: Questions may be inherently difficult
- If "Mixed" is high: Models have different strengths/weaknesses
- High "All Correct" + "All Wrong" with low "Mixed": Models behave similarly
"""
    with open(output_dir / "error_analysis.md", "w") as f:
        f.write(analysis)
    print("✓ Saved error_analysis.md")


def main():
    # Determine results file
    results_file = sys.argv[1] if len(sys.argv) > 1 else "mmlu_results.json"

    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        print("Run enhanced_mmlu_eval.py first to generate results.")
        sys.exit(1)

    print(f"\n📊 Loading results from {results_file}...")
    results = load_results(results_file)

    # Create output directory
    output_dir = Path("graphs")
    output_dir.mkdir(exist_ok=True)

    print(f"\n📈 Generating graphs in {output_dir}/...\n")

    # Generate all graphs
    plot_accuracy_comparison(results, output_dir)
    plot_timing_comparison(results, output_dir)
    plot_subject_heatmap(results, output_dir)
    analyze_error_patterns(results, output_dir)

    print(f"\n✅ All graphs saved to {output_dir}/")
    print("\nGenerated files:")
    for f in sorted(output_dir.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
