import json
import os.path as osp

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = [
    "linguistic_clustering",
    "embedding_clustering",
    "per_language",
    "all_mixed",
]
CONDITION_LABELS = {
    "linguistic_clustering": "Linguistic",
    "embedding_clustering": "Embedding",
    "per_language": "Per-Language",
    "all_mixed": "All Mixed",
}
LANGUAGES = ["ja", "ko", "mn", "ru", "en"]
PRIMARY_METRIC = "mongolian_f1"

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Baseline",
}


def generate_color_palette(n):
    cmap = plt.get_cmap("tab10")
    return [mcolors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, n)]


def load_run_data(run_dir):
    final_info_path = osp.join(run_dir, "final_info.json")
    if not osp.exists(final_info_path):
        return None
    with open(final_info_path, "r") as f:
        final_info = json.load(f)
    clustering_meta = None
    meta_path = osp.join(run_dir, "clustering_meta.json")
    if osp.exists(meta_path):
        with open(meta_path, "r") as f:
            clustering_meta = json.load(f)
    return {"final_info": final_info, "clustering_meta": clustering_meta}


runs = [r for r in labels if osp.isdir(r)]
colors = generate_color_palette(len(runs))
run_data = {run: load_run_data(run) for run in runs}

# Plot 1: Mongolian F1 across conditions (primary metric)
plt.figure(figsize=(10, 6))
x = np.arange(len(CONDITIONS))
width = 0.8 / max(len(runs), 1)

for i, run in enumerate(runs):
    data = run_data[run]
    if data is None:
        continue
    means = []
    stderrs = []
    for cond in CONDITIONS:
        cond_data = data["final_info"].get(cond, {})
        means.append(cond_data.get("means", {}).get(PRIMARY_METRIC, 0.0))
        stderrs.append(cond_data.get("stderrs", {}).get(PRIMARY_METRIC, 0.0))
    offset = (i - (len(runs) - 1) / 2) * width
    plt.bar(
        x + offset,
        means,
        width,
        yerr=stderrs,
        label=labels[run],
        color=colors[i],
        capsize=4,
    )

plt.xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS])
plt.ylabel("Mongolian Entity F1")
plt.title("Primary Metric: Mongolian (mn) F1 Across Training Regimes")
plt.ylim(0, 1.0)
plt.legend()
plt.grid(True, axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig("mongolian_f1_across_conditions.png")
plt.close()

# Plot 2: Per-language F1 for run_0 baseline (grouped by condition)
baseline_run = "run_0" if "run_0" in run_data and run_data["run_0"] else runs[0] if runs else None
if baseline_run and run_data[baseline_run]:
    plt.figure(figsize=(12, 6))
    x = np.arange(len(LANGUAGES))
    width = 0.8 / len(CONDITIONS)

    for i, cond in enumerate(CONDITIONS):
        cond_data = run_data[baseline_run]["final_info"].get(cond, {})
        means = [cond_data.get("means", {}).get(lang, 0.0) for lang in LANGUAGES]
        stderrs = [cond_data.get("stderrs", {}).get(lang, 0.0) for lang in LANGUAGES]
        offset = (i - (len(CONDITIONS) - 1) / 2) * width
        plt.bar(
            x + offset,
            means,
            width,
            yerr=stderrs,
            label=CONDITION_LABELS[cond],
            capsize=3,
        )

    plt.xticks(x, LANGUAGES)
    plt.ylabel("Entity F1")
    plt.title(f"Per-Language F1 by Condition ({labels.get(baseline_run, baseline_run)})")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig("per_language_f1_by_condition.png")
    plt.close()

# Plot 3: Cluster assignments visualization (baseline run)
if baseline_run and run_data[baseline_run]:
    meta = run_data[baseline_run].get("clustering_meta")
    if meta:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, (key, title) in zip(
            axes,
            [
                ("linguistic", "Linguistic Clustering (Head Parameter)"),
                ("embedding", "Embedding Clustering (Agglomerative)"),
            ],
        ):
            section = meta.get(key, {})
            cluster_map = section.get("cluster_map", {})
            if not cluster_map:
                continue
            langs = list(cluster_map.keys())
            cluster_ids = [cluster_map[l] for l in langs]
            unique_clusters = sorted(set(cluster_ids))
            palette = generate_color_palette(len(unique_clusters))
            cid_to_color = {c: palette[i] for i, c in enumerate(unique_clusters)}
            bar_colors = [cid_to_color[cid] for cid in cluster_ids]
            ax.bar(langs, [1] * len(langs), color=bar_colors)
            ax.set_title(title)
            ax.set_ylabel("Cluster Assignment")
            ax.set_ylim(0, 1.2)
            for j, (lang, cid) in enumerate(zip(langs, cluster_ids)):
                ax.text(j, 0.5, f"C{cid}", ha="center", va="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig("cluster_assignments.png")
        plt.close()

print("Plots saved: mongolian_f1_across_conditions.png, per_language_f1_by_condition.png, cluster_assignments.png")
