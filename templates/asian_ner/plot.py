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
    "matched_random_embedding",
    "matched_random_linguistic",
]
CONDITION_LABELS = {
    "linguistic_clustering": "Linguistic",
    "embedding_clustering": "Embedding",
    "per_language": "Per-Language",
    "all_mixed": "All Mixed",
    "matched_random_embedding": "Matched (Emb)",
    "matched_random_linguistic": "Matched (Ling)",
}
PRIMARY_METRIC = "mongolian_f1"
LOW_RESOURCE_METRIC = "low_resource_macro_f1"

# CREATE LEGEND -- ADD RUNS HERE THAT WILL BE PLOTTED
labels = {
    "run_0": "Human floor",
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
plt.figure(figsize=(12, 6))
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

plt.xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS], rotation=25, ha="right")
plt.ylabel("Mongolian F1")
plt.title("Mongolian NER F1 by Training Regime")
plt.legend()
plt.tight_layout()
plt.savefig("mongolian_f1_by_condition.png", dpi=150)
plt.close()

# Plot 2: Low-resource macro F1
plt.figure(figsize=(12, 6))
for i, run in enumerate(runs):
    data = run_data[run]
    if data is None:
        continue
    means = []
    stderrs = []
    for cond in CONDITIONS:
        cond_data = data["final_info"].get(cond, {})
        means.append(cond_data.get("means", {}).get(LOW_RESOURCE_METRIC, 0.0))
        stderrs.append(cond_data.get("stderrs", {}).get(LOW_RESOURCE_METRIC, 0.0))
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

plt.xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS], rotation=25, ha="right")
plt.ylabel("Low-Resource Macro F1")
plt.title("Low-Resource Stratum Macro F1 by Training Regime")
plt.legend()
plt.tight_layout()
plt.savefig("low_resource_macro_f1_by_condition.png", dpi=150)
plt.close()

# Plot 3: Per-language F1 heatmap for the first available run
if runs and run_data[runs[0]] is not None:
    fi = run_data[runs[0]]["final_info"]
    skip = {PRIMARY_METRIC, LOW_RESOURCE_METRIC, "average_f1", "mongolian_f1"}
    langs = sorted(
        k
        for k in fi.get("embedding_clustering", {}).get("means", {})
        if k not in skip
    )
    plot_conditions = [
        "linguistic_clustering",
        "embedding_clustering",
        "matched_random_embedding",
        "matched_random_linguistic",
        "all_mixed",
    ]
    matrix = []
    for cond in plot_conditions:
        row = [fi.get(cond, {}).get("means", {}).get(lang, 0.0) for lang in langs]
        matrix.append(row)
    matrix = np.array(matrix)

    plt.figure(figsize=(14, 5))
    plt.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(label="F1")
    plt.yticks(range(len(plot_conditions)), [CONDITION_LABELS[c] for c in plot_conditions])
    plt.xticks(range(len(langs)), langs, rotation=90)
    plt.title(f"Per-Language F1 ({runs[0]})")
    plt.tight_layout()
    plt.savefig("per_language_f1_heatmap.png", dpi=150)
    plt.close()

print("Saved mongolian_f1_by_condition.png, low_resource_macro_f1_by_condition.png")
