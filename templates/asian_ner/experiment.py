"""
asian_ner — AI Scientist v1 template (minimal PoC)

Compares four multilingual NER training regimes on WikiAnn:
  1. linguistic_clustering  — Head-parameter grouping (ACL 2023 lineage)
  2. embedding_clustering   — XLM-R [CLS] embeddings + agglomerative clustering
  3. per_language           — monolingual baseline (Mongolian only for mn metric)
  4. all_mixed              — train on all languages jointly

Primary metric: Mongolian (mn) entity-level F1 under 100-sample WikiAnn regime.

Usage:
  python experiment.py --out_dir run_0
  python experiment.py --out_dir run_0 --quick   # fewer steps for smoke test

Dependencies (AI-Scientist venv):
  torch, transformers, datasets, sklearn, seqeval, numpy, accelerate
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.cluster import AgglomerativeClustering
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

try:
    from seqeval.metrics import f1_score as seqeval_f1
except ImportError:  # pragma: no cover
    seqeval_f1 = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_NAME = "xlm-roberta-base"
LANGUAGES = ["ja", "ko", "mn", "ru", "en"]
TARGET_LANG = "mn"
PRIMARY_METRIC_KEY = "mongolian_f1"

HEAD_FINAL = {"ja", "ko", "mn"}
HEAD_INITIAL = {"en", "ru"}

LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

NER_TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


@dataclass
class RunConfig:
    out_dir: str
    seed: int
    max_epochs: int
    batch_size: int
    learning_rate: float
    max_length: int
    max_train_per_lang: int
    max_embed_samples_per_lang: int
    max_eval_samples_per_lang: int
    lang_id_epochs: int
    n_clusters: int
    quick: bool


def build_config(args: argparse.Namespace) -> RunConfig:
    quick = args.quick
    return RunConfig(
        out_dir=args.out_dir,
        seed=args.seed,
        max_epochs=1 if quick else args.max_epochs,
        batch_size=8 if quick else args.batch_size,
        learning_rate=args.learning_rate,
        max_length=96 if quick else args.max_length,
        max_train_per_lang=50 if quick else args.max_train_per_lang,
        max_embed_samples_per_lang=50 if quick else args.max_embed_samples_per_lang,
        max_eval_samples_per_lang=50 if quick else 0,
        lang_id_epochs=1,
        n_clusters=2,
        quick=quick,
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Skip MPS: token-classification eval hits position_ids bugs on Apple Silicon.
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_wikiann_split(lang: str, split: str) -> Dataset:
    ds = load_dataset("wikiann", lang, split=split, trust_remote_code=True)
    return ds


def cap_dataset(ds: Dataset, max_samples: int) -> Dataset:
    if max_samples <= 0 or len(ds) <= max_samples:
        return ds
    return ds.select(range(max_samples))


def tokenize_and_align(
    examples: Dict,
    tokenizer,
    max_length: int,
) -> Dict:
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )
    labels = []
    for i, tag_row in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word:
                label_ids.append(int(tag_row[word_id]))
            else:
                label_ids.append(-100)
            prev_word = word_id
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized


def prepare_ner_dataset(
    langs: List[str],
    split: str,
    tokenizer,
    cfg: RunConfig,
) -> Dataset:
    parts = []
    for lang in langs:
        ds = load_wikiann_split(lang, split)
        if split == "train":
            ds = cap_dataset(ds, cfg.max_train_per_lang)
        parts.append(ds)
    merged = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    return merged.map(
        lambda x: tokenize_and_align(x, tokenizer, cfg.max_length),
        batched=True,
        remove_columns=merged.column_names,
    )


# ---------------------------------------------------------------------------
# Clustering  (EVOLVE-BLOCK — AI Scientist may modify these functions)
# ---------------------------------------------------------------------------


def cluster_linguistic(langs: List[str]) -> Dict[str, int]:
    """Head-parameter rule-based clustering."""
    mapping: Dict[str, int] = {}
    for lang in langs:
        if lang in HEAD_FINAL:
            mapping[lang] = 0
        elif lang in HEAD_INITIAL:
            mapping[lang] = 1
        else:
            mapping[lang] = 1
    return mapping


def _sentence_cls_embeddings(
    langs: List[str],
    cfg: RunConfig,
    device: torch.device,
) -> Tuple[Dict[str, np.ndarray], AutoTokenizer]:
    """Language-ID fine-tuned XLM-R [CLS] embeddings per language (mean-pooled)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(langs),
        id2label={i: l for i, l in enumerate(langs)},
        label2id={l: i for i, l in enumerate(langs)},
    ).to(device)

    train_texts: List[str] = []
    train_labels: List[int] = []
    lang2id = {l: i for i, l in enumerate(langs)}
    for lang in langs:
        ds = cap_dataset(load_wikiann_split(lang, "validation"), cfg.max_embed_samples_per_lang)
        for row in ds:
            train_texts.append(" ".join(row["tokens"]))
            train_labels.append(lang2id[lang])

    enc = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    labels_t = torch.tensor(train_labels, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    model.train()
    bs = min(cfg.batch_size, len(train_texts))
    for _ in range(cfg.lang_id_epochs):
        perm = torch.randperm(len(train_texts))
        for start in range(0, len(train_texts), bs):
            idx = perm[start : start + bs]
            batch = {k: v[idx] for k, v in enc.items()}
            y = labels_t[idx]
            out = model(**batch, labels=y)
            loss = out.loss
            loss.backward()
            optim.step()
            optim.zero_grad()

    model.eval()
    lang_embeddings: Dict[str, List[np.ndarray]] = {l: [] for l in langs}
    with torch.no_grad():
        for lang in langs:
            ds = cap_dataset(load_wikiann_split(lang, "validation"), cfg.max_embed_samples_per_lang)
            for row in ds:
                inputs = tokenizer(
                    " ".join(row["tokens"]),
                    truncation=True,
                    max_length=cfg.max_length,
                    return_tensors="pt",
                ).to(device)
                hidden = model.base_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()[0]
                lang_embeddings[lang].append(hidden)

    pooled = {lang: np.mean(v, axis=0) for lang, v in lang_embeddings.items()}
    return pooled, tokenizer


def cluster_embedding(
    langs: List[str],
    cfg: RunConfig,
    device: torch.device,
) -> Tuple[Dict[str, int], Dict]:
    """Embedding-based clustering following Shaffer (2021) / Imai et al. (2023)."""
    pooled, _ = _sentence_cls_embeddings(langs, cfg, device)
    keys = list(langs)
    matrix = np.stack([pooled[k] for k in keys], axis=0)
    clustering = AgglomerativeClustering(n_clusters=cfg.n_clusters)
    labels = clustering.fit_predict(matrix)
    cluster_map = {lang: int(label) for lang, label in zip(keys, labels)}
    meta = {
        "method": "agglomerative",
        "n_clusters": cfg.n_clusters,
        "linkage": "ward",
        "distance_metric": "euclidean",
        "embedding_source": "xlm-roberta-base_lang_id_cls",
        "cluster_map": cluster_map,
    }
    return cluster_map, meta


def langs_in_cluster(target: str, cluster_map: Dict[str, int]) -> List[str]:
    cid = cluster_map[target]
    return [l for l, c in cluster_map.items() if c == cid]


# ---------------------------------------------------------------------------
# NER train / eval
# ---------------------------------------------------------------------------


def train_and_evaluate_ner(
    train_langs: List[str],
    eval_langs: List[str],
    cfg: RunConfig,
    device: torch.device,
    save_dir: Optional[str] = None,
) -> Dict[str, float]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = prepare_ner_dataset(train_langs, "train", tokenizer, cfg)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    args = TrainingArguments(
        output_dir=save_dir or os.path.join(cfg.out_dir, "_tmp_train"),
        num_train_epochs=cfg.max_epochs,
        per_device_train_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=cfg.seed,
        use_cpu=device.type == "cpu",
    )
    collator = DataCollatorForTokenClassification(tokenizer)
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)
    trainer.train()
    model.to(device)

    scores: Dict[str, float] = {}
    for lang in eval_langs:
        scores[lang] = evaluate_lang_f1(model, tokenizer, lang, cfg, device)
    scores[PRIMARY_METRIC_KEY] = scores.get(TARGET_LANG, 0.0)
    scores["average_f1"] = float(np.mean(list(scores.values())))
    return scores


def evaluate_lang_f1(
    model,
    tokenizer,
    lang: str,
    cfg: RunConfig,
    device: torch.device,
) -> float:
    if seqeval_f1 is None:
        raise ImportError("seqeval is required: pip install seqeval")

    test_ds = load_wikiann_split(lang, "test")
    test_ds = cap_dataset(test_ds, cfg.max_eval_samples_per_lang)
    model.eval()
    all_preds: List[List[str]] = []
    all_refs: List[List[str]] = []

    for row in test_ds:
        tokenized = tokenizer(
            row["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        word_ids = tokenized.word_ids(batch_index=0)
        inputs = {k: v.to(device) for k, v in tokenized.items() if k != "offset_mapping"}
        with torch.no_grad():
            logits = model(**inputs).logits[0].cpu().numpy()
        pred_ids = logits.argmax(-1)
        preds, refs = [], []
        prev = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id == prev:
                continue
            preds.append(ID2LABEL.get(int(pred_ids[idx]), "O"))
            refs.append(NER_TAGS[int(row["ner_tags"][word_id])])
            prev = word_id
        all_preds.append(preds)
        all_refs.append(refs)

    return float(seqeval_f1(all_refs, all_preds))


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def run_condition(
    name: str,
    train_langs: List[str],
    eval_langs: List[str],
    cluster_map: Optional[Dict[str, int]],
    cfg: RunConfig,
    device: torch.device,
) -> Dict:
    subdir = os.path.join(cfg.out_dir, name)
    os.makedirs(subdir, exist_ok=True)
    scores = train_and_evaluate_ner(
        train_langs=train_langs,
        eval_langs=eval_langs,
        cfg=cfg,
        device=device,
        save_dir=subdir,
    )
    return {
        "means": scores,
        "train_langs": train_langs,
        "cluster_map": cluster_map or {},
    }


def build_clustering_meta(
    ling_map: Dict[str, int],
    emb_meta: Dict,
    cfg: RunConfig,
) -> Dict:
    return {
        "linguistic": {
            "method": "head_parameter",
            "head_final": sorted(HEAD_FINAL),
            "head_initial": sorted(HEAD_INITIAL),
            "n_clusters": 2,
            "cluster_map": ling_map,
        },
        "embedding": emb_meta,
        "config": {
            "languages": LANGUAGES,
            "target_lang": TARGET_LANG,
            "seed": cfg.seed,
            "max_epochs": cfg.max_epochs,
            "max_train_per_lang": cfg.max_train_per_lang,
            "max_embed_samples_per_lang": cfg.max_embed_samples_per_lang,
            "max_eval_samples_per_lang": cfg.max_eval_samples_per_lang,
            "n_clusters": cfg.n_clusters,
            "quick": cfg.quick,
        },
    }


def aggregate_final_info(all_runs: Dict[str, Dict], n_seeds: int) -> Dict:
    """Aggregate per-seed results into AI Scientist-compatible final_info.json."""
    conditions = [
        "linguistic_clustering",
        "embedding_clustering",
        "per_language",
        "all_mixed",
    ]
    final_info: Dict = {}
    for cond in conditions:
        metric_keys = set()
        for i in range(n_seeds):
            metric_keys.update(all_runs[f"seed_{i}"][cond]["means"].keys())
        means = {}
        stderrs = {}
        for key in sorted(metric_keys):
            vals = [
                all_runs[f"seed_{i}"][cond]["means"][key]
                for i in range(n_seeds)
            ]
            means[key] = float(np.mean(vals))
            stderrs[key] = (
                float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            )
        final_info[cond] = {"means": means, "stderrs": stderrs}
    return final_info


def run_experiment(cfg: RunConfig) -> Dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = get_device()

    ling_map = cluster_linguistic(LANGUAGES)
    emb_map, emb_meta = cluster_embedding(LANGUAGES, cfg, device)
    clustering_meta = build_clustering_meta(ling_map, emb_meta, cfg)

    with open(os.path.join(cfg.out_dir, "cluster_linguistic.json"), "w") as f:
        json.dump(ling_map, f, indent=2)
    with open(os.path.join(cfg.out_dir, "cluster_embedding.json"), "w") as f:
        json.dump(emb_map, f, indent=2)
    with open(os.path.join(cfg.out_dir, "clustering_meta.json"), "w") as f:
        json.dump(clustering_meta, f, indent=2)

    eval_langs = LANGUAGES
    results = {
        "linguistic_clustering": run_condition(
            "linguistic_clustering",
            train_langs=langs_in_cluster(TARGET_LANG, ling_map),
            eval_langs=eval_langs,
            cluster_map=ling_map,
            cfg=cfg,
            device=device,
        ),
        "embedding_clustering": run_condition(
            "embedding_clustering",
            train_langs=langs_in_cluster(TARGET_LANG, emb_map),
            eval_langs=eval_langs,
            cluster_map=emb_map,
            cfg=cfg,
            device=device,
        ),
        "per_language": run_condition(
            "per_language",
            train_langs=[TARGET_LANG],
            eval_langs=[TARGET_LANG],
            cluster_map=None,
            cfg=cfg,
            device=device,
        ),
        "all_mixed": run_condition(
            "all_mixed",
            train_langs=LANGUAGES,
            eval_langs=eval_langs,
            cluster_map=None,
            cfg=cfg,
            device=device,
        ),
        "metadata": {
            "languages": LANGUAGES,
            "target_lang": TARGET_LANG,
            "primary_metric": PRIMARY_METRIC_KEY,
            "model": MODEL_NAME,
            "seed": cfg.seed,
            "max_epochs": cfg.max_epochs,
            "quick": cfg.quick,
        },
    }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="asian_ner minimal PoC experiment")
    p.add_argument("--out_dir", type=str, default="run_0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--max_train_per_lang", type=int, default=500)
    p.add_argument("--max_embed_samples_per_lang", type=int, default=200)
    p.add_argument("--quick", action="store_true", help="Smoke test (1 epoch, tiny data)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = build_config(args)

    seeds = [cfg.seed] if cfg.quick else [cfg.seed, cfg.seed + 1, cfg.seed + 2]
    all_runs: Dict[str, Dict] = {}

    for i, seed in enumerate(seeds):
        run_dir = args.out_dir if len(seeds) == 1 else os.path.join(args.out_dir, f"seed_{i}")
        run_cfg = RunConfig(
            out_dir=run_dir,
            seed=seed,
            max_epochs=cfg.max_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            max_length=cfg.max_length,
            max_train_per_lang=cfg.max_train_per_lang,
            max_embed_samples_per_lang=cfg.max_embed_samples_per_lang,
            max_eval_samples_per_lang=cfg.max_eval_samples_per_lang,
            lang_id_epochs=cfg.lang_id_epochs,
            n_clusters=cfg.n_clusters,
            quick=cfg.quick,
        )
        print(f"=== Running seed {seed} -> {run_dir} ===")
        result = run_experiment(run_cfg)
        with open(os.path.join(run_dir, "final_info.json"), "w") as f:
            json.dump(result, f, indent=2)
        all_runs[f"seed_{i}"] = result

    final_info = aggregate_final_info(all_runs, len(seeds))
    with open(os.path.join(args.out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f, indent=2)

    with open(os.path.join(args.out_dir, "detailed_results.json"), "w") as f:
        json.dump(all_runs, f, indent=2)

    # Promote clustering artifacts from the first seed to the top-level out_dir.
    first_seed_dir = args.out_dir if len(seeds) == 1 else os.path.join(args.out_dir, "seed_0")
    for fname in (
        "clustering_meta.json",
        "cluster_linguistic.json",
        "cluster_embedding.json",
    ):
        src = os.path.join(first_seed_dir, fname)
        if os.path.exists(src):
            with open(src, "r") as f:
                data = json.load(f)
            with open(os.path.join(args.out_dir, fname), "w") as f:
                json.dump(data, f, indent=2)

    print("Done. Wrote final_info.json, detailed_results.json, clustering_meta.json")
