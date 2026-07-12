"""Resolve experiment language pool: asian family union + mn hypothesis controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from language_family import filter_asian_pool, load_pool_definitions

TEMPLATE_DIR = Path(__file__).parent
SCAN_PATH = TEMPLATE_DIR / "wikiann_pool_scan.json"
DEFINITIONS_PATH = TEMPLATE_DIR / "asian_pool_definition.json"


def _load_scan_rows() -> list[dict]:
    if not SCAN_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SCAN_PATH}. Run: python scan_wikiann_pool.py --pool asian_east_sea"
        )
    scan = json.loads(SCAN_PATH.read_text(encoding="utf-8"))
    return [r for r in scan["all_in_range"] if "train" in r]


def resolve_experiment_languages(
    pool_name: Optional[str] = None,
    controls: Optional[List[str]] = None,
) -> Dict:
    defs = load_pool_definitions()
    pool_name = pool_name or defs.get("active_experiment_pool", defs["recommended"])
    controls = list(controls if controls is not None else defs.get("mn_hypothesis_controls", []))

    rows = _load_scan_rows()
    row_by_lang = {r["lang"]: r for r in rows}

    scan = json.loads(SCAN_PATH.read_text(encoding="utf-8"))
    scan_pool_mode = scan.get("rules", {}).get("pool_mode")
    if scan_pool_mode == pool_name and scan.get("proposed_pool"):
        base_rows = [dict(r) for r in scan["proposed_pool"]]
    else:
        base_rows = filter_asian_pool(rows, pool_name)

    base_langs = [r["lang"] for r in base_rows]
    missing_controls = [lang for lang in controls if lang not in row_by_lang]
    if missing_controls:
        raise ValueError(f"Hypothesis controls missing from WikiAnn scan: {missing_controls}")

    languages: List[str] = []
    seen = set()
    for lang in base_langs + controls:
        if lang in seen:
            continue
        seen.add(lang)
        languages.append(lang)

    train_counts = {lang: int(row_by_lang[lang]["train"]) for lang in languages}
    t_low = int(defs.get("low_resource_threshold", 500))
    low_resource_langs = [lang for lang in languages if train_counts[lang] <= t_low]

    return {
        "pool_name": pool_name,
        "hypothesis_controls": controls,
        "languages": languages,
        "n_languages": len(languages),
        "train_counts": train_counts,
        "low_resource_threshold": t_low,
        "low_resource_langs": low_resource_langs,
        "family_rows": {
            r["lang"]: r.get("family")
            for r in base_rows
            if r["lang"] in languages
        },
    }
