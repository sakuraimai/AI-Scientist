#!/usr/bin/env python3
"""Scan WikiAnn language configs and propose rule-based language pools."""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import get_dataset_config_info, get_dataset_config_names, load_dataset_builder

from language_family import filter_asian_pool, load_pool_definitions

WIKIANN = "unimelb-nlp/wikiann"
N_MIN = 50
N_MAX = 50_000
T_LOW = 500
P_MAX = 30
HIGH_SAMPLE_CAP = 10
SEED = 1
REQUIRE_LANGS = ["mn"]
MAX_WORKERS = 1
MAX_RETRIES = 4
RETRY_SLEEP_S = 75


def count_splits(lang: str) -> dict:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            info = get_dataset_config_info(WIKIANN, lang, trust_remote_code=True)
            train_n = info.splits["train"].num_examples
            test_n = info.splits["test"].num_examples
            return {
                "lang": lang,
                "train": train_n,
                "test": test_n,
                "low_resource": train_n <= T_LOW,
                "in_range": N_MIN <= train_n <= N_MAX,
            }
        except Exception as exc:
            last_exc = exc
            if "429 Too Many Requests" in str(exc) and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP_S)
                continue
            break

    try:
        builder = load_dataset_builder(WIKIANN, lang, trust_remote_code=True)
        builder.download_and_prepare(download_mode="reuse_cache_if_exists")
        train_n = builder.info.splits["train"].num_examples
        test_n = builder.info.splits["test"].num_examples
        return {
            "lang": lang,
            "train": train_n,
            "test": test_n,
            "low_resource": train_n <= T_LOW,
            "in_range": N_MIN <= train_n <= N_MAX,
        }
    except Exception as exc:
        last_exc = exc

    return {"lang": lang, "error": str(last_exc)}


def load_cache(cache_path: Path) -> dict[str, dict]:
    if not cache_path.exists():
        return {}
    rows: dict[str, dict] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[row["lang"]] = row
    return rows


def append_cache(cache_path: Path, row: dict) -> None:
    with cache_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def get_all_configs(cache_path: Path) -> list[str]:
    cached = load_cache(cache_path)
    try:
        configs = sorted(get_dataset_config_names(WIKIANN))
        if len(configs) >= 100:
            return configs
    except Exception:
        configs = []
    if cached:
        return sorted(cached)
    if configs:
        return configs
    raise RuntimeError("Could not discover WikiAnn configs from Hub or cache")


def scan_configs(configs: list[str], cache_path: Path) -> list[dict]:
    cached = load_cache(cache_path)
    rows: list[dict] = []
    pending = [
        lang
        for lang in configs
        if lang not in cached or "error" in cached[lang]
    ]

    for lang in configs:
        if lang in cached and "error" not in cached[lang]:
            rows.append(cached[lang])

    if not pending:
        return rows

    done = 0
    total = len(pending)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(count_splits, lang): lang for lang in pending}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if "error" not in row:
                append_cache(cache_path, row)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"Scanned {done}/{total} pending configs", flush=True)

    rows.sort(key=lambda r: r["lang"])
    return rows


def build_pool(ok: list[dict]) -> list[dict]:
    low = [r for r in ok if r["low_resource"]]
    high = [r for r in ok if not r["low_resource"]]

    rng = random.Random(SEED)
    high_k = min(len(high), HIGH_SAMPLE_CAP)
    pool_by_lang = {r["lang"]: r for r in rng.sample(high, high_k)} if high_k else {}

    for lang in REQUIRE_LANGS:
        row = next((r for r in ok if r["lang"] == lang), None)
        if row:
            pool_by_lang[lang] = row

    remaining = max(0, P_MAX - len(pool_by_lang))
    low_candidates = [r for r in low if r["lang"] not in pool_by_lang]
    low_k = min(len(low_candidates), remaining)
    for row in rng.sample(low_candidates, low_k):
        pool_by_lang[row["lang"]] = row

    return sorted(
        pool_by_lang.values(),
        key=lambda r: (not r["low_resource"], r["train"], r["lang"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan WikiAnn and build language pools")
    parser.add_argument(
        "--pool",
        default="sampled",
        choices=["sampled", "asian_east_sea", "asian_broad", "asian_core"],
        help="Pool rule: sampled=P_MAX mix; asian_*=full family union from asian_pool_definition.json",
    )
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    cache_path = out_dir / "wikiann_pool_scan_cache.jsonl"
    configs = get_all_configs(cache_path)
    rows = scan_configs(configs, cache_path)

    ok = [r for r in rows if "train" in r and r["in_range"]]
    low = [r for r in ok if r["low_resource"]]
    high = [r for r in ok if not r["low_resource"]]
    pool = (
        filter_asian_pool(ok, args.pool)
        if args.pool != "sampled"
        else build_pool(ok)
    )

    pool_rules = {
        "n_min": N_MIN,
        "n_max": N_MAX,
        "t_low": T_LOW,
        "pool_mode": args.pool,
    }
    if args.pool == "sampled":
        pool_rules.update(
            {
                "p_max": P_MAX,
                "high_sample_cap": HIGH_SAMPLE_CAP,
                "seed": SEED,
                "require_langs": REQUIRE_LANGS,
            }
        )
    else:
        pool_rules["definition"] = load_pool_definitions()["definitions"][args.pool]

    out = {
        "dataset": WIKIANN,
        "rules": pool_rules,
        "summary": {
            "total_configs": len(configs),
            "loaded_ok": len(ok),
            "load_failed": len([r for r in rows if "error" in r]),
            "low_resource_count": len(low),
            "high_resource_in_range": len(high),
            "proposed_pool_size": len(pool),
            "pool_low_resource": sum(1 for r in pool if r["low_resource"]),
            "pool_high_resource": sum(1 for r in pool if not r["low_resource"]),
        },
        "current_five": [
            next((r for r in ok if r["lang"] == lang), {"lang": lang, "missing": True})
            for lang in ["ja", "ko", "mn", "ru", "en"]
        ],
        "proposed_pool": pool,
        "all_in_range": ok,
    }

    out_path = out_dir / "wikiann_pool_scan.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== WikiAnn scan summary ===")
    print(json.dumps(out["summary"], indent=2))
    print("\n=== Current 5 languages ===")
    for row in out["current_five"]:
        print(row)
    print(f"\n=== Proposed pool ({len(pool)} langs) ===")
    for row in pool:
        fam = f" fam={row['family']}" if "family" in row else ""
        print(
            f"  {row['lang']:6s} train={row['train']:6d} test={row['test']:4d} "
            f"low={row['low_resource']}{fam}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
