"""Fast URIEL family lookup via lang2vec npz (avoids slow get_features API)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pkg_resources

DEFINITIONS_PATH = Path(__file__).parent / "asian_pool_definition.json"


@lru_cache(maxsize=1)
def _family_db() -> tuple[dict[str, int], list[str], np.ndarray, dict[str, str]]:
    letter_codes_path = pkg_resources.resource_filename("lang2vec", "data/letter_codes.json")
    with open(letter_codes_path, encoding="utf-8") as fh:
        letter_codes = json.load(fh)

    npz_path = pkg_resources.resource_filename("lang2vec", "data/family_features.npz")
    data = np.load(npz_path, allow_pickle=True)
    feats = [str(x) for x in data["feats"]]
    langs = [str(x) for x in data["langs"]]
    matrix = data["data"]
    lang_idx = {lang: i for i, lang in enumerate(langs)}
    return lang_idx, feats, matrix, letter_codes


def resolve_iso_code(code: str) -> Optional[str]:
    lang_idx, _, _, letter_codes = _family_db()
    if code in lang_idx:
        return code
    if code in letter_codes and letter_codes[code] in lang_idx:
        return letter_codes[code]
    base = code.split("-")[0]
    if base in lang_idx:
        return base
    if base in letter_codes and letter_codes[base] in lang_idx:
        return letter_codes[base]
    return None


def top_family(code: str) -> Optional[str]:
    lang_idx, feats, matrix, _ = _family_db()
    iso = resolve_iso_code(code)
    if iso is None:
        return None
    vec = matrix[lang_idx[iso], :, 0]
    active = [feats[i].removeprefix("F_") for i, v in enumerate(vec) if v > 0.5]
    return active[0] if active else None


def load_pool_definitions() -> dict:
    return json.loads(DEFINITIONS_PATH.read_text(encoding="utf-8"))


def is_in_named_pool(code: str, pool_name: str) -> tuple[bool, Optional[str]]:
    defs = load_pool_definitions()
    if pool_name not in defs["definitions"]:
        raise KeyError(f"Unknown pool definition: {pool_name}")
    allowed = set(defs["definitions"][pool_name]["allowed_top_families"])
    family = top_family(code)
    if family is None:
        return False, None
    return family in allowed, family


def filter_asian_pool(rows: list[dict], pool_name: str) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        matched, family = is_in_named_pool(row["lang"], pool_name)
        if not matched:
            continue
        enriched = dict(row)
        enriched["family"] = family
        out.append(enriched)
    return sorted(out, key=lambda r: (not r["low_resource"], r["train"], r["lang"]))
