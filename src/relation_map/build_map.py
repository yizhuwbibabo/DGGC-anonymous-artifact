import argparse

import csv

import json

import math

import os

import re

import time

from collections import Counter, defaultdict

from concurrent.futures import ProcessPoolExecutor, as_completed

from dataclasses import dataclass

from difflib import SequenceMatcher

from pathlib import Path

from typing import Any, Dict, Iterable, List, Tuple

try:

    import pandas as pd

except Exception as e:

    raise RuntimeError("This script requires pandas. Install it or use the project environment.") from e

try:

    from tqdm import tqdm

except Exception:

    tqdm = None

STOPWORDS = {

    "the", "a", "an", "of", "for", "to", "in", "on", "by", "and", "or", "with",

    "id", "ids", "no", "num", "number", "code", "name", "type", "date", "value",

    "amount", "total", "desc", "description", "data", "table", "field", "column"

}

TYPE_WORDS = {

    "date": {"date", "time", "year", "month", "day"},

    "money": {"amount", "price", "cost", "payment", "salary", "revenue", "expense", "spend", "spending"},

    "identifier": {"id", "code", "number", "num", "identifier", "ref", "reference"},

    "organization": {"supplier", "vendor", "company", "agency", "department", "body", "entity", "organization"},

    "location": {"city", "state", "country", "county", "address", "postcode", "zip", "location", "region"},

    "person": {"person", "name", "firstname", "lastname", "student", "employee", "customer"},

    "category": {"type", "category", "class", "group", "area", "status"},

}

_DATE_RE = re.compile(r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s*$")

_NUM_RE = re.compile(r"^\s*[-+]?\$?[\d,]+(\.\d+)?%?\s*$")

_ID_RE = re.compile(r"^[A-Za-z]{0,4}[-_ ]?\d{2,}[A-Za-z0-9_-]*$")

def norm_text(x: Any) -> str:

    if x is None:

        return ""

    s = str(x).strip().lower()

    s = re.sub(r"[_\-./\\]+", " ", s)

    s = re.sub(r"[^a-z0-9 ]+", " ", s)

    s = re.sub(r"\s+", " ", s).strip()

    return s

def tokens_of(x: Any, keep_stop: bool = False) -> List[str]:

    s = norm_text(x)

    toks = [t for t in s.split() if t]

    if not keep_stop:

        toks = [t for t in toks if t not in STOPWORDS and len(t) > 1]

    return toks

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:

    sa, sb = set(a), set(b)

    if not sa or not sb:

        return 0.0

    return len(sa & sb) / max(1, len(sa | sb))

def seq_ratio(a: str, b: str) -> float:

    if not a or not b:

        return 0.0

    return SequenceMatcher(None, a, b).ratio()

def safe_read_csv(path: str, nrows: int) -> pd.DataFrame:

    for kwargs in [

        dict(nrows=nrows, low_memory=False),

        dict(nrows=nrows, low_memory=False, encoding="latin1"),

        dict(nrows=nrows, low_memory=False, sep=None, engine="python"),

        dict(nrows=nrows, low_memory=False, encoding="latin1", sep=None, engine="python"),

    ]:

        try:

            df = pd.read_csv(path, **kwargs)

            if df is not None and len(df.columns) > 0:

                return df

        except Exception:

            pass

    return pd.DataFrame()

def infer_col_type(header: str, values: List[str]) -> str:

    htoks = set(tokens_of(header, keep_stop=True))

    for t, words in TYPE_WORDS.items():

        if htoks & words:

            return t

    vals = [str(v).strip() for v in values if str(v).strip() and str(v).strip().lower() not in {"nan", "none", "null"}]

    if not vals:

        return "empty"

    sample = vals[:80]

    date_hits = sum(1 for v in sample if _DATE_RE.match(v))

    num_hits = sum(1 for v in sample if _NUM_RE.match(v))

    id_hits = sum(1 for v in sample if _ID_RE.match(v))

    uniq_ratio = len(set(sample)) / max(1, len(sample))

    if date_hits / len(sample) >= 0.45:

        return "date"

    if num_hits / len(sample) >= 0.70:

        if uniq_ratio > 0.90 and id_hits / len(sample) >= 0.45:

            return "identifier"

        return "numeric"

    if id_hits / len(sample) >= 0.55:

        return "identifier"

    if uniq_ratio <= 0.25:

        return "category"

    return "text"

def type_compat(a: str, b: str) -> float:

    if a == b:

        return 1.0

    compatible = {

        ("money", "numeric"), ("numeric", "money"),

        ("identifier", "text"), ("text", "identifier"),

        ("organization", "text"), ("text", "organization"),

        ("category", "text"), ("text", "category"),

    }

    if (a, b) in compatible:

        return 0.65

    if a == "empty" or b == "empty":

        return 0.1

    return 0.25

def value_token_set(values: List[str], max_tokens: int = 80) -> List[str]:

    cnt = Counter()

    for v in values:

        for t in tokens_of(v):

            if len(t) > 1 and not t.isdigit():

                cnt[t] += 1

    return [t for t, _ in cnt.most_common(max_tokens)]

def value_overlap(a: List[str], b: List[str]) -> float:

    return jaccard(a, b)

def profile_table(path: str, nrows: int, sample_values: int, max_cols: int) -> Dict[str, Any]:

    name = os.path.basename(path)

    df = safe_read_csv(path, nrows=nrows)

    if df.empty or len(df.columns) == 0:

        return {

            "name": name,

            "path": path,

            "nrows": 0,

            "ncols": 0,

            "cols": [],

            "tokens": [],

            "block_tokens": [],

        }

    cols = []

    for c in list(df.columns)[:max_cols]:

        series = df[c].dropna().astype(str).head(sample_values).tolist()

        header = str(c)

        h_toks_keep = tokens_of(header, keep_stop=True)

        h_toks = tokens_of(header, keep_stop=False)

        v_toks = value_token_set(series)

        ctype = infer_col_type(header, series)

        role_hits = []

        hset = set(h_toks_keep)

        for role, words in TYPE_WORDS.items():

            if hset & words:

                role_hits.append(role)

        cols.append({

            "header": header,

            "norm_header": norm_text(header),

            "header_tokens": h_toks,

            "header_tokens_keep": h_toks_keep,

            "value_tokens": v_toks,

            "type": ctype,

            "roles": role_hits,

            "examples": series[:5],

        })

    block = []

    for col in cols:

        block.extend(col["header_tokens"])

        block.append("type:" + col["type"])

        for r in col["roles"]:

            block.append("role:" + r)

        block.extend(["val:" + t for t in col["value_tokens"][:8]])

    tokens = []

    for col in cols:

        tokens.extend(col["header_tokens"])

        tokens.extend(col["value_tokens"][:12])

    return {

        "name": name,

        "path": path,

        "nrows": int(len(df)),

        "ncols": int(len(cols)),

        "cols": cols,

        "tokens": [t for t, _ in Counter(tokens).most_common(200)],

        "block_tokens": [t for t, _ in Counter(block).most_common(120)],

    }

def col_match_score(ca: Dict[str, Any], cb: Dict[str, Any], weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> Dict[str, float]:

    # Eq. (5): A_ij = w^T s(c_i, c'_j) over three normalized signals --

    # header similarity, value-token overlap, and type compatibility -- with

    # weights summing to one.

    hsim = seq_ratio(ca["norm_header"], cb["norm_header"])

    vo = value_overlap(ca["value_tokens"], cb["value_tokens"])

    tc = type_compat(ca["type"], cb["type"])

    combined = weights[0] * hsim + weights[1] * vo + weights[2] * tc

    return {

        "combined": float(combined),

        "header_sim": float(hsim),

        "value_overlap": float(vo),

        "type_compat": float(tc),

    }

def greedy_one_to_one_matches(a: Dict[str, Any], b: Dict[str, Any], min_col_score: float = 0.0, weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> List[Dict[str, Any]]:

    # Greedy one-to-one column alignment: iteratively select the column pair

    # with the highest compatibility (Sec. 3.4). min_col_score defaults to 0,

    # i.e., pure greedy selection as described in the paper.

    pairs = []

    for i, ca in enumerate(a["cols"]):

        for j, cb in enumerate(b["cols"]):

            s = col_match_score(ca, cb, weights)

            if s["combined"] >= min_col_score:

                pairs.append((s["combined"], i, j, s))

    pairs.sort(reverse=True, key=lambda x: x[0])

    used_i, used_j = set(), set()

    out = []

    for _, i, j, s in pairs:

        if i in used_i or j in used_j:

            continue

        used_i.add(i)

        used_j.add(j)

        ca, cb = a["cols"][i], b["cols"][j]

        out.append({

            "left": ca["header"],

            "right": cb["header"],

            "left_type": ca["type"],

            "right_type": cb["type"],

            **s,

        })

    return out

def table_pair_score(a: Dict[str, Any], b: Dict[str, Any], gamma: float, weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)) -> Tuple[float, Dict[str, Any]]:

    """Eqs. (6)-(7): M(t,t') is the average one-to-one alignment strength

    over the anchor schema and f(t,t') = M(t,t') * I[M(t,t') > gamma]."""

    if a["ncols"] == 0 or b["ncols"] == 0:

        return 0.0, {}

    matches = greedy_one_to_one_matches(a, b, min_col_score=0.0, weights=weights)

    if not matches:

        return 0.0, {}

    n_anchor = max(1, a["ncols"])

    M = sum(x["combined"] for x in matches) / n_anchor

    f = M if M > gamma else 0.0

    evidence = {

        "matched_columns": matches,

        "matched_count": len(matches),

        "left_columns": a["ncols"],

        "right_columns": b["ncols"],

        "alignment_strength": float(M),

        "avg_match": float(M),

        "semantic_score": float(f),

        "score": float(f),

        "confidence": float(f),

    }

    return float(f), evidence

_G_PROFILES: List[Dict[str, Any]] = []

_G_NAMES: List[str] = []

_G_TAU: float = 0.65

_G_GAMMA: float = 0.10

_G_WEIGHTS: Tuple[float, float, float] = (0.4, 0.3, 0.3)



def _init_worker(profiles, tau, gamma, weights):

    global _G_PROFILES, _G_NAMES, _G_TAU, _G_GAMMA, _G_WEIGHTS

    _G_PROFILES = profiles

    _G_NAMES = [p["name"] for p in profiles]

    _G_TAU = tau

    _G_GAMMA = gamma

    _G_WEIGHTS = weights



def _score_anchor(args):

    i, cand_ids = args

    a = _G_PROFILES[i]

    out = []

    for j in cand_ids:

        if i == j:

            continue

        b = _G_PROFILES[j]

        # Eqs. (6)-(7): f(t,t') = M(t,t') * I[M(t,t') > gamma]; a pair is

        # retained only if f(t,t') > tau (top-K capping happens at the end).

        score, ev = table_pair_score(a, b, gamma=_G_GAMMA, weights=_G_WEIGHTS)

        if not ev:

            continue

        if score <= _G_TAU:

            continue

        ai, bj = _G_NAMES[i], _G_NAMES[j]

        entry_ab = {

            "target": bj,

            "table": bj,

            "neighbor": bj,

            "score": score,

            "semantic_score": score,

            "confidence": ev["confidence"],

            **ev,

        }

        entry_ba = {

            "target": ai,

            "table": ai,

            "neighbor": ai,

            "score": score,

            "semantic_score": score,

            "confidence": ev["confidence"],

            **{k: v for k, v in ev.items() if k != "matched_columns"},

            "matched_columns": [

                {

                    **{kk: vv for kk, vv in m.items() if kk not in {"left", "right", "left_type", "right_type"}},

                    "left": m["right"],

                    "right": m["left"],

                    "left_type": m["right_type"],

                    "right_type": m["left_type"],

                }

                for m in ev["matched_columns"]

            ],

        }

        out.append((ai, bj, entry_ab, entry_ba))

    return out

def build_candidates(

    profiles: List[Dict[str, Any]],

    max_block_freq: int,

    min_shared_block: int,

    max_candidates_per_table: int,

) -> List[List[int]]:

    inv = defaultdict(list)

    for i, p in enumerate(profiles):

        toks = set(p["block_tokens"])

        for t in toks:

            inv[t].append(i)

    inv = {t: ids for t, ids in inv.items() if 1 < len(ids) <= max_block_freq}

    candidates: List[List[int]] = []

    for i, p in enumerate(profiles):

        cnt = Counter()

        for t in set(p["block_tokens"]):

            ids = inv.get(t)

            if not ids:

                continue

            for j in ids:

                if j != i:

                    cnt[j] += 1

        cand = [(j, c) for j, c in cnt.items() if c >= min_shared_block]

        cand.sort(key=lambda x: x[1], reverse=True)

        cand_ids = [j for j, _ in cand[:max_candidates_per_table]]

        candidates.append(cand_ids)

    return candidates

def maybe_iter(x, total=None, desc=None):

    if tqdm is not None:

        return tqdm(x, total=total, desc=desc)

    return x

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", required=True)

    ap.add_argument("--output_file", required=True)

    ap.add_argument("--nrows", type=int, default=300)

    ap.add_argument("--sample_values", type=int, default=80)

    ap.add_argument("--max_cols", type=int, default=12)

    ap.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 2))

    ap.add_argument("--tau", type=float, default=0.65, help="Minimum filtering score f(t,t') for a retained pair")

    ap.add_argument("--gamma", type=float, default=0.10, help="Minimum alignment strength M(t,t')")

    ap.add_argument("--col_weights", default="0.4,0.3,0.3", help="Weights (header, value, type) for Eq. (5); must sum to 1")

    ap.add_argument("--min_shared_block", type=int, default=1)

    ap.add_argument("--max_block_freq", type=int, default=500)

    ap.add_argument("--max_candidates_per_table", type=int, default=500)

    ap.add_argument("--K", type=int, default=16, help="Retained-neighbor width (top-K candidates per anchor)")

    ap.add_argument("--profile_cache", default=None)

    ap.add_argument("--candidate_cache", default=None)

    args = ap.parse_args()

    weights = tuple(float(x) for x in args.col_weights.split(","))

    t0 = time.time()

    data_dir = Path(args.data_dir)

    files = sorted(str(p) for p in data_dir.glob("*.csv"))

    if not files:

        raise FileNotFoundError(f"No CSV files found under {args.data_dir}")

    out_path = Path(args.output_file)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile_cache = Path(args.profile_cache) if args.profile_cache else out_path.with_suffix(".profiles.json")

    candidate_cache = Path(args.candidate_cache) if args.candidate_cache else out_path.with_suffix(".candidates.json")

    print("=" * 60)

    print("Parallel relation-map construction")

    print("=" * 60)

    print(f"data_dir                  : {args.data_dir}")

    print(f"tables                    : {len(files)}")

    print(f"output_file               : {args.output_file}")

    print(f"profile_cache             : {profile_cache}")

    print(f"candidate_cache           : {candidate_cache}")

    print(f"workers                   : {args.workers}")

    print(f"nrows/sample_values/max_cols: {args.nrows}/{args.sample_values}/{args.max_cols}")

    print(f"tau                       : {args.tau}")

    print(f"gamma                     : {args.gamma}")

    print(f"col_weights               : {args.col_weights}")

    print(f"min_avg_match_score       : {args.min_avg_match_score}")

    print(f"blocking max_freq/candidates: {args.max_block_freq}/{args.max_candidates_per_table}")

    print("=" * 60)

    if profile_cache.exists():

        print(f"Loading profile cache: {profile_cache}")

        profiles = json.loads(profile_cache.read_text(encoding="utf-8"))

    else:

        print("Profiling tables...")

        profiles = []

        for fp in maybe_iter(files, total=len(files), desc="Profiling"):

            profiles.append(profile_table(fp, args.nrows, args.sample_values, args.max_cols))

        profile_cache.write_text(json.dumps(profiles, ensure_ascii=False), encoding="utf-8")

        print(f"Saved profile cache: {profile_cache}")

    good = sum(1 for p in profiles if p["ncols"] > 0)

    print(f"Profiles loaded: {len(profiles)}, non-empty: {good}")

    if candidate_cache.exists():

        print(f"Loading candidate cache: {candidate_cache}")

        candidates = json.loads(candidate_cache.read_text(encoding="utf-8"))

    else:

        print("Building blocking candidates...")

        candidates = build_candidates(

            profiles,

            max_block_freq=args.max_block_freq,

            min_shared_block=args.min_shared_block,

            max_candidates_per_table=args.max_candidates_per_table,

        )

        candidate_cache.write_text(json.dumps(candidates), encoding="utf-8")

        print(f"Saved candidate cache: {candidate_cache}")

    total_candidate_pairs = sum(len(x) for x in candidates)

    print(f"Candidate comparisons after blocking: {total_candidate_pairs}")

    print("Scoring candidate pairs in parallel...")

    relation_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    tasks = [(i, candidates[i]) for i in range(len(profiles)) if candidates[i]]

    done = 0

    edge_count = 0

    with ProcessPoolExecutor(

        max_workers=args.workers,

        initializer=_init_worker,

        initargs=(profiles, args.tau, args.gamma, weights),

    ) as ex:

        futs = [ex.submit(_score_anchor, t) for t in tasks]

        iterator = as_completed(futs)

        if tqdm is not None:

            iterator = tqdm(iterator, total=len(futs), desc="Scoring anchors")

        for fut in iterator:

            rows = fut.result()

            done += 1

            for ai, bj, entry_ab, entry_ba in rows:

                relation_map[ai].append(entry_ab)

                relation_map[bj].append(entry_ba)

                edge_count += 2

            if tqdm is None and done % 100 == 0:

                print(f"  processed anchors={done}/{len(tasks)}, directed_edges={edge_count}")

    final = {}

    for k, vals in relation_map.items():

        best = {}

        for e in vals:

            tgt = e["target"]

            if tgt not in best or e.get("semantic_score", 0.0) > best[tgt].get("semantic_score", 0.0):

                best[tgt] = e

        ordered = sorted(best.values(), key=lambda x: x.get("semantic_score", x.get("score", 0.0)), reverse=True)

        final[k] = ordered[: args.K]

    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    undirected_est = edge_count // 2

    print("=" * 60)

    print("Relation map finished")

    print("=" * 60)

    print(f"anchors with positives : {len(final)}")

    print(f"directed edges kept    : {sum(len(v) for v in final.values())}")

    print(f"undirected edge estimate before cap: {undirected_est}")

    print(f"output_file            : {out_path}")

    print(f"elapsed seconds        : {time.time() - t0:.1f}")

    print("=" * 60)

if __name__ == "__main__":

    main()
