"""TRANSFERFIT - data loading and entity resolution.

Two ID namespaces exist in the raw data:

* Namespace A (parquet universe): players.parquet, player_careers,
  player_season_stats*, player_valuations.parquet, teams.parquet, ...
* Namespace B (ledger universe): club_transfers.csv, player_valuations_full.csv,
  club_financial_metrics.csv, team_full_dossier.csv.

There is no direct ID link between namespaces; entity resolution is done by
normalised player/club name matching.
"""
from __future__ import annotations

import unicodedata
import re
from difflib import SequenceMatcher

import pandas as pd

from src.config import FILES, FILES_B, RAW_DIR


def normalize_text(s: object) -> str:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def name_tokens(s: object) -> set[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return set()
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return set(t for t in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if len(t) > 1)


def load_namespace_a() -> dict[str, pd.DataFrame]:
    """Load all parquet datasets (namespace A)."""
    data = {}
    for name, path in FILES.items():
        if path.exists():
            data[name] = pd.read_parquet(path)
    return data


def load_namespace_b() -> dict[str, pd.DataFrame]:
    """Load the ledger CSV datasets (namespace B)."""
    data = {}
    for name, path in FILES_B.items():
        if path.exists():
            kwargs = {"encoding": "latin-1"} if name == "sd_db" else {}
            data[name] = pd.read_csv(path, **kwargs)
    return data


def load_events_sample():
    """Load the sample events JSON if present."""
    import json
    path = RAW_DIR / "15946 (1).json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    return pd.json_normalize(obj)


def _candidates(players: pd.DataFrame) -> pd.DataFrame:
    df = players.copy()
    df["_short_norm"] = df["short_name"].apply(normalize_text)
    df["_full_norm"] = df.apply(
        lambda r: normalize_text(f"{r['first_name'] or ''} {r['last_name'] or ''}"),
        axis=1,
    )
    return df


def resolve_player_b2a(players_a: pd.DataFrame, names_b: list[str]) -> dict[str, pd.DataFrame]:
    """Resolve namespace-B player names to namespace-A player rows.

    Priority: exact short_name -> exact full name -> token-coverage fuzzy.
    Only unambiguous or strongly supported matches are kept.
    """
    cand = _candidates(players_a)
    by_short = {k: g for k, g in cand.groupby("_short_norm")}
    by_full = {k: g for k, g in cand.groupby("_full_norm")}

    resolved = {}
    ambiguous = {}
    unresolved = []

    for name in names_b:
        n = normalize_text(name)
        if not n:
            unresolved.append(name)
            continue
        if n in by_full and len(by_full[n]) == 1:
            resolved[name] = by_full[n].iloc[0].to_dict()
            continue
        if n in by_short and len(by_short[n]) == 1:
            resolved[name] = by_short[n].iloc[0].to_dict()
            continue

        tokens = name_tokens(name)
        scored = []
        for _, row in cand.iterrows():
            a_tok = name_tokens(row["short_name"]) | name_tokens(
                f"{row['first_name'] or ''} {row['last_name'] or ''}"
            )
            inter = a_tok & tokens
            if not inter:
                continue
            coverage = len(inter) / max(len(tokens), 1)
            seq = max(
                SequenceMatcher(None, n, row["_short_norm"]).ratio(),
                SequenceMatcher(None, n, row["_full_norm"]).ratio(),
            )
            scored.append((coverage, seq, row))
        if not scored:
            unresolved.append(name)
            continue
        best = max(scored, key=lambda t: (t[0], t[1]))
        top = sorted(scored, key=lambda t: (t[0], t[1]), reverse=True)[:2]
        if len(scored) == 1 or top[0][:2] >= (1.0, 0.75) or (top[0][1] - top[1][1]) >= 0.2:
            resolved[name] = best[2].to_dict()
        else:
            ambiguous[name] = [{"player_id": t[2]["player_id"], "name": t[2]["short_name"],
                                "coverage": t[0], "seq": t[1]} for t in top]

    return {"resolved": resolved, "ambiguous": ambiguous, "unresolved": unresolved}


def resolve_team_names(teams_a: pd.DataFrame, names_b: list[str]) -> dict[str, int]:
    """Resolver club names (namespace B) to team_id (namespace A) by exact name after normalisation."""
    mapping = {}
    for n in names_b:
        nn = normalize_text(n)
        hits = teams_a[teams_a["name"].apply(normalize_text) == nn]
        mapping[n] = int(hits.iloc[0]["team_id"]) if len(hits) == 1 else None
    return mapping