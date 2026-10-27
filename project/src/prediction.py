"""TRANSFERFIT - prediction interface.

Given a player, source club, target club and evaluation date, estimate
the adaptation probability at the target club using ONLY information that
existed before the evaluation date. Missing inputs are reported explicitly;
nothing is fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data_loading import (
    load_namespace_a,
    load_namespace_b,
    resolve_player_b2a,
    normalize_text,
)
from src.feature_engineering import (
    profile_features,
    pre_features,
    club_mv_stats,
    squad_minutes_share,
    target_role_share,
)
from src.model import build_X, load_artifact
from src.preprocessing import build_stint_panel
from src.similarity import compute_peer_features, find_similar_players


def resolve_player(
    players: pd.DataFrame,
    name: str
) -> pd.Series | None:

    n = normalize_text(name)

    exact = players[
        (
            players["short_name"]
            .fillna("")
            .apply(normalize_text)
            == n
        )
        |
        (
            players["first_name"]
            .fillna("")
            .apply(normalize_text)
            +
            players["last_name"]
            .fillna("")
            .apply(normalize_text)
            == n
        )
    ]

    if len(exact) == 1:
        return exact.iloc[0]

    # Namespace B bridge for transfer-ledger names
    b = load_namespace_b()

    names_b = list(
        b["club_transfers"]["player_name"].dropna().unique()
    )

    if name in names_b:
        bridge = resolve_player_b2a(players, [name])

        if name in bridge["resolved"]:
            row = bridge["resolved"][name]

            matched = players[
                players["player_id"] == row["player_id"]
            ]

            if not matched.empty:
                return matched.iloc[0]

    return None


def build_query_row(
    stint,
    players,
    teams,
    valuation,
    player,
    source_team,
    target_team,
    eval_date,
) -> tuple[pd.Series | None, list[str]]:

    pid = int(player["player_id"])
    move_ts = pd.Timestamp(eval_date)

    missing = []

    # ---------------------------------------------------------
    # Find completed seasons before the evaluation date
    # ---------------------------------------------------------

    season_dates = (
        stint[["super_season_id", "end_date"]]
        .drop_duplicates()
    )

    season_dates["end_date"] = pd.to_datetime(
        season_dates["end_date"]
    )

    cand_seasons = [
        int(s)
        for s in season_dates.loc[
            season_dates["end_date"] <= move_ts,
            "super_season_id",
        ].sort_values(ascending=False)
    ]

    # ---------------------------------------------------------
    # Find player's latest valid pre-transfer stint
    # ---------------------------------------------------------

    pre = None
    s_pre = None

    if not cand_seasons:

        missing.append(
            "no completed pre-transfer season in coverage"
        )

        # IMPORTANT:
        # There is no valid pre-transfer season, so we MUST stop.
        # Otherwise later code tries to access pre["super_season_id"].
        return None, missing

    else:

        # Prefer the player's stint at the source club.
        # If unavailable, use the most recent available stint.
        # No data is fabricated.
        for s in cand_seasons:

            pre_stints = stint[
                (stint["player_id"] == pid)
                &
                (stint["super_season_id"] == s)
            ]

            if pre_stints.empty:
                continue

            # Prefer source-club performance
            if source_team:

                at_source = pre_stints[
                    pre_stints["team_id"] == source_team
                ]

                if not at_source.empty:

                    pre = (
                        at_source
                        .sort_values("minutes_played")
                        .iloc[-1]
                    )

                    s_pre = s
                    break

            # If source club is not supplied,
            # use player's most recent stint.
            else:

                pre = (
                    pre_stints
                    .sort_values("minutes_played")
                    .iloc[-1]
                )

                s_pre = s
                break

        # -----------------------------------------------------
        # Player has no valid pre-transfer performance record
        # -----------------------------------------------------

        if pre is None:

            missing.append(
                "no performance on record in any completed season "
                "before the evaluation date"
            )

            # IMPORTANT:
            # Do not continue to code below because it requires
            # pre["super_season_id"] and other pre-transfer fields.
            return None, missing

        # -----------------------------------------------------
        # Report if latest available season is older than the
        # latest completed season in the dataset.
        # -----------------------------------------------------

        elif s_pre != cand_seasons[0]:

            missing.append(
                f"latest season {cand_seasons[0]-1}/"
                f"{cand_seasons[0]} unavailable; "
                "using most recent available pre-transfer season"
            )

    # At this point PRE is guaranteed to exist.
    # ---------------------------------------------------------
    # Base player information
    # ---------------------------------------------------------

    row = {
        "player_id": pid,

        "player_name": player["short_name"],

        "source_team_id": source_team,

        "target_team_id": target_team,

        "position_group": (
            player["role_name"]
            if pd.notna(player["role_name"])
            else None
        ),

        "foot": player["foot"],

        "height_cm": player["height_cm"],

        "weight_kg": player["weight_kg"],
    }

    # ---------------------------------------------------------
    # Player profile features
    # ---------------------------------------------------------

    row.update(
        profile_features(
            players,
            pid,
            move_ts,
        )
    )

    # ---------------------------------------------------------
    # Pre-transfer performance features
    # ---------------------------------------------------------

    row.update(
        pre_features(pre)
    )

    # ---------------------------------------------------------
    # Club market-value features
    # ---------------------------------------------------------

    row["source_squad_mv"] = (
        club_mv_stats(
            valuation,
            source_team,
            move_ts,
        )
        if source_team
        else np.nan
    )

    row["target_squad_mv"] = (
        club_mv_stats(
            valuation,
            target_team,
            move_ts,
        )
        if target_team
        else np.nan
    )

    if (
        pd.notna(row.get("target_squad_mv"))
        and pd.notna(row.get("source_squad_mv"))
    ):
        row["squad_mv_diff"] = (
            row["target_squad_mv"]
            - row["source_squad_mv"]
        )
    else:
        row["squad_mv_diff"] = np.nan

    # ---------------------------------------------------------
    # Source-club squad minutes share
    # ---------------------------------------------------------

    if source_team:

        row["source_minutes_share"] = squad_minutes_share(
            stint,
            pid,
            source_team,
            int(pre["super_season_id"]),
        )

    else:

        row["source_minutes_share"] = np.nan

    # ---------------------------------------------------------
    # Target-club role share
    # ---------------------------------------------------------

    if target_team:

        row["target_role_share"] = target_role_share(
            stint,
            players,
            row["position_group"],
            target_team,
            int(pre["super_season_id"]),
        )

    else:

        row["target_role_share"] = np.nan

    # ---------------------------------------------------------
    # Pre-transfer market value
    # ---------------------------------------------------------

    mv = valuation[
        (valuation["player_id"] == pid)
        &
        (
            pd.to_datetime(
                valuation["valuation_date"]
            )
            <= move_ts
        )
    ]

    if not mv.empty:

        row["pre_market_value"] = float(
            mv.sort_values("valuation_date")
            .iloc[-1]["market_value"]
        )

    else:

        row["pre_market_value"] = np.nan

    if pd.isna(row["pre_market_value"]):

        missing.append(
            "no pre-transfer market value on record"
        )

    # ---------------------------------------------------------
    # Club wins proxies
    # Only information available in the selected
    # pre-transfer season is used.
    # ---------------------------------------------------------

    pre_season_id = int(
        pre["super_season_id"]
    )

    if source_team:

        seg_src = stint[
            (stint["team_id"] == source_team)
            &
            (
                stint["super_season_id"]
                == pre_season_id
            )
        ]

    else:

        seg_src = None

    if target_team:

        seg_tgt = stint[
            (stint["team_id"] == target_team)
            &
            (
                stint["super_season_id"]
                == pre_season_id
            )
        ]

    else:

        seg_tgt = None

    # Source club wins
    if seg_src is not None and len(seg_src):

        row["source_club_wins_pre"] = float(
            seg_src["wins"].sum()
        )

    else:

        row["source_club_wins_pre"] = np.nan

    # Target club wins
    if seg_tgt is not None and len(seg_tgt):

        row["target_club_wins_pre"] = float(
            seg_tgt["wins"].sum()
        )

    else:

        row["target_club_wins_pre"] = np.nan

    # Difference
    if (
        pd.notna(row.get("target_club_wins_pre"))
        and pd.notna(row.get("source_club_wins_pre"))
    ):

        row["club_wins_diff"] = (
            row["target_club_wins_pre"]
            - row["source_club_wins_pre"]
        )

    else:

        row["club_wins_diff"] = np.nan

    return (
        pd.Series(row, dtype=object),
        missing,
    )


def predict_transfer(
    player_name: str,
    source_team: int | None,
    target_team: int | None,
    eval_date: str,
    context: dict | None = None,
) -> dict:

    """Predict adaptation of a player to a target club on a given date.

    context:
        Pre-built datasets containing:
        stint,
        players,
        teams,
        valuation,
        historical pool with labels,
        artifact.

    Built lazily when not supplied.
    """

    # ---------------------------------------------------------
    # Load context
    # ---------------------------------------------------------

    if context is None:
        context = default_context()

    players = context["players"]
    teams = context["teams"]
    stint = context["stint"]
    valuation = context["valuation"]

    historical = context["historical"]
    artifact = context["artifact"]

    # ---------------------------------------------------------
    # Resolve player
    # ---------------------------------------------------------

    player = resolve_player(
        players,
        player_name,
    )

    result = {
        "player_query": player_name,

        "player_resolved": player is not None,

        "source_team": source_team,

        "target_team": target_team,

        "eval_date": eval_date,

        "missing_data": [],
    }

    # Player not found
    if player is None:

        result["missing_data"].append(
            "player not found in the LaLiga "
            "2022/23-2024/25 roster"
        )

        result["status"] = "blocked"

        return result

    # Target club required
    if target_team is None:

        result["missing_data"].append(
            "target club required"
        )

        result["status"] = "blocked"

        return result

    # ---------------------------------------------------------
    # Resolve club names
    # ---------------------------------------------------------

    if (
        source_team
        and (teams["team_id"] == source_team).any()
    ):

        src_name = teams.loc[
            teams["team_id"] == source_team,
            "name",
        ].iloc[0]

    else:

        src_name = "unknown"

    if (
        (teams["team_id"] == target_team).any()
    ):

        tgt_name = teams.loc[
            teams["team_id"] == target_team,
            "name",
        ].iloc[0]

    else:

        tgt_name = "unknown"

    result["source_team_name"] = src_name
    result["target_team_name"] = tgt_name

    # ---------------------------------------------------------
    # Build query row
    # ---------------------------------------------------------

    query_row, missing = build_query_row(
        stint,
        players,
        teams,
        valuation,
        player,
        source_team,
        target_team,
        eval_date,
    )

    result["missing_data"] += missing

    # ---------------------------------------------------------
    # Missing required information
    # ---------------------------------------------------------

    if query_row is None:

        result["status"] = "blocked"

        return result

    # ---------------------------------------------------------
    # Historical peer features
    #
    # Historical pool is labelled.
    # Query row remains unlabelled.
    # ---------------------------------------------------------

    pool = historical[
        historical["label"].notna()
    ].copy()

    _pool_tr, qdf = compute_peer_features(
        pool,
        query_row.to_frame().T,
    )

    qdf = qdf.iloc[0]

    # Peer adaptation rate
    if "peer_adaptation_rate" in qdf.index:

        result["peer_adaptation_rate"] = float(
            qdf["peer_adaptation_rate"]
        )

    else:

        result["peer_adaptation_rate"] = np.nan

    # Peer average post-transfer GA90
    if "peer_avg_post_ga90" in qdf.index:

        result["peer_avg_post_ga90"] = float(
            qdf["peer_avg_post_ga90"]
        )

    else:

        result["peer_avg_post_ga90"] = np.nan

    # ---------------------------------------------------------
    # Similar historical transfers
    # ---------------------------------------------------------

    sim = find_similar_players(
        query_row,
        pool,
        k=6,
    )

    if not sim.empty:

        result["similar_transfers"] = sim[
            [
                "player_name",
                "source_team_name",
                "target_team_name",
                "post_minutes",
                "label",
            ]
        ].to_dict("records")

    else:

        result["similar_transfers"] = []

    # ---------------------------------------------------------
    # Build model input
    # ---------------------------------------------------------

    X = build_X(
        query_row.to_frame().T,
        artifact["variant"],
    )

    # Add peer features
    X["peer_adaptation_rate"] = result.get(
        "peer_adaptation_rate",
        np.nan,
    )

    X["peer_avg_post_ga90"] = result.get(
        "peer_avg_post_ga90",
        np.nan,
    )

    # Match exact model feature order
    X = X.reindex(
        columns=artifact["feature_columns"],
        fill_value=0.0,
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------

    pipe = artifact["pipeline"]

    proba = float(
        pipe.predict_proba(X)[:, 1][0]
    )

    score = round(
        proba * 100,
        1,
    )

    result["probability"] = round(
        proba,
        4,
    )

    result["adaptation_score"] = score

    result["status"] = "ok"

    # ---------------------------------------------------------
    # Explain prediction
    # ---------------------------------------------------------

    est = pipe.named_steps["model"]

    if hasattr(est, "coef_"):

        Xt = pipe[:-1].transform(X)

        contribs = (
            est.coef_[0]
            * Xt[0]
        )

        pos = sorted(
            zip(X.columns, contribs),
            key=lambda t: -t[1],
        )[:4]

        neg = sorted(
            zip(X.columns, contribs),
            key=lambda t: t[1],
        )[:4]

        result["positive_factors"] = [
            {
                "feature": k,
                "contribution": round(
                    float(v),
                    3,
                ),
            }
            for k, v in pos
            if v > 0
        ]

        result["risk_factors"] = [
            {
                "feature": k,
                "contribution": round(
                    float(v),
                    3,
                ),
            }
            for k, v in neg
            if v < 0
        ]

    else:

        result["positive_factors"] = []
        result["risk_factors"] = []

    # ---------------------------------------------------------
    # Interpretation
    # ---------------------------------------------------------

    result["interpretation"] = interpret(
        score
    )

    # ---------------------------------------------------------
    # Human-readable feature explanation
    # ---------------------------------------------------------

    result["features_explained"] = {

        "pre_minutes": _g(
            query_row,
            "pre_minutes",
        ),

        "pre_ga90": _g(
            query_row,
            "pre_ga90",
        ),

        "age_at_transfer": _g(
            query_row,
            "age_at_transfer",
        ),

        "position_group": _g(
            query_row,
            "position_group",
        ),

        "source_club_wins_pre": _g(
            query_row,
            "source_club_wins_pre",
        ),

        "target_club_wins_pre": _g(
            query_row,
            "target_club_wins_pre",
        ),

        "squad_mv_diff": _g(
            query_row,
            "squad_mv_diff",
        ),

        "pre_market_value": _g(
            query_row,
            "pre_market_value",
        ),
    }

    return result


def _g(
    row: pd.Series,
    key: str,
):
    """Safely retrieve a feature for display."""

    v = row.get(key)

    if v is None:
        return None

    if isinstance(v, (float, np.floating)) and pd.isna(v):
        return None

    if isinstance(
        v,
        (
            int,
            float,
            np.number,
        ),
    ):

        return float(v)

    return v


def interpret(score: float) -> str:

    if score >= 75:

        return (
            "High predicted adaptation: similar players who "
            "moved to comparable clubs established themselves. "
            "Transfer risk is comparatively low."
        )

    if score >= 55:

        return (
            "Moderate-to-good predicted adaptation: favourable "
            "profile, but the target environment still carries "
            "uncertainty."
        )

    if score >= 40:

        return (
            "Ambiguous predicted adaptation: mixed signals from "
            "comparable transitions."
        )

    return (
        "Elevated adaptation risk: analogues with this "
        "pre-transfer profile struggled at comparable clubs."
    )


# =============================================================
# Context cache
# =============================================================

_cache: dict | None = None


def default_context() -> dict:

    global _cache

    if _cache is not None:
        return _cache

    # ---------------------------------------------------------
    # Load namespace A
    # ---------------------------------------------------------

    a = load_namespace_a()

    # ---------------------------------------------------------
    # Build player/team/season stint panel
    # ---------------------------------------------------------

    stint = build_stint_panel(a)

    # ---------------------------------------------------------
    # Historical transfers
    # ---------------------------------------------------------

    historical = pd.read_csv(
        config.DATA_PROCESSED
        / "historical_transfers.csv",
        parse_dates=["transfer_date"],
    )

    from src.adaptation_label import apply_label

    historical = apply_label(
        historical
    )

    # ---------------------------------------------------------
    # Load trained model
    # ---------------------------------------------------------

    artifact = load_artifact(
        "transferfit_model.joblib"
    )

    # ---------------------------------------------------------
    # Assemble context
    # ---------------------------------------------------------

    context = {

        "players": a["players"],

        "teams": a["teams"],

        "stint": stint,

        "valuation": a["player_valuations"],

        "historical": historical,

        "artifact": artifact,
    }

    _cache = context

    return context