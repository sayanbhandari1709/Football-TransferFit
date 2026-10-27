"""TRANSFERFIT - feature engineering and historical transfer dataset assembly.

One row = one historical transfer (pre-season S_pre -> post-season S_post).

No-leakage contract:
* input/pre-transfer features are built ONLY from stints strictly before the
  move and valuations dated before the move;
* outcome/target features are built ONLY from the stint at the target club
  AFTER the move;
* club-context features use only information available before the move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

TOTAL_KEYS = [
    "goals", "assists", "shots", "shots_on_target", "xg_shot", "xg_assist",
    "key_passes", "progressive_passes", "passes_to_final_third", "crosses",
    "through_passes", "forward_passes", "long_passes", "dribbles",
    "successful_dribbles", "interceptions", "recoveries", "clearances",
    "accelerations", "progressive_run", "touch_in_box", "received_pass",
    "shot_assists", "defensive_duels", "aerial_duels",
]

PER90_COLS = [
    "key_passes_per90", "progressive_passes_per90",
    "passes_to_final_third_per90", "crosses_per90", "through_passes_per90",
    "forward_passes_per90", "long_passes_per90", "dribbles_per90",
    "successful_dribbles_per90", "recoveries_per90", "clearances_per90",
    "accelerations_per90", "progressive_run_per90", "touch_in_box_per90",
    "received_pass_per90", "shot_assists_per90", "aerial_duels_per90",
]

PERCENT_COLS = [
    "pass_accuracy", "duels_won_percent", "defensive_duels_won_percent",
    "aerial_duels_won_percent", "successful_dribbles_percent",
    "shots_on_target_percent", "goal_conversion_percent",
    "successful_passes_to_final_third_percent",
    "successful_progressive_passes_percent",
]


def per90(row: pd.Series, col: str) -> float:
    if row is None or len(row) == 0:
        return np.nan

    minutes = row.get("minutes_played", np.nan)

    if pd.isna(minutes) or minutes <= 0:
        return np.nan

    val = row.get(col, np.nan)

    if pd.isna(val):
        return np.nan

    return float(val) * 90.0 / float(minutes)


def pre_features(row: pd.Series) -> dict:
    # Missing pre-transfer record
    if row is None or len(row) == 0:
        row = pd.Series(dtype=float)

    minutes = row.get("minutes_played", np.nan)
    minutes = float(minutes) if pd.notna(minutes) else np.nan

    matches = row.get("matches", np.nan)
    matches = float(matches) if pd.notna(matches) else np.nan

    matches_started = row.get("matches_started", np.nan)

    goals90 = per90(row, "goals")
    assists90 = per90(row, "assists")

    feats = {
        "pre_minutes": minutes,

        "pre_matches": matches,

        "pre_start_ratio": (
            float(matches_started) / matches
            if pd.notna(matches_started)
            and pd.notna(matches)
            and matches > 0
            else np.nan
        ),

        "pre_ga90": (
            goals90 + (assists90 if not np.isnan(assists90) else 0.0)
            if not np.isnan(goals90)
            else (
                assists90
                if not np.isnan(assists90)
                else np.nan
            )
        ),

        "pre_goals90": goals90,

        "pre_assists90": assists90,

        "pre_xg90": per90(row, "xg_shot"),

        "pre_xa90": per90(row, "xg_assist"),

        "pre_interceptions90": per90(row, "interceptions"),

        "pre_defensive_actions90": (
            float(row.get("defensive_actions")) * 90.0 / minutes
            if pd.notna(row.get("defensive_actions"))
            and pd.notna(minutes)
            and minutes > 0
            else np.nan
        ),

        "pre_duels90": per90(row, "duels"),

        "pre_duels_won90": per90(row, "duels_won"),
    }

    for c in PER90_COLS:
        value = row.get(c, np.nan)
        feats[f"pre_{c}"] = value if pd.notna(value) else np.nan

    for c in PERCENT_COLS:
        value = row.get(c, np.nan)
        feats[f"pre_{c}"] = value if pd.notna(value) else np.nan

    return feats

def post_features(pre: pd.Series, post: pd.Series) -> dict:
    out = {
        "post_minutes": float(post["minutes_played"]),
        "post_matches": float(post["matches"]) if pd.notna(post.get("matches")) else np.nan,
        "post_starts": float(post["matches_started"]) if pd.notna(post.get("matches_started")) else np.nan,
        "post_start_ratio": (
            post["matches_started"] / post["matches"]
            if pd.notna(post.get("matches")) and post["matches"] > 0 else np.nan
        ),
        "post_ga90": per90(post, "goals") + (per90(post, "assists") if not np.isnan(per90(post, "assists")) else 0.0),
        "post_xg90": per90(post, "xg_shot"),
    }
    prem, postm = pre["minutes_played"], post["minutes_played"]
    out["retention_playtime"] = (postm / prem) if prem and prem > 0 else np.nan
    pre_xg = per90(pre, "xg_shot")
    post_xg = per90(post, "xg_shot")
    out["xg_retention"] = (post_xg / pre_xg) if pre_xg and not np.isnan(pre_xg) and pre_xg > 0 else np.nan
    return out


def profile_features(players: pd.DataFrame, pid: int, move_ts) -> dict:
    sel = players[players["player_id"] == pid]
    if sel.empty:
        return {"position_group": None, "foot": None, "height_cm": np.nan,
                "weight_kg": np.nan, "age_at_transfer": np.nan, "is_spanish": 0}
    p = sel.iloc[0]
    age = np.nan
    if pd.notna(p["birth_date"]):
        age = (pd.Timestamp(move_ts) - pd.Timestamp(p["birth_date"])).days / 365.25
    span = 0
    if pd.notna(p["birth_area_id"]):
        span = 1 if int(p["birth_area_id"]) == config.LALIGA_AREA_ID else 0
    return {
        "position_group": p["role_name"] if pd.notna(p["role_name"]) else None,
        "foot": p["foot"],
        "height_cm": p["height_cm"],
        "weight_kg": p["weight_kg"],
        "age_at_transfer": age,
        "is_spanish": 1 if pd.notna(p["passport_area_id"]) and int(p["passport_area_id"]) == config.LALIGA_AREA_ID else span,
    }


def club_mv_stats(valuation: pd.DataFrame, team_id, before_ts) -> float:
    v = valuation[
        (valuation["team_id"] == team_id)
        & (pd.to_datetime(valuation["valuation_date"]) <= pd.Timestamp(before_ts))
    ]
    if v.empty:
        return np.nan
    return float(np.log1p(v["market_value"].mean()))


def squad_minutes_share(stint: pd.DataFrame, player_id, team_id, season) -> float:
    seg = stint[(stint["team_id"] == team_id) & (stint["super_season_id"] == season)]
    tot = seg["minutes_played"].sum()
    if not tot or tot <= 0:
        return np.nan
    own = seg[seg["player_id"] == player_id]["minutes_played"].sum()
    return float(own) / tot


def target_role_share(stint: pd.DataFrame, players: pd.DataFrame, role, team_id, season) -> float:
    """Minutes share of players already playing the same role at the target
    club in the season BEFORE the move (competition-for-position proxy)."""
    if role is None:
        return np.nan
    seg = stint[(stint["team_id"] == team_id) & (stint["super_season_id"] == season)]
    tot = seg["minutes_played"].sum()
    if not tot or tot <= 0:
        return np.nan
    pids = players[players["role_name"] == role]["player_id"].tolist()
    same = seg[seg["player_id"].isin(pids)]["minutes_played"].sum()
    return float(same) / tot


def build_transfer_row(transfer, stint, players, teams, valuation) -> dict | None:
    pid = transfer["player_id"]
    s_pre, s_post = int(transfer["pre_season"]), int(transfer["post_season"])
    source_team, target_team = int(transfer["source_team_id"]), int(transfer["target_team_id"])
    if source_team == target_team:
        return None
    kind = transfer["source"]

    if kind == "winter":
        pre_stints = stint[
            (stint["player_id"] == pid)
            & (stint["super_season_id"] == s_pre)
            & (stint["team_id"] != target_team)
        ]
        move_ts = pd.Timestamp(f"{s_pre}-01-25")
    else:
        pre_stints = stint[
            (stint["player_id"] == pid)
            & (stint["super_season_id"] == s_pre)
            & (stint["team_id"] == source_team)
        ]
        move_ts = pd.Timestamp(f"{s_post}-07-01")

    # Need a usable pre-transfer stint (performance before the move)
    pre = pre_stints.sort_values("minutes_played").iloc[-1] if not pre_stints.empty else None
    if pre is None:
        return None

    post_stints = stint[
        (stint["player_id"] == pid)
        & (stint["super_season_id"] == s_post)
        & (stint["team_id"] == target_team)
    ]
    post = post_stints.iloc[0] if not post_stints.empty else None
    if post is None:
        return None

    row = {
        "transfer_id": transfer["transfer_id"],
        "player_id": pid,
        "player_name": players.loc[players["player_id"] == pid, "short_name"].iloc[0]
        if (players["player_id"] == pid).any() else None,
        "source_team_id": source_team,
        "target_team_id": target_team,
        "pre_season": s_pre,
        "post_season": s_post,
        "source_kind": kind,
        "transfer_date": move_ts,
    }
    row["source_team_name"] = teams.loc[teams["team_id"] == source_team, "name"].iloc[0] if (teams["team_id"] == source_team).any() else str(source_team)
    row["target_team_name"] = teams.loc[teams["team_id"] == target_team, "name"].iloc[0] if (teams["team_id"] == target_team).any() else str(target_team)

    row.update(profile_features(players, pid, move_ts))
    row.update(pre_features(pre))
    row.update(post_features(pre, post))

    # club context (leakage safe: uses data up to the pre season)
    row["source_squad_mv"] = club_mv_stats(valuation, source_team, move_ts)
    row["target_squad_mv"] = club_mv_stats(valuation, target_team, move_ts)
    row["squad_mv_diff"] = row["target_squad_mv"] - row["source_squad_mv"] if (
        pd.notna(row["target_squad_mv"]) and pd.notna(row["source_squad_mv"])) else np.nan
    row["source_minutes_share"] = squad_minutes_share(stint, pid, source_team, s_pre)
    row["target_role_share"] = target_role_share(
        stint, players, row["position_group"], target_team, s_pre)

    # pre-transfer market value
    mv = valuation[
        (valuation["player_id"] == pid)
        & (pd.to_datetime(valuation["valuation_date"]) <= pd.Timestamp(move_ts))
    ]
    row["pre_market_value"] = float(mv.sort_values("valuation_date").iloc[-1]["market_value"]) if not mv.empty else np.nan

    # club strength proxy: league position via wins (available pre transfer)
    row["source_club_wins_pre"] = _club_wins(stint, source_team, s_pre)
    row["target_club_wins_pre"] = _club_wins(stint, target_team, s_pre)
    row["club_wins_diff"] = row["target_club_wins_pre"] - row["source_club_wins_pre"]
    return row


def _club_wins(stint: pd.DataFrame, team_id, season) -> float:
    seg = stint[(stint["team_id"] == team_id) & (stint["super_season_id"] == season)]
    return float(seg["wins"].sum()) if len(seg) else np.nan


def build_historical_dataset(transfers, stint, players, teams, valuation):
    rows = []
    for _, t in transfers.iterrows():
        r = build_transfer_row(t.to_dict(), stint, players, teams, valuation)
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)