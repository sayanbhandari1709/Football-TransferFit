"""TRANSFERFIT - adaptation label definition.

The label is NOT "did the player score goals". It measures whether the player
established himself reasonably well at the new club in the first post-transfer
season, using only post-transfer information.

Composite rule (v1, fully documented):
1. PLAYING TIME OK   : post_minutes >= MINUTES_THRESHOLD
2. NO PRODUCTION COLLAPSE:
       if pre_ga90 > NET_OUTPUT_FLOOR  -> post_ga90 >= RETENTION_FLOOR * pre_ga90
       (alternative: heavy usage post_minutes >= HEAVY_USE saves the label)

success = PLAYING TIME OK AND NO PRODUCTION COLLAPSE

Care: the label uses only post-transfer season data at the target club.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = {
    "minutes_threshold": 900,   # ~10 full matches of first-team exposure
    "heavy_use": 1500,          # ~17 matches -> usage outweighs raw output dip
    "retention_floor": 0.5,     # keep at least half of per-90 output
    "net_output_floor": 0.15,   # only judge collapse for players who were productive pre-transfer
}


def apply_label(df: pd.DataFrame, **params) -> pd.DataFrame:
    p = {**DEFAULTS, **params}
    out = df.copy()

    minutes_ok = out["post_minutes"] >= p["minutes_threshold"]

    ga_pre = out["pre_ga90"].fillna(0.0)
    ga_post = out["post_ga90"].fillna(0.0)
    productive_pre = ga_pre > p["net_output_floor"]
    retained = ga_post >= (p["retention_floor"] * ga_pre)
    heavy_use = out["post_minutes"] >= p["heavy_use"]
    collapse_prevention = (~productive_pre) | retained | heavy_use

    out["minutes_ok"] = minutes_ok
    out["no_collapse"] = collapse_prevention
    out["retained"] = retained
    out["label"] = (minutes_ok & collapse_prevention).astype(int)
    out["label_rules"] = np.select(
        [
            minutes_ok & collapse_prevention,
            minutes_ok & ~collapse_prevention,
            ~minutes_ok,
        ],
        [
            "adapted",
            "played but output collapsed",
            "insufficient playing time",
        ],
        default="insufficient playing time",
    )
    return out


def label_summary(df: pd.DataFrame) -> str:
    s = df["label"]
    total = len(s)
    pos = int(s.sum())
    return (
        f"transfers={total}, adapted={pos} ({pos / total:.1%}), "
        f"not adapted={total - pos} ({1 - pos / total:.1%})"
    )


def sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Report label count across a grid of thresholds (for transparency)."""
    rows = []
    for minutes in (600, 900, 1200, 1500):
        for retention in (0.4, 0.5, 0.6):
            res = apply_label(df, minutes_threshold=minutes, retention_floor=retention)
            rows.append({
                "minutes_threshold": minutes,
                "retention_floor": retention,
                "n": len(res),
                "adapted": int(res["label"].sum()),
                "positive_rate": round(float(res["label"].mean()), 3),
            })
    return pd.DataFrame(rows)