"""TRANSFERFIT - preprocessing: stint panel and transfer derivation.

A "stint" is the grain of the performance universe:
one row per (player, team, super_season) with all available statistics.

Three transfer sources are combined:

* clean:
  The player was registered at one club in season S and a different
  club in season S+1. The transfer date is approximated by the start
  of the target season.

* winter:
  The player was registered at two clubs within one season. The source
  club is the club with fewer minutes and the target club is the club
  with more minutes. The transfer date is approximated by the midpoint
  of that season.

* club_transfers:
  The Real Madrid ledger with exact transfer dates from namespace B,
  bridged by player name into namespace A.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.data_loading import (
    load_namespace_a,
    load_namespace_b,
    resolve_player_b2a,
)


# ============================================================
# STINT PANEL
# ============================================================

def build_stint_panel(
    data: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Build the player/team/season performance universe.

    Grain:
        one row per
        (player_id, team_id, super_season_id, competition_id)

    Statistics are joined from:
        - player_season_stats
        - player_season_stats_total
        - player_season_stats_per90
        - player_season_stats_percent
    """

    # --------------------------------------------------------
    # Base tables
    # --------------------------------------------------------

    stats = data["player_season_stats"].copy()

    seasons = data["seasons"][
        [
            "season_id",
            "super_season_id",
            "start_date",
            "end_date",
        ]
    ].copy()

    careers = data["player_careers"][
        [
            "player_id",
            "team_id",
            "season_id",
            "competition_id",
        ]
    ].copy()

    # --------------------------------------------------------
    # Attach season information to career records
    # --------------------------------------------------------

    careers = careers.merge(
        seasons,
        on="season_id",
        how="left",
    )

    # --------------------------------------------------------
    # Join base season statistics
    # --------------------------------------------------------

    stint = careers.merge(
        stats,
        on=[
            "player_id",
            "team_id",
            "super_season_id",
            "competition_id",
        ],
        how="left",
        validate="m:1",
    )

    # --------------------------------------------------------
    # Join additional statistics
    # --------------------------------------------------------

    merges = [
        (
            data["pss_total"],
            "player_season_stats_id",
        ),
        (
            data["pss_per90"],
            "player_season_stats_id",
        ),
        (
            data["pss_percent"],
            "player_season_stats_id",
        ),
    ]

    for table, key in merges:

        cols = [
            key
        ] + [
            c
            for c in table.columns
            if c != key
        ]

        stint = stint.merge(
            table[cols],
            on=key,
            how="left",
        )

    # --------------------------------------------------------
    # Basic cleaning
    # --------------------------------------------------------

    stint["minutes_played"] = (
        stint["minutes_played"]
        .fillna(0.0)
    )

    stint["has_stats"] = (
        stint["player_season_stats_id"]
        .notna()
    )

    # --------------------------------------------------------
    # Normalize season/date columns
    # --------------------------------------------------------

    stint["super_season_id"] = pd.to_numeric(
        stint["super_season_id"],
        errors="coerce",
    )

    stint["start_date"] = pd.to_datetime(
        stint["start_date"],
        errors="coerce",
    )

    stint["end_date"] = pd.to_datetime(
        stint["end_date"],
        errors="coerce",
    )

    return stint


# ============================================================
# CLEAN / SUMMER TRANSFERS
# ============================================================

def build_transfers_clean(
    stint: pd.DataFrame
) -> pd.DataFrame:
    """
    Derive summer-window transfers between consecutive
    seasons.

    Unlike the original implementation, seasons are NOT
    hardcoded to 2023->2024 and 2024->2025.

    Every consecutive season available in the stint panel
    is examined.

    Example:

        2020 -> 2021
        2021 -> 2022
        2022 -> 2023
        2023 -> 2024
        2024 -> 2025

    Only players who have statistics in both seasons and
    exactly one club per season are considered.
    """

    # --------------------------------------------------------
    # Keep only player/team records with statistics
    # --------------------------------------------------------

    joined = stint[
        stint["has_stats"]
    ].copy()

    if joined.empty:

        return pd.DataFrame(
            columns=[
                "transfer_id",
                "player_id",
                "source_team_id",
                "target_team_id",
                "pre_season",
                "post_season",
                "source",
            ]
        )

    # --------------------------------------------------------
    # Determine number of clubs per player per season
    # --------------------------------------------------------

    team_by_player_season = (
        joined
        .groupby(
            [
                "player_id",
                "super_season_id",
            ]
        )["team_id"]
        .nunique()
        .rename("nclubs")
        .reset_index()
    )

    # --------------------------------------------------------
    # Keep players with exactly one club in each season
    # --------------------------------------------------------

    single = joined.merge(
        team_by_player_season,
        on=[
            "player_id",
            "super_season_id",
        ],
        how="left",
    )

    single = single[
        single["nclubs"] == 1
    ].copy()

    if single.empty:

        return pd.DataFrame(
            columns=[
                "transfer_id",
                "player_id",
                "source_team_id",
                "target_team_id",
                "pre_season",
                "post_season",
                "source",
            ]
        )

    # --------------------------------------------------------
    # Create player x season -> club matrix
    # --------------------------------------------------------

    piv = single.pivot_table(
        index="player_id",
        columns="super_season_id",
        values="team_id",
        aggfunc="first",
    )

    # --------------------------------------------------------
    # Automatically discover all available seasons
    # --------------------------------------------------------

    seasons = sorted(
        int(s)
        for s in piv.columns
        if pd.notna(s)
    )

    rows = []

    # --------------------------------------------------------
    # Compare every consecutive season
    # --------------------------------------------------------

    for i in range(
        len(seasons) - 1
    ):

        s_pre = seasons[i]
        s_post = seasons[i + 1]

        # Make sure both columns exist
        if (
            s_pre not in piv.columns
            or
            s_post not in piv.columns
        ):
            continue

        # ----------------------------------------------------
        # Players present in both seasons
        # ----------------------------------------------------

        both = piv.dropna(
            subset=[
                s_pre,
                s_post,
            ]
        )

        # ----------------------------------------------------
        # Club changed
        # ----------------------------------------------------

        moved = both[
            both[s_pre] != both[s_post]
        ]

        # ----------------------------------------------------
        # Create transfer rows
        # ----------------------------------------------------

        for player_id, r in moved.iterrows():

            source_team = r[s_pre]
            target_team = r[s_post]

            if pd.isna(
                source_team
            ) or pd.isna(
                target_team
            ):
                continue

            rows.append(
                {
                    "transfer_id":
                        f"clean_{int(player_id)}_{s_post}",

                    "player_id":
                        int(player_id),

                    "source_team_id":
                        int(source_team),

                    "target_team_id":
                        int(target_team),

                    "pre_season":
                        int(s_pre),

                    "post_season":
                        int(s_post),

                    "source":
                        "clean",
                }
            )

    # --------------------------------------------------------
    # Return consistent dataframe even when empty
    # --------------------------------------------------------

    return pd.DataFrame(
        rows,
        columns=[
            "transfer_id",
            "player_id",
            "source_team_id",
            "target_team_id",
            "pre_season",
            "post_season",
            "source",
        ],
    )


# ============================================================
# WINTER TRANSFERS
# ============================================================

def build_transfers_winter(
    stint: pd.DataFrame
) -> pd.DataFrame:
    """
    Detect in-season transfers.

    A player is considered a winter transfer when they have
    statistics for at least two different clubs within the
    same super-season.

    Source club:
        club with fewer minutes

    Target club:
        club with more minutes
    """

    # --------------------------------------------------------
    # Keep only records with statistics
    # --------------------------------------------------------

    valid = stint[
        stint["has_stats"]
    ].copy()

    if valid.empty:

        return pd.DataFrame(
            columns=[
                "transfer_id",
                "player_id",
                "source_team_id",
                "target_team_id",
                "pre_season",
                "post_season",
                "source",
            ]
        )

    # --------------------------------------------------------
    # Find player-seasons involving multiple clubs
    # --------------------------------------------------------

    multi = (
        valid
        .groupby(
            [
                "player_id",
                "super_season_id",
            ]
        )["team_id"]
        .nunique()
        .rename("nclubs")
        .reset_index()
    )

    multi = multi[
        multi["nclubs"] >= 2
    ].copy()

    rows = []

    # --------------------------------------------------------
    # Process each multi-club player-season
    # --------------------------------------------------------

    for _, m in multi.iterrows():

        pid = int(
            m["player_id"]
        )

        season = int(
            m["super_season_id"]
        )

        seg = valid[
            (valid["player_id"] == pid)
            &
            (
                valid["super_season_id"]
                == season
            )
        ].copy()

        # ----------------------------------------------------
        # Total minutes per club
        # ----------------------------------------------------

        by_team = (
            seg
            .groupby("team_id")[
                "minutes_played"
            ]
            .sum()
            .sort_values()
        )

        if len(by_team) < 2:
            continue

        # ----------------------------------------------------
        # Approximate source/target using minutes
        # ----------------------------------------------------

        source = by_team.index[0]
        target = by_team.index[-1]

        if source == target:
            continue

        rows.append(
            {
                "transfer_id":
                    f"winter_{pid}_{season}_"
                    f"{int(source)}_{int(target)}",

                "player_id":
                    pid,

                "source_team_id":
                    int(source),

                "target_team_id":
                    int(target),

                "pre_season":
                    season,

                "post_season":
                    season,

                "source":
                    "winter",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "transfer_id",
            "player_id",
            "source_team_id",
            "target_team_id",
            "pre_season",
            "post_season",
            "source",
        ],
    )


# ============================================================
# REAL MADRID TRANSFER LEDGER
# ============================================================

def build_rm_ledger(
    data_a,
    data_b,
    bridge: dict
) -> pd.DataFrame:
    """
    Build transfer records from the namespace-B Real Madrid
    transfer ledger.

    Player names from namespace B are bridged to namespace A
    player IDs.

    Transfer dates remain the exact ledger dates.
    """

    ct = data_b[
        "club_transfers"
    ].copy()

    ct["transfer_date"] = pd.to_datetime(
        ct["transfer_date"],
        errors="coerce",
    )

    resolved = bridge.get(
        "resolved",
        {},
    )

    rows = []

    # --------------------------------------------------------
    # Process every ledger transfer
    # --------------------------------------------------------

    for _, r in ct.iterrows():

        player_name = r.get(
            "player_name"
        )

        if pd.isna(
            player_name
        ):
            continue

        player_name = str(
            player_name
        ).strip()

        # ----------------------------------------------------
        # Find corresponding namespace-A player
        # ----------------------------------------------------

        a_player = resolved.get(
            player_name
        )

        if a_player is None:
            continue

        # ----------------------------------------------------
        # Transfer direction
        # ----------------------------------------------------

        direction = r[
            "direction"
        ]

        rm_id = config.RM_TEAM_ID_A

        counterpart = r[
            "counterpart_club_id"
        ]

        if pd.isna(
            counterpart
        ):
            continue

        counterpart = int(
            counterpart
        )

        if direction == "out":

            source = rm_id
            target = counterpart

        else:

            source = counterpart
            target = rm_id

        # ----------------------------------------------------
        # Transfer ID
        # ----------------------------------------------------

        transfer_date = r[
            "transfer_date"
        ]

        if pd.isna(
            transfer_date
        ):
            date_text = "unknown_date"
        else:
            date_text = str(
                transfer_date.date()
            )

        transfer_id = (
            f"ledger_{player_name}_"
            f"{date_text}_{direction}"
        )

        # ----------------------------------------------------
        # Build row
        # ----------------------------------------------------

        rows.append(
            {
                "transfer_id":
                    transfer_id,

                "player_id":
                    int(
                        a_player[
                            "player_id"
                        ]
                    ),

                "player_name_b":
                    player_name,

                "transfer_date":
                    transfer_date,

                "source_team_id":
                    source,

                "target_team_id":
                    target,

                "direction":
                    direction,

                "price":
                    r.get(
                        "price"
                    ),

                "market_value_at_transfer":
                    r.get(
                        "market_value_at_transfer"
                    ),

                "is_loan":
                    bool(
                        r.get(
                            "is_loan",
                            False,
                        )
                    ),

                "source":
                    "club_transfers",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "transfer_id",
            "player_id",
            "player_name_b",
            "transfer_date",
            "source_team_id",
            "target_team_id",
            "direction",
            "price",
            "market_value_at_transfer",
            "is_loan",
            "source",
        ],
    )


# ============================================================
# PREVIEW / DATASET CHECK
# ============================================================

def preview_count():
    """
    Load the datasets and print a quick summary of the
    preprocessing output.

    Useful for checking whether older seasons are actually
    present in the source data.
    """

    # --------------------------------------------------------
    # Load namespaces
    # --------------------------------------------------------

    a = load_namespace_a()
    b = load_namespace_b()

    # --------------------------------------------------------
    # Build stint panel
    # --------------------------------------------------------

    stint = build_stint_panel(
        a
    )

    # --------------------------------------------------------
    # Show season coverage
    # --------------------------------------------------------

    available_seasons = sorted(
        stint[
            "super_season_id"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        "\nAvailable super-seasons:"
    )

    print(
        available_seasons
    )

    # --------------------------------------------------------
    # Player-season coverage
    # --------------------------------------------------------

    if not stint.empty:

        min_date = stint[
            "start_date"
        ].min()

        max_date = stint[
            "end_date"
        ].max()

        print(
            "\nStint date coverage:"
        )

        print(
            f"{min_date} -> {max_date}"
        )

    # --------------------------------------------------------
    # Build namespace-B bridge
    # --------------------------------------------------------

    bridge = resolve_player_b2a(
        a["players"],
        sorted(
            b[
                "club_transfers"
            ][
                "player_name"
            ]
            .dropna()
            .unique()
        ),
    )

    # --------------------------------------------------------
    # Derive transfer datasets
    # --------------------------------------------------------

    clean = build_transfers_clean(
        stint
    )

    winter = build_transfers_winter(
        stint
    )

    ledger = build_rm_ledger(
        a,
        b,
        bridge,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        f"\nstint rows: {len(stint):,}"
    )

    print(
        f"clean transfers: {len(clean):,}"
    )

    print(
        f"winter transfers: {len(winter):,}"
    )

    print(
        f"ledger transfers (bridged): {len(ledger):,}"
    )

    # --------------------------------------------------------
    # Clean transfer season coverage
    # --------------------------------------------------------

    if not clean.empty:

        print(
            "\nClean transfer season pairs:"
        )

        pairs = (
            clean[
                [
                    "pre_season",
                    "post_season",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                [
                    "pre_season",
                    "post_season",
                ]
            )
        )

        print(
            pairs.to_string(
                index=False
            )
        )

    return (
        clean,
        winter,
        ledger,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    preview_count()