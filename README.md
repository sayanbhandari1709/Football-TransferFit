# LaLiga 2022/23 – 2024/25 — research dataset

13 Parquet files (1.5 MB total). Spanish first division (`La Liga`, `competition_id = 795`),
three completed seasons. Player-level, season granularity, plus market valuations.

| Season | `season_id` | `super_season_id` | Span |
|---|---|---|---|
| 2022/2023 | 188124 | 2023 | 2022-08-12 → 2023-06-04 |
| 2023/2024 | 189012 | 2024 | 2023-08-11 → 2024-05-26 |
| 2024/2025 | 189974 | 2025 | 2024-08-17 → 2025-05-25 |

## Files

**Keys / lookups**
- `competitions` (1) — the competition itself
- `seasons` (3), `super_seasons` (3) — see the join note below, this matters
- `areas` (263) — country lookup for `teams.area_id`, `players.birth_area_id`,
  `players.passport_area_id`, `players.second_citizenship_area_id`

**Entities**
- `teams` (24) — 20 per season; the extra 4 are promotion/relegation churn
- `players` (1,265) — distinct players across the three seasons
- `player_careers` (2,223) — **squad membership per season**, incl. `shirt_number`.
  Grain: one row per (player, team, season).

**Valuations**
- `player_valuations` (8,433) — `market_value` in EUR, no nulls. Window 2022-07-01 → 2025-06-30
  (deliberately wider than the match calendar so pre-season transfer windows are included).
  Covers 1,132 of the 1,265 players (89.5%); the rest have no valuation on record.

**Season statistics** (1,805 player-seasons: 596 / 608 / 601)
- `player_season_stats` — the spine. Grain: (player, team, competition, super_season).
  Carries `player_season_stats_id`, `matches`, `matches_started`, `wins`, `positions` (JSON).
- `player_season_stats_total` (107 cols) — raw counting stats
- `player_season_stats_average` (100 cols) — per-match averages
- `player_season_stats_per90` (97 cols) — per-90-minutes rates
- `player_season_stats_percent` (35 cols) — success percentages

All four roll-ups join 1:1 to the spine on `player_season_stats_id`, with full coverage
(0 missing rows, 0 nulls in `minutes_played` / `goals` / `assists` / `passes`).

## The one join gotcha

Squad data is keyed by **`season_id`**, statistics by **`super_season_id`**. They are not the
same number. Go through the `seasons` table to move between them:

```python
import pandas as pd

seasons = pd.read_parquet("seasons.parquet")          # season_id <-> super_season_id
careers = pd.read_parquet("player_careers.parquet")   # keyed by season_id
stats   = pd.read_parquet("player_season_stats.parquet")  # keyed by super_season_id

careers = careers.merge(seasons[["season_id", "super_season_id"]], on="season_id")
df = careers.merge(stats, on=["player_id", "team_id", "super_season_id"], how="left")
df = df.merge(pd.read_parquet("player_season_stats_total.parquet"),
              on="player_season_stats_id", how="left")
```

The `how="left"` is intentional: 2,223 registrations vs 1,805 stat rows, because a registered
player who never appeared has no statistics row.

## Known gaps

- **No team-level season statistics.** `team_season_stats` and its three roll-ups exist in the
  source schema but are empty upstream (0 rows for every competition, not just LaLiga), so they
  are not shipped. Team aggregates have to be derived from the player rows.
- Nothing at match granularity, no financial metrics, no model predictions — out of scope by
  design.

## Notes on types

`numeric` columns were cast to `float64` and `jsonb` columns to JSON strings so the files load
cleanly in pandas/DuckDB without decimal or struct handling. Internal ingestion columns
(`created_at`, `updated_at`) and `players.image_url` were dropped.

Source: `soccersolver_unified`, schema `sports`, snapshot of 2026-08-11. See `_meta.json` for the
exact provenance and `_manifest.csv` for per-file row/column counts.
