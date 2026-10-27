# Football Player Transfer Adaptation Intelligence

**TransferFit** is a machine-learning research project focused on predicting how well a football player may adapt after transferring to a new club.

The project investigates player performance, historical transfer information, and source/target club context to estimate player adaptation and support better transfer decisions.

---

## Dataset: LaLiga 2022/23–2024/25

13 Parquet files containing Spanish first-division football data (`competition_id = 795`) across three completed seasons.

| Season | `season_id` | `super_season_id` | Span |
|---|---:|---:|---|
| 2022/2023 | 188124 | 2023 | 2022-08-12 → 2023-06-04 |
| 2023/2024 | 189012 | 2024 | 2023-08-11 → 2024-05-26 |
| 2024/2025 | 189974 | 2025 | 2024-08-17 → 2025-05-25 |

### Dataset contents

**Keys and lookups**
- `competitions` (1)
- `seasons` (3)
- `super_seasons` (3)
- `areas` (263)

**Entities**
- `teams` (24)
- `players` (1,265)
- `player_careers` (2,223)

**Valuations**
- `player_valuations` (8,433)
- Market values in EUR, covering 1,132 of 1,265 players.

**Season statistics**
- `player_season_stats` (1,805 player-seasons)
- `player_season_stats_total` (107 columns)
- `player_season_stats_average` (100 columns)
- `player_season_stats_per90` (97 columns)
- `player_season_stats_percent` (35 columns)

All four statistical roll-ups join to the main statistics table through `player_season_stats_id`.

### Important join relationship

Squad data uses `season_id`, whereas player statistics use `super_season_id`. The `seasons` table provides the mapping between them.

```python
import pandas as pd

seasons = pd.read_parquet("seasons.parquet")
careers = pd.read_parquet("player_careers.parquet")
stats = pd.read_parquet("player_season_stats.parquet")

careers = careers.merge(
    seasons[["season_id", "super_season_id"]],
    on="season_id"
)

df = careers.merge(
    stats,
    on=["player_id", "team_id", "super_season_id"],
    how="left"
)

df = df.merge(
    pd.read_parquet("player_season_stats_total.parquet"),
    on="player_season_stats_id",
    how="left"
)
```

The left join preserves registered players who did not appear in matches and therefore have no statistics row.

### Known dataset limitations

- Team-level season statistics are unavailable in the source snapshot.
- The dataset does not contain match-level records.
- Some players have no recorded market valuation.
- Team aggregates must be derived from player-level statistics if required.

### Data provenance

Source: `soccersolver_unified`, schema `sports`, snapshot dated 2026-08-11.

Refer to `_meta.json` and `_manifest.csv` for provenance and file-level metadata.

---

## Project objective

TransferFit aims to:

1. Analyze a player's previous performance.
2. Consider the source and target club environments.
3. Estimate potential adaptation after a transfer.
4. Generate an Adaptation Score (0–100).
5. Support transfer-risk assessment and recruitment decisions.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Machine Learning
- Parquet datasets

---

*Research prototype. Predictions are intended as decision-support estimates, not definitive judgments of player performance.*
