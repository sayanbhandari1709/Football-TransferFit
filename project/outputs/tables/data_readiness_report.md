# TRANSFERFIT - Data Readiness Report

Generated on C:\Football\DATA\project\outputs run.

## A. Datasets available
- Namespace A (parquet): areas, competitions, player_careers, player_season_stats, player_valuations, players, pss_average, pss_per90, pss_percent, pss_total, seasons, super_seasons, teams
- Namespace B (ledger CSVs): club_financial_metrics, club_transfers, sd_db, team_dossier, valuations_full
- Auxiliary: sample events JSON (one match), Real Madrid La Fabrica workbook, PhD proposal PDF
Coverage: single competition **La Liga** (competition_id=795), three seasons 2022/23 - 2024/25,
1265 players, 2,223 player-club-season registrations, 8,433 valuations.

## B. Datasets relevant to TRANSFERFIT
- players, player_careers, player_season_stats (spine + total/per90/percent), player_valuations,
- teams, seasons, super_seasons, areas, competitions (league lookup), club_transfers (ledger)

## C. Datasets not relevant (or secondary)
- laliga_2022_2025.zip: empty archive (no members)
- SD-DB(PLAYERS).csv, rm_la_fabrica workbook, PhdIntvProposal.pdf: Real Madrid academy/sale analytics, out of scope
- club_financial_metrics.csv: Real Madrid only (6 rows)
- team_full_dossier.csv: contains future predictions (leakage); excluded as features
- 15946 (1).json: single Barcelona-Alavés match; not league-scale, excluded from model input

## D. Join keys
- Namespace A joins directly on player_id / team_id / competition_id / super_season_id / player_season_stats_id
- Namespace B (club_transfers, valuations_full, financials) does NOT share IDs with A
- Entity resolution: name-based bridge; exact short/full-name match, token-coverage fuzzy, manual verification
  -> 92 of 105 ledger rows bridged to namespace A
- Ambiguous/unresolved ledger names (no LaLiga 22/23-24/25 data): 9 ambiguous, 4 unresolved

## E. Available pre-transfer features
- profile: age_at_transfer, role group, foot, height, weight, nationality (Spanish flag)
- performance (last stint before the move): minutes, appearances, starts, goals, assists, xG, xA,
  key passes, progressive passes, final-third passes, crosses, through balls, dribbles,
  defensive actions, interceptions, recoveries, clearances, duels, aerials (+ per-90 and % rates)
- market value: latest valuation dated strictly before the transfer
- club context: source/target squad market value, league wins proxy, minutes share, target role share

## F. Available post-transfer outcomes
- minutes, matches, starts, start ratio, goals+assists per-90 (retention ratio vs pre), xG per-90

## G. Data leakage risks (controls)
- post-transfer season stats are ONLY used to build the label, never as inputs
- valuations filtered to valuation_date <= transfer date
- club context uses only pre-transfer season data
- team_full_dossier future predictions excluded entirely
- winter-window moves approximate transfer date (season midpoint) - documented approximation

## H. Missing data
- pre-transfer performance only available inside La Liga 2022/23-2024/25 (foreign league stints absent)
- 56/105 ledger rows lack transfer fee; valuations cover ~89.5% of players
- no match-level, injury, tactical or formation data; no team haditable financial statements
- transfer fees in ledger: 49/105 present; many loans without fee
- exact transfer dates only for the RM ledger; derived moves use season boundaries/season midpoint

## I. Usable transfer cases
- clean summer-window moves (2022/23->2023/24, 2023/24->2024/25): 112
- in-season (winter) moves: 36
- feature-complete historical transfers with label: 148
- labelled positive rate: 69.6%
- RM ledger rows bridged: 92; tracked separately as reference list

## J. Recommended next step
Build the time-based model on the derived labelled pool, report metrics transparently,
and use the bridged RM ledger only for illustrative look-ups. Clearly document that the
prototype transfer history covers three La Liga seasons; a production system needs a
multi-league, multi-season transfer ledger.

## Provenance
- Parquet files: private unified DB snapshot (schema `sports`), dated 2026-08-11. License: NEEDS VERIFICATION.
- club_transfers / valuations_full / financials / dossiers: Transfermarkt-style data. License: NEEDS VERIFICATION (third-party proprietary).
- events JSON: StatsBomb Open Data structure (exact match not verified). License: NEEDS VERIFICATION.
- derived processed files are our own output (no external license).