# OpenCode prompt — TRANSFERFIT v2 pipeline

Paste everything below the line into OpenCode, opened at `C:\Football\DATA\project`.
Put `transferfit_phase0_patch.zip` and `TRANSFERFIT_RESEARCH_PLAN.md` in that folder first.

---

## ROLE
You are a senior sports-analytics ML engineer extending the existing TRANSFERFIT project (Python, scikit-learn, Streamlit) into a research-grade, open-source, end-to-end pipeline for a MIT Sloan Sports Analytics Conference research paper on **player transfer adaptation**. Read `TRANSFERFIT_RESEARCH_PLAN.md` first; it defines the research questions, datasets, outcomes and evaluation protocol. Follow it.

## GROUND RULES (non-negotiable)
1. Inspect existing files before editing. Do **not** rewrite working files; prefer small targeted edits. Preserve the current schema of `data/processed/historical_transfers.csv`.
2. Do **not** change existing paths in `src/config.py`. You may only *append* new constants (e.g. `DATA_EXTERNAL`, `SB_DIR`, `WY_DIR`).
3. **No leakage.** Features for a move may only use matches strictly before the move; outcomes only matches after. Every feature builder must take an explicit `cutoff` (date or match order) and be unit-tested for this.
4. **Never fabricate data.** If a source is missing or a join fails, log it and stop that branch; report counts.
5. Key everything on provider IDs (`team_id`, `player_id`), never on names (StatsBomb has name variants like "Marseille"/"Olympique de Marseille").
6. All held-out numbers must come from models fit on training data only. Every model is compared against baselines (base rate, market value where available, minutes) with 95% bootstrap CIs and paired tests.
7. Deterministic: fixed seeds, pinned `requirements.txt`, one command per stage.
8. Credit StatsBomb (source + logo requirement) and Wyscout/Pappalardo & Massucco (CC BY 4.0, cite Sci Data 6, 236) in README and paper materials.
9. Stop and report (do not continue to neural models) if fewer than 60 usable movers remain in any evaluation split after filtering.

## STEP 0 — Apply the Phase-0 patch
Unzip `transferfit_phase0_patch.zip` over the project (files: `src/honest_eval.py`, `src/similarity.py`, `scripts/run_honest_eval.py`, `scripts/train_evaluate.py`, `app/streamlit_app.py`). Run `python scripts/run_honest_eval.py --fast`, then the full run. Confirm `outputs/tables/honest_eval_time_split.csv` exists and that `time_split_test_predictions.csv` now comes from train-only models. Commit.

## STEP 1 — Data download scripts (`scripts/download_*.py`)
`download_statsbomb.py`:
- Fetch `competitions.json` from `https://raw.githubusercontent.com/hudl/open-data/master/data/`.
- Select men's league seasons **2015/2016** for La Liga, Premier League, Serie A, Ligue 1 (must be full seasons: 377–380 matches each; assert this). Ignore everything else by default; add an optional `--barcelona-la-liga` flag for 2016/17–2020/21 matches whose team is Barcelona.
- Download matches, lineups and events into `data/external/statsbomb/{matches,lineups,events}/` with resumable, parallel (≤16 workers), retrying downloads and a manifest CSV (path, bytes, sha256). Events total ≈4.5 GB — support `--limit N` for smoke tests.
`download_wyscout.py`:
- The Pappalardo & Massucco (2019) Wyscout dataset is on figshare (collection DOI 10.6084/m9.figshare.c.4415000). Implement a downloader that takes figshare file URLs from a small config list; if the environment blocks the download, print manual instructions and exit cleanly. Files needed: matches, events, players, teams, competitions, coaches for the five domestic leagues (2017/18). Verify the licence text in the downloaded metadata and record it in `data/external/LICENSES.md`.

## STEP 2 — Common event schema (`src/ingest/`)
- `statsbomb.py`, `wyscout.py`: stream events (never load all into memory) and emit a normalised long table: `provider, match_id, date, competition, team_id, player_id, minute, second, period, type, outcome, x, y, end_x, end_y` with pitch coordinates rescaled to a shared 0–100 frame (StatsBomb is 120×80; Wyscout is 0–100). Save as parquet partitioned by competition.
- `lineups.py`: minutes per player per match from StatsBomb lineups (positions with `from`/`to`; null `to` = end of match) and from Wyscout formation/substitution data. Output `player_match_minutes.parquet`.
- Tests: minutes never exceed match length; team sums ≈ 990 minutes per match.

## STEP 3 — Harmonised features (`src/features_v2/`)
- Player-window features (per 90, computed for any cutoff window) using **only event types present in both providers**: passes, pass completion, progressive passes (≥25% of distance to goal closed and ≥10 m), passes into final third and box, key passes/shot assists, shots, shots on target, duels and duel win %, aerial duels, tackles/interceptions/clearances/recoveries where available, dribbles/take-ons, crosses, fouls, touches in box.
- **Own xG model**: train a small logistic/GBM xG on StatsBomb shots (distance, angle, body part, shot type) using StatsBomb's own `shot_statsbomb_xg` only as a sanity check; apply the same model to Wyscout shots so xG is comparable across providers.
- Team/league style vectors per team-season: possession share, pass tempo, directness (forward pass share), pressing proxy (defensive actions in opponent half per opponent pass), shot volume, crossing share, aerial share. League vector = minutes-weighted mean.
- Position group from provider position data (GK/DEF/MID/FWD) via minutes-weighted mode.

## STEP 4 — Mover detection and outcomes (`src/movers.py`)
- For each provider-season, find players with match appearances for ≥2 different `team_id`s. Order by match date; a **move** is a change of team between consecutive appearances. Exclude loans-and-returns (A→B→A), players with ≥3 clubs, and moves with <300 minutes on either side. The move cutoff = date of the last appearance for the source club.
- Record `same_league` vs `cross_league`.
- Pre features: matches up to the cutoff, at the source club. Target-club context: target team style vector from matches **before the player's first appearance**. Source/target league shift = difference of style vectors.
- Outcomes (post window = matches after the first appearance at the target):
  - `opportunity` = post minutes ÷ minutes the target team played in the window (cap 1).
  - `performance_z` = role-specific composite (z-score within position group using all player-windows in the provider-season), minus the **expected** value from a shrinkage prior model fit on non-movers (position, age if available, league, pre performance) — this corrects regression to the mean.
  - Secondary binary `adapted` = `opportunity ≥ 0.4` and `performance_z ≥ −0.5`, with a sensitivity grid.
- Output `data/processed/movers_statsbomb_2015.parquet`, `movers_wyscout_2017.parquet` and a `mover_counts_report.md` (raw movers, after filters, by position, same/cross league). Apply the Ground Rule 9 check.

## STEP 5 — Baselines and honest evaluation (`src/eval_v2.py`, reuse `src/honest_eval.py`)
- Protocol A (primary): train on StatsBomb movers → test on Wyscout movers. Protocol B: reverse. Protocol C: repeated grouped CV by player within each provider.
- Baselines: mean predictor, minutes-only, position-only, pre-performance-only, ridge/LR, XGBoost/HistGB (install `xgboost`; treat it as experimental).
- Metrics: for regression targets MAE, RMSE, R², Spearman; for binary AUC, Brier, log loss, ECE; 95% bootstrap CIs; paired bootstrap deltas versus the best baseline. Subgroups: position, cross-league vs same-league.
- RQ1 report: variance decomposition — fraction of "not adapted" cases explained by low opportunity only, low performance only, both.
- Write `outputs/tables/v2_*.csv` and `docs/RESULTS.md`.

## STEP 6 — Neural models (`src/nn/`, PyTorch, CPU-friendly)
1. `encoder.py`: player-window autoencoder (input = harmonised per-90 features, 2 hidden layers, 16–32-d embedding, dropout, weight decay) plus a contrastive variant (two random sub-windows of the same player-season are positives). Pretrain on **all** player-season windows from every dataset (unlabelled); save embeddings.
2. `head.py`: multi-task head predicting `opportunity` and `performance_z` from [source embedding, target-style vector, league-shift vector, profile]; heteroscedastic Gaussian output or a 5-seed ensemble for uncertainty.
3. `retrieval.py` (optional): attention over the k most similar historical movers for explanation.
4. Training: early stopping on an inner validation split, 20 seeds, report mean ± SD. Ablations: no embedding, no league-shift, no target style, embedding only.
5. **Promotion rule:** a neural model is promoted only if it beats the best classical model on Protocol A by a paired bootstrap (95% CI of the difference excludes 0). Otherwise report it as a negative result. Do not tune on the test provider.

## STEP 7 — Calibration and abstention
Split-conformal prediction intervals for the regression targets and conformal sets for `adapted`; report coverage on the external test set; define an "insufficient evidence" state when the interval is wider than a stated threshold.

## STEP 8 — Case studies
Using `--barcelona-la-liga` data, score Umtiti, Alcácer, Denis Suárez, Digne and André Gomes **using only information available before each move**; show predicted vs observed outcomes. Present as qualitative validation, not evidence of accuracy.

## STEP 9 — Streamlit app (extend `app/streamlit_app.py`, do not replace)
- Prediction page: player + target club → predicted opportunity and performance with intervals, "insufficient evidence" state, top drivers, 5 most similar historical movers.
- Validation page: baselines vs models with CIs, calibration plot, ablation table, subgroup results.
- Data page: source attribution (StatsBomb, Wyscout), mover counts, limitations, model card.

## STEP 10 — Reproducibility and paper assets
- `Makefile`/`run_all.py`: one command per stage; `requirements.txt` pinned; `README.md` with data-download instructions and attribution; `docs/RESULTS.md`, `docs/LIMITATIONS.md`, `docs/DATA_LICENSES.md`.
- Figures for the paper (PDF/PNG, 300 dpi): pipeline diagram, opportunity/performance decomposition, baseline-vs-model forest plot, calibration, league-shift map.
- Leakage test suite (`tests/test_no_leakage.py`): shuffle post-move data and assert features/predictions do not change; assert all feature timestamps < cutoff.

## DELIVERABLE FORMAT
After each step: list files created/changed, commands to reproduce, key numbers (counts, metrics with CIs), and any assumption or failure. Do not proceed past a failed acceptance check. Finish with a summary that states plainly which research questions the evidence answers, which it does not, and what the neural models did or did not add.
