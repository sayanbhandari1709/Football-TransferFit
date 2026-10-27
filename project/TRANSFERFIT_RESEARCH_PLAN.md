# TRANSFERFIT — Research Plan & Data Download List

## 0. Where we are (verified by re-running your code)

| Finding | Evidence |
|---|---|
| Reported test AUC 0.97 is in-sample | `time_split_test_predictions.csv` was produced by the production model trained on all 148 transfers |
| Honest held-out AUC of the production model (LR, variant D) is ~0.68 (after similarity fixes: 0.675), 95% CI ≈ 0.53–0.81 | `honest_eval_time_split.csv` |
| **A one-variable baseline (`pre_market_value`) beats every model** | AUC 0.712, Brier 0.192, log loss 0.565 vs. LR-D 0.675 / 0.235 / 0.997 |
| No model is significantly better than that baseline | paired ΔAUC CIs all include 0 |
| LR/GBM/MLP are worse than a constant on log loss | overconfident probabilities |
| MLP on 148 rows gives no gain | AUC 0.674 (C) vs 0.685 LR |
| Similarity (variant D) had 3 bugs | train/test scaling mismatch; 3 features silently dropped (missing `pre_` prefix); pandas-3 dtype issue |
| All 148 transfers are La Liga → La Liga | source and target team both in `teams.parquet` |
| 30% of rows are "insufficient playing time", 1 row is a real production collapse | label = mostly minutes |

**Consequence:** the current data cannot support a neural-network claim. The research question has to be re-posed so that
(a) cross-league moves are observed, (b) the outcome separates opportunity from performance, (c) data is open and redistributable.

## 1. Data: what to download

### 1A. StatsBomb Open Data (`github.com/hudl/open-data`) — YES, but only this slice

| Take | Why | Size |
|---|---|---|
| `data/competitions.json` | index | tiny |
| `data/matches/<comp>/<season>.json` for **La Liga, Premier League, Serie A, Ligue 1, season 2015/16** | the only full men's league seasons in the repo (377–380 matches each) | small |
| `data/lineups/<match_id>.json` for those 1,517 matches | minutes, positions, subs, cards | ~36 MB (I fetched them) |
| `data/events/<match_id>.json` for those 1,517 matches | player/team event profiles | ~3 MB each, ~4.5 GB total |
| `doc/` (event spec) | schema | tiny |
| *Optional:* La Liga (Barcelona-only) 2016/17–2020/21 matches+lineups+events | case studies of arrivals (Umtiti, Alcácer, D. Suárez, Digne, André Gomes) | ~1 GB |

**Skip:** World Cups, Euros, Copa América, AFCON, women's, youth, MLS, Indian Super League, one-match Champions League/Copa del Rey seasons, Bundesliga (Leverkusen only), `three-sixty` (tracking-like data for a handful of matches).

**Attribution is mandatory** in anything you publish: credit StatsBomb and use their logo (media pack).

Important quirks found: (i) team names vary inside one competition (e.g. "Marseille" vs "Olympique de Marseille") — always key on `team_id`, never name; (ii) a lineup position with `to = null` means played to the end; (iii) `has_360` flags in `competitions.json` are not reliable for coverage.

### 1B. Wyscout public event dataset (Pappalardo & Massucco 2019) — recommended addition

Full 2017/18 season of La Liga, Serie A, Premier League, Bundesliga, Ligue 1 (+ WC 2018, Euro 2016). Published on figshare; the code repos describing it state it is released under **CC BY 4.0**. Cite: Pappalardo et al., *Scientific Data* 6, 236 (2019). Download: `matches`, `events`, `players`, `teams`, `competitions`, `coaches` for the five leagues. (I confirmed existence and license from public pages; I did **not** download it — check file sizes and terms yourself before relying on it.)

### 1C. Existing data (keep)
La Liga 2022/23–2024/25 parquet + valuations + ledger. **Blocker for SSAC:** these are marked "license NEEDS VERIFICATION" and SSAC requires open-source inputs. Treat as *supplementary* unless the licence is cleared.

## 2. The re-posed research design

**Title (working):** *Lost opportunity or lost performance? A leakage-safe, open benchmark of transfer adaptation across leagues.*

**Unit of analysis:** a player who changes club **within a season** (winter window) in a provider's full-season data, with events observed both before and after the move, within one provider (so pre/post features are consistent).

| Dataset | Provider | Movers (expected) | Role |
|---|---|---|---|
| StatsBomb 2015/16 (4 leagues) | StatsBomb | 145 raw movers, 67 with ≥300 min at both clubs (cross-league is a small subset — recount after cleaning club-name variants) | open training/validation set |
| Wyscout 2017/18 (5 leagues) | Wyscout | untested — count during build | **external test set** (different provider and season) |
| La Liga 2022–25 | private | 148 intra-league | supplementary, with market values |
| Barcelona arrivals 2016–21 | StatsBomb | ~5 direct | qualitative case studies |

Honest caveat to state in the paper: winter movers are a selected sample with ≤ ~half a season of post data; results describe in-season adaptation, not summer-window moves.

### Research questions
1. **RQ1 Decomposition:** what share of "failed" moves is lost *opportunity* (minutes share) vs lost *performance* (role-appropriate output vs expectation)?
2. **RQ2 Predictability:** do pre-move features (profile, source/target style, league shift) predict either outcome beyond simple baselines (market value where available, minutes, base rate)?
3. **RQ3 Representation:** does a self-supervised player-style embedding plus a league/club-shift representation beat hand-crafted features? Test: train on StatsBomb 2015/16, test on Wyscout 2017/18.
4. **RQ4 Calibration & abstention:** are probabilities calibrated, and does conformal prediction let the tool abstain?

### Outcomes (replace the 900-minute binary label)
- **Opportunity** = post-move minutes ÷ minutes available after arrival (handles half-season exposure).
- **Performance** = role-specific composite (z-scored per position): attackers → xG+xA-type value and shot/carry output; midfielders → progression/passing/defensive mix; defenders → defensive actions, duels, clearances, passing; keepers excluded from v1 unless shot-stopping is derivable. Expressed as post minus **expected** (shrinkage prior by position/age/league), to avoid regression-to-the-mean artefacts.
- Binary "adapted" retained only as a secondary target with a sensitivity grid.

### Models
Baselines → regularised LR → GBM/XGBoost → **neural**: (a) player-season autoencoder/contrastive encoder pretrained on every player-season in all datasets, (b) league/club-shift features (difference of team/league style vectors: pace, pressing, directness, possession), (c) small multi-task head (opportunity + performance, heteroscedastic or 5-seed ensemble), (d) optional retrieval over similar historical movers. Promote a neural model **only** if it beats the best classical model by a paired bootstrap on the external Wyscout test set. Report negative results honestly.

### Evaluation protocol
Train on StatsBomb 2015/16 → test Wyscout 2017/18 (primary). Reverse direction and grouped CV (by player) as secondary. Always report: baselines, 95% bootstrap CIs, paired ΔAUC/ΔMAE vs baseline, Brier, log loss, ECE, subgroup results (position, cross-league vs same-league, age band).

## 3. Build order
0. Apply the Phase-0 patch (leak fix, honest harness, similarity fixes).
1. Data ingestion (StatsBomb, Wyscout) → common event schema.
2. Harmonised player-season features + team/league style vectors (+ own xG model trained on StatsBomb shots, applied to both providers).
3. Mover detection + pre/post windows + outcomes.
4. Baselines + honest evaluation on movers.
5. Neural encoder + heads + ablations.
6. Conformal calibration, subgroup analysis, case studies.
7. App (Streamlit): interval output, opportunity vs performance breakdown, comparable movers, model card.
8. Paper + open repo (README, pinned env, download scripts, attribution).

## 4. Risks
- Mover counts may be small (≈60–150 per provider); the paper's claims must fit that.
- Cross-provider feature harmonisation is imperfect → restrict to features computable in both, include provider as a check.
- 2015–2018 data are old; state this limit.
- Winter-window selection bias; loans and returns need explicit handling.
- Licences: verify Wyscout terms and resolve the private-data licence.
