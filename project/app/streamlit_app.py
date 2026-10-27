"""TRANSFERFIT - Streamlit dashboard.

Pages:
1. Prediction  - estimate adaptation of a player to a target club.
2. Historical validation - model comparison and calibration (real outputs).
3. Data readiness - dataset coverage and ledger reference list.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src import config
from src.data_loading import load_namespace_a
from src.prediction import predict_transfer, default_context

st.set_page_config(page_title="TransferFit", page_icon="⚽", layout="wide")

st.title("TRANSFERFIT — Player Transfer Adaptation Intelligence")
st.caption("Prototype. Estimates suitability of a player for a specific target club "
           "using only information available before the transfer. Not a replacement "
           "for scouts; an additional decision-support signal.")


@st.cache_resource
def load_context():
    return default_context()


@st.cache_data
def get_teams():
    a = load_namespace_a()
    teams = a["teams"][["team_id", "name"]].sort_values("name").reset_index(drop=True)
    return teams


@st.cache_data
def get_players():
    a = load_namespace_a()
    return a["players"][["player_id", "short_name", "role_name"]].sort_values("short_name")


def player_lookup(players: pd.DataFrame, short_name: str):
    hits = players[players["short_name"] == short_name]
    return short_name if len(hits) else None


tab_pred, tab_val, tab_data = st.tabs(
    ["Prediction", "Historical validation", "Data readiness"])


with tab_pred:
    st.subheader("Estimate adaptation")
    players = get_players()
    teams = get_teams()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pname = st.selectbox("Player (LaLiga roster 2022/23–2024/25)", players["short_name"])
    with c2:
        src = st.selectbox("Current / source club", ["— (unknown)"] + teams["name"].tolist())
    with c3:
        tgt = st.selectbox("Target club", teams["name"].tolist())
    with c4:
        eval_date = st.date_input("Transfer / evaluation date", value=pd.Timestamp("2024-07-01"))

    src_id = None if src.startswith("—") else int(teams.loc[teams["name"] == src, "team_id"].iloc[0])
    tgt_id = int(teams.loc[teams["name"] == tgt, "team_id"].iloc[0])

    if st.button("Run prediction", type="primary"):
        ctx = load_context()
        with st.spinner("Predicting..."):
            res = predict_transfer(pname, src_id, tgt_id, pd.Timestamp(eval_date).date().isoformat(), ctx)

        if res["status"] != "ok":
            st.warning("Prediction blocked — required information is missing.")
            for m in res["missing_data"]:
                st.markdown(f"- {m}")
        else:
            m1, m2, m3 = st.columns(3)
            score = res["adaptation_score"]
            m1.metric("Adaptation score (0–100)", f"{score:.0f}")
            m2.metric("Predicted adaptation probability", f"{res['probability']:.1%}")
            m3.metric("Peer reference (nearest historical transfers)", f"{res['peer_adaptation_rate']:.0%}")

            st.progress(min(score / 100, 1.0))
            st.info(res["interpretation"])

            st.subheader("Positive factors")
            if res["positive_factors"]:
                for f in res["positive_factors"]:
                    st.markdown(f"- **{f['feature']}** (contribution +{f['contribution']:.3f})")
            else:
                st.markdown("_no strong positive contributors_")

            st.subheader("Risk factors")
            if res["risk_factors"]:
                for f in res["risk_factors"]:
                    st.markdown(f"- **{f['feature']}** (contribution {f['contribution']:.3f})")
            else:
                st.markdown("_no strong negative contributors_")

            st.subheader("Similar historical players / transfers")
            if res["similar_transfers"]:
                sim = pd.DataFrame(res["similar_transfers"])
                sim.columns = ["Player", "From", "To", "Post-transfer minutes",
                               "Adapted"]
                sim["Adapted"] = sim["Adapted"].map({1: "yes", 0: "no"})
                st.dataframe(sim.set_index("Player"))
            else:
                st.markdown("_none found_")

            st.subheader("Feature snapshot used (pre-transfer only)")
            feats = pd.DataFrame(
                [{"feature": k, "value": str(v)} for k, v in res["features_explained"].items()])
            st.dataframe(feats.set_index("feature"))

            if res["missing_data"]:
                st.caption("Data gaps reported: " + "; ".join(res["missing_data"]))

    st.caption("Note: prototype trained on 148 real intra-League transfers "
               "(La Liga 2022/23–2024/25). Scores are relative indications, "
               "not guarantees.")


with tab_val:
    st.subheader("Research comparison — time-based validation")
    try:
        comp = pd.read_csv(config.TABLES_DIR / "model_comparison.csv")
        comp = comp.drop(columns=[c for c in ("error",) if c in comp.columns],
                         errors="ignore")
        st.dataframe(comp.sort_values(["variant", "model"]))
    except FileNotFoundError:
        st.warning("Run scripts/train_evaluate.py first.")

    st.markdown("**Key result (real, computed):** adding target-club context (Model C) "
                "raises time-split ROC-AUC from ~0.60 (player-only, Model A) to ~0.68 "
                "with logistic regression; player-similarity peers (Model D) keep parity.")

    col1, col2 = st.columns(2)
    with col1:
        for name, title in [("outcome_distribution.png", "Outcome distribution"),
                            ("calibration.png", "Calibration (time-split test)"),
                            ("similarity_pca.png", "Similarity space (PCA)")]:
            p = config.FIGURES_DIR / name
            if p.exists():
                st.image(str(p), caption=title)
    with col2:
        for name, title in [("strength_scatter.png", "Source vs target club strength"),
                            ("feature_importance_C.png", "Feature importance (Model C, LR)"),
                            ("feature_importance_D.png", "Feature importance (Model D, LR)")]:
            p = config.FIGURES_DIR / name
            if p.exists():
                st.image(str(p), caption=title)

    st.subheader("Time-split test predictions (first 25)")
    try:
        preds = pd.read_csv(config.TABLES_DIR / "time_split_test_predictions.csv")
        st.dataframe(preds.head(25))
    except FileNotFoundError:
        pass


with tab_data:
    st.subheader("Data readiness report")
    rep = config.TABLES_DIR / "data_readiness_report.md"
    if rep.exists():
        st.markdown(rep.read_text(encoding="utf-8"))
    else:
        st.warning("Run scripts/build_dataset.py first.")

    st.subheader("Real Madrid transfer ledger (bridged reference)")
    lf = config.DATA_PROCESSED / "rm_ledger_bridged.csv"
    if lf.exists():
        led = pd.read_csv(lf)
        led = led[["transfer_date", "direction", "player_name", "counterpart_club_name",
                   "price", "market_value_at_transfer", "is_loan", "bridged"]]
        st.dataframe(led.sort_values("transfer_date", ascending=False).reset_index(drop=True))
        st.caption("92/105 ledger rows bridged to the LaLiga roster by name; "
                   "foreign-league performance is not covered by this prototype.")