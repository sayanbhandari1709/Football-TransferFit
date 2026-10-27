"""TRANSFERFIT - model training, time-based validation and research variants.

Research comparison (section 19 of the brief):
* Model A: player profile + pre-transfer performance
* Model B: A + source-club features
* Model C: B + target-club context
* Model D: C + player-similarity peer features

Validation is time-based: older transfers train, later transfers test.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.similarity import compute_peer_features

PRE_PERF = [
    "pre_minutes", "pre_matches", "pre_start_ratio", "pre_ga90",
    "pre_goals90", "pre_assists90", "pre_xg90", "pre_xa90",
    "pre_interceptions90", "pre_defensive_actions90", "pre_duels90",
    "pre_duels_won90", "pre_key_passes_per90", "pre_progressive_passes_per90",
    "pre_passes_to_final_third_per90", "pre_crosses_per90",
    "pre_through_passes_per90", "pre_forward_passes_per90",
    "pre_long_passes_per90", "pre_dribbles_per90", "pre_successful_dribbles_per90",
    "pre_recoveries_per90", "pre_clearances_per90", "pre_accelerations_per90",
    "pre_progressive_run_per90", "pre_touch_in_box_per90",
    "pre_received_pass_per90", "pre_shot_assists_per90", "pre_aerial_duels_per90",
    "pre_pass_accuracy", "pre_duels_won_percent", "pre_defensive_duels_won_percent",
    "pre_aerial_duels_won_percent", "pre_successful_dribbles_percent",
    "pre_shots_on_target_percent", "pre_goal_conversion_percent",
]

PROFILE = ["age_at_transfer", "height_cm", "weight_kg", "is_spanish", "foot", "position_group"]
SOURCE = ["source_squad_mv", "source_club_wins_pre", "source_minutes_share"]
TARGET = ["target_squad_mv", "target_club_wins_pre", "target_role_share",
          "squad_mv_diff", "club_wins_diff", "pre_market_value"]
SIM_FEATS = ["peer_adaptation_rate", "peer_avg_post_ga90"]

VARIANTS = {
    "A": PROFILE + PRE_PERF,
    "B": PROFILE + PRE_PERF + SOURCE,
    "C": PROFILE + PRE_PERF + SOURCE + TARGET,
    "D": PROFILE + PRE_PERF + SOURCE + TARGET + SIM_FEATS,
}

CAT_COLS = ["foot", "position_group"]


def encode_categoricals(df: pd.DataFrame, cols=CAT_COLS) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].fillna("UNKNOWN")
    return out


def build_X(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Return (already one-hot encoded) numeric feature matrix for a variant."""
    cols = [c for c in VARIANTS[variant] if c in df.columns]
    out = df[cols].copy()
    out = encode_categoricals(out)
    foot = pd.get_dummies(out["foot"], prefix="foot").astype(float) if "foot" in out else pd.DataFrame()
    role = pd.get_dummies(out["position_group"], prefix="role").astype(float) if "position_group" in out else pd.DataFrame()
    num_cols = [c for c in cols if c not in CAT_COLS and c in out.columns]
    out = pd.concat([out[num_cols].apply(pd.to_numeric, errors="coerce"), foot, role], axis=1)
    return out


def make_pipeline(model_name: str):
    if model_name == "lr":
        model = LogisticRegression(C=1.0, max_iter=4000, solver="liblinear", class_weight="balanced")
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced_subsample",
            max_features="sqrt", random_state=42)
    elif model_name == "gbm":
        model = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=10,
            min_samples_leaf=8, random_state=42)
    else:
        raise ValueError(model_name)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def time_split(df: pd.DataFrame):
    train = df[df["post_season"] <= 2024].copy()
    test = df[df["post_season"] == 2025].copy()
    return train, test


def run_variant(train: pd.DataFrame, test: pd.DataFrame, variant: str, model_name: str) -> dict:
    tr, te = train.copy(), test.copy()
    if variant == "D":
        tr, te = compute_peer_features(tr, te)
    Xtr = build_X(tr, variant)
    Xte = build_X(te, variant)
    ytr = tr["label"].astype(int)
    yte = te["label"].astype(int)

    pipe = make_pipeline(model_name)
    pipe.fit(Xtr, ytr)
    proba = pipe.predict_proba(Xte)[:, 1] if hasattr(pipe, "predict_proba") else None
    pred = (proba >= 0.5).astype(int) if proba is not None else pipe.predict(Xte)

    metrics = {
        "variant": variant,
        "model": model_name,
        "train_n": int(len(tr)),
        "test_n": int(len(te)),
        "pos_rate_test": float(yte.mean()),
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "roc_auc": roc_auc_score(yte, proba) if proba is not None and len(np.unique(yte)) > 1 else np.nan,
        "log_loss": log_loss(yte, proba) if proba is not None else np.nan,
    }
    return metrics


def run_experiment(df: pd.DataFrame, models=("lr", "rf", "gbm"), variants=("A", "B", "C", "D")):
    train, test = time_split(df)
    results = []
    for variant in variants:
        for model_name in models:
            try:
                results.append(run_variant(train, test, variant, model_name))
            except Exception as exc:  # noqa: BLE001
                results.append({"variant": variant, "model": model_name,
                                "error": str(exc)})
    return train, test, pd.DataFrame(results)


def train_final_model(df: pd.DataFrame, variant="D", model_name="lr") -> dict:
    """Fit the production model on all available transfers and persist it."""
    from src.similarity import compute_peer_features_loo

    train = df.copy()
    if variant == "D":
        train = compute_peer_features_loo(train)
    X = build_X(train, variant)
    y = train["label"].astype(int)
    pipe = make_pipeline(model_name)
    pipe.fit(X, y)

    artifact = {
        "pipeline": pipe,
        "variant": variant,
        "model_name": model_name,
        "feature_columns": list(X.columns),
        "peer_columns": [c for c in X.columns if c.startswith("peer_")],
        "train_n": len(train),
        "positive_rate": float(y.mean()),
    }
    return artifact


def save_artifact(artifact: dict, name: str) -> Path:
    path = config.MODELS_SAVED_DIR / name
    joblib.dump(artifact, path)
    return path


def load_artifact(name: str) -> dict:
    return joblib.load(config.MODELS_SAVED_DIR / name)