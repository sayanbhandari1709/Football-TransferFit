"""TRANSFERFIT - player similarity.

Similarity is computed on PRE-TRANSFER player profiles (scaled). It is used to
(a) produce a "peer adaptation rate" feature (Model D) and
(b) retrieve similar historical transfers/players for the interpretation layer.

Peer features are leakage-safe: for training rows the neighbours are evaluated
leave-one-out within the training split; for test rows the neighbours are drawn
from the training split only.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SIM_FEATURES = [
    "pre_ga90", "pre_xg90", "pre_assists90", "pre_minutes",
    "pass_accuracy", "successful_dribbles_percent", "duels_won_percent",
    "pre_interceptions90", "pre_defensive_actions90", "pre_recoveries_per90",
    "age_at_transfer", "height_cm", "position_group",
]


def _is_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_object_dtype(series):
        return False
    if pd.api.types.is_datetime64_any_dtype(series):
        return False
    if pd.api.types.is_bool_dtype(series):
        return True
    return True


def build_sim_numeric(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Raw (unscaled, un-imputed) encoded similarity feature matrix."""
    base = [f for f in features if f in df.columns]
    numeric = [f for f in base if _is_numeric(df[f])]
    X = df[numeric].copy().apply(pd.to_numeric, errors="coerce")
    role_col = features if "position_group" in features else []
    role = pd.get_dummies(df["position_group"].fillna("UNKNOWN"), prefix="role").astype(float)
    if not role.empty:
        X = pd.concat([X, role], axis=1)
    return X


def scale_matrix(X: pd.DataFrame, scaler: StandardScaler | None = None):
    X = X.copy()
    med = X.median()
    X = X.fillna(med)
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return pd.DataFrame(scaler.transform(X), columns=X.columns), scaler


def similarity_matrix(df: pd.DataFrame, features: list[str]):
    """Standardise the similarity features and return scaled matrix + scaler + cols."""
    X = build_sim_numeric(df, features)
    Xs, scaler = scale_matrix(X, None)
    return Xs, scaler, list(X.columns)


def compute_peer_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 5,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add peer_adaptation_rate and peer_avg_post_ga90 to train and test."""
    features = features or SIM_FEATURES
    tr = train.reset_index(drop=True)
    te = test.reset_index(drop=True)

    Xtr_raw = build_sim_numeric(tr, features)
    med = Xtr_raw.median()
    Xtr = Xtr_raw.fillna(med)
    scaler = StandardScaler().fit(Xtr)
    cols = list(Xtr_raw.columns)

    Xte_raw = build_sim_numeric(te, features)
    Xte_raw = Xte_raw.reindex(columns=cols, fill_value=0.0)
    Xte_raw = Xte_raw.fillna(med.reindex(cols).fillna(0.0))
    Xte = pd.DataFrame(scaler.transform(Xte_raw), columns=cols, index=te.index)

    # leave-one-out peers for train
    tr_peer_rate, tr_peer_ga = [], []
    for i in range(len(tr)):
        nn = NearestNeighbors(n_neighbors=min(k, len(tr) - 1), metric="cosine").fit(
            np.delete(Xtr.values, i, axis=0)
        )
        dist, idx = nn.kneighbors(Xtr.values[[i]])
        idx = idx[0]
        tr_peer_rate.append(float(tr["label"].iloc[idx].mean()))
        tr_peer_ga.append(float(tr["post_ga90"].iloc[idx].mean()))
    tr["peer_adaptation_rate"] = tr_peer_rate
    tr["peer_avg_post_ga90"] = tr_peer_ga

    nn = NearestNeighbors(n_neighbors=min(k, len(tr)), metric="cosine").fit(Xtr.values)
    dist, idx = nn.kneighbors(Xte.values)
    te["peer_adaptation_rate"] = tr["label"].to_numpy()[idx].mean(axis=1)
    te["peer_avg_post_ga90"] = tr["post_ga90"].to_numpy()[idx].mean(axis=1)

    return tr, te


def compute_peer_features_loo(
    df: pd.DataFrame,
    k: int = 5,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Leave-one-out peer features for the full pool (used for the production fit)."""
    features = features or SIM_FEATURES
    df = df.reset_index(drop=True).copy()
    Xtr_raw = build_sim_numeric(df, features)
    med = Xtr_raw.median()
    Xtr = Xtr_raw.fillna(med)
    scaler = StandardScaler().fit(Xtr)
    X = pd.DataFrame(scaler.transform(Xtr), columns=Xtr_raw.columns)
    rate, ga = [], []
    for i in range(len(df)):
        nn = NearestNeighbors(n_neighbors=min(k, len(df) - 1), metric="cosine").fit(
            np.delete(X.values, i, axis=0))
        _, idx = nn.kneighbors(X.values[[i]])
        idx = idx[0]
        rate.append(float(df["label"].iloc[idx].mean()))
        ga.append(float(df["post_ga90"].iloc[idx].mean()))
    df["peer_adaptation_rate"] = rate
    df["peer_avg_post_ga90"] = ga
    return df


def find_similar_players(
    query: pd.Series,
    pool: pd.DataFrame,
    k: int = 6,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Return the k most similar players/transfers to the query row."""
    features = features or SIM_FEATURES
    qdf = query.to_frame().T
    merged = pd.concat([pool, qdf], ignore_index=True)
    Xs, scaler, cols = similarity_matrix(merged, features)
    q_ix = len(merged) - 1
    ref_indices = [i for i in range(len(merged)) if i != q_ix]
    ref = Xs.iloc[ref_indices]
    nn = NearestNeighbors(n_neighbors=min(k, len(ref)), metric="cosine").fit(ref.values)
    dist, idx = nn.kneighbors(Xs.iloc[[q_ix]].values)
    hits = [ref_indices[i] for i in idx[0]]
    out = pool.iloc[hits].copy()
    out["_sim_dist"] = dist[0][: len(hits)]
    return out.sort_values("_sim_dist").reset_index(drop=True)