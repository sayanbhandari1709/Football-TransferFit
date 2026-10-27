"""TRANSFERFIT - evaluation, diagnostics and research-quality charts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.model import build_X, make_pipeline, time_split

sns.set_theme(style="whitegrid")


def outcome_plot(df: pd.DataFrame, path: Path | None = None):
    pdf = df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(pdf["post_minutes"], bins=20, ax=axes[0])
    axes[0].set_title("Post-transfer minutes (target club)")
    sns.countplot(data=pdf, x="label", ax=axes[1])
    axes[1].set_title("Adaptation label")
    sns.boxplot(data=pdf, x="label", y="retention_playtime", ax=axes[2])
    axes[2].set_title("Playing-time retention vs label")
    fig.tight_layout()
    if path:
        plt.savefig(path, dpi=130)
    plt.close(fig)


def fitted_probabilities(df: pd.DataFrame, variant="C", model="lr"):
    """Return per-transfer predicted probabilities on the time-split test set."""
    train, test = time_split(df)
    from src.similarity import compute_peer_features
    tr, te = compute_peer_features(train, test)
    Xtr, Xte = build_X(tr, variant), build_X(te, variant)
    pipe = make_pipeline(model)
    pipe.fit(Xtr, tr["label"].astype(int))
    out = test.copy()
    out["pred_prob"] = pipe.predict_proba(Xte)[:, 1]
    out["adaptation_score"] = (out["pred_prob"] * 100).round(1)
    out["pred_label"] = (out["pred_prob"] >= 0.5).astype(int)
    return out


def calibration_plot(df: pd.DataFrame, path=None, variant="C", model="lr"):
    out = fitted_probabilities(df, variant, model)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(0, 1, 11)
    binned = out.copy()
    binned["bin"] = pd.cut(binned["pred_prob"], bins=bins, include_lowest=True, labels=False)
    group = binned.groupby("bin", observed=True).agg(
        m=("pred_prob", "mean"), n=("label", "mean"), c=("label", "count"))
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].plot(group["m"], group["n"], "o-")
    axes[0].set_xlabel("Mean predicted probability"); axes[0].set_ylabel("Observed adaptation rate")
    axes[0].set_title("Calibration (time-split test)")
    sns.histplot(out["pred_prob"], bins=20, ax=axes[1])
    axes[1].set_title("Predicted adaptation probabilities")
    fig.tight_layout()
    if path:
        plt.savefig(path, dpi=130)
    plt.close(fig)
    return out


def feature_importance_plot(df: pd.DataFrame, variant="C", model="lr", path=None):
    train, test = time_split(df)
    if variant == "D":
        from src.similarity import compute_peer_features
        train, test = compute_peer_features(train, test)
    Xtr, Xte = build_X(train, variant), build_X(test, variant)
    pipe = make_pipeline(model)
    pipe.fit(Xtr, train["label"].astype(int))
    est = pipe.named_steps["model"]
    names = list(Xtr.columns)
    if model == "lr":
        coefs = np.abs(est.coef_[0])
        top = min(20, len(coefs))
        order = np.argsort(coefs)[::-1][:top]
        values = coefs[order]
        title = "Logistic regression |coefficient|"
    else:
        if hasattr(est, "feature_importances_"):
            imp = est.feature_importances_
            top = min(20, len(imp))
            order = np.argsort(imp)[::-1][:top]
            values = imp[order]
            title = "Ensemble feature importance"
        else:
            return None
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([names[i] for i in order], values)
    ax.invert_yaxis(); ax.set_title(title)
    fig.tight_layout()
    if path:
        plt.savefig(path, dpi=130)
    plt.close(fig)
    return [(names[i], float(values[j])) for j, i in enumerate(order)]


def strength_scatter(df: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    d = df[["source_squad_mv", "target_squad_mv", "label"]].dropna()
    sns.scatterplot(data=d, x="source_squad_mv", y="target_squad_mv",
                    hue="label", palette=["tab:red", "tab:green"], ax=ax)
    ax.plot([d.source_squad_mv.min(), d.source_squad_mv.max()],
            [d.source_squad_mv.min(), d.source_squad_mv.max()], "--", color="grey")
    ax.set_xlabel("Source club log squad value"); ax.set_ylabel("Target club log squad value")
    ax.set_title("Source vs target club strength by adaptation outcome")
    fig.tight_layout()
    if path:
        plt.savefig(path, dpi=130)
    plt.close(fig)


def similarity_plot(df: pd.DataFrame, features, path=None):
    """2D PCA of the similarity feature space coloured by outcome."""
    from sklearn.decomposition import PCA
    from src.similarity import similarity_matrix
    Xs, _, _ = similarity_matrix(df, features)
    keep = [c for c in Xs.columns if Xs[c].nunique() > 1]
    pca = PCA(n_components=2, random_state=42).fit(Xs[keep])
    comp = pca.transform(Xs[keep])
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(comp[:, 0], comp[:, 1], c=df["label"].astype(int), cmap="bwr", alpha=0.7)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Similarity space (PCA) by adaptation outcome")
    fig.colorbar(sc)
    fig.tight_layout()
    if path:
        plt.savefig(path, dpi=130)
    plt.close(fig)


def evaluation_report(results: pd.DataFrame, out_dir):
    best = results.loc[results["test_n"].notna()].sort_values("roc_auc", ascending=False)
    best.to_csv(out_dir / "model_comparison.csv", index=False)
    return best