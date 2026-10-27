"""Train and evaluate TRANSFERFIT models (time-based validation).

Research comparison: Model A (player only) -> D (player + clubs + similarity).
Models: Logistic Regression, Random Forest, Gradient Boosting (sklearn).
Artifacts are saved to models_saved/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import config
from src.adaptation_label import apply_label
from src.model import run_experiment, train_final_model, save_artifact
from src import evaluation


def main():
    df = pd.read_csv(config.DATA_PROCESSED / "historical_transfers.csv",
                     parse_dates=["transfer_date"])
    df = apply_label(df)
    print(f"Historical transfer rows: {len(df)}")

    train, test, results = run_experiment(df)
    print(f"Time split: train n={len(train)} (<=2023/24), test n={len(test)} (2024/25)")
    print("\nRESEARCH COMPARISON (time-based validation)")
    show = results.dropna(subset=["test_n"]).sort_values(
        ["variant", "model"])
    print(show[["variant", "model", "accuracy", "precision", "recall", "f1",
                "roc_auc", "log_loss", "test_n"]].to_string(index=False))
    results.to_csv(config.TABLES_DIR / "model_comparison.csv", index=False)

    # ---- figures ------------------------------------------------------
    evaluation.outcome_plot(df, config.FIGURES_DIR / "outcome_distribution.png")
    evaluation.calibration_plot(df, config.FIGURES_DIR / "calibration.png",
                                variant="D", model="lr")
    evaluation.strength_scatter(df, config.FIGURES_DIR / "strength_scatter.png")
    for variant in ("A", "C", "D"):
        evaluation.feature_importance_plot(
            df, variant, "lr", config.FIGURES_DIR / f"feature_importance_{variant}.png")
    evaluation.similarity_plot(
        df, ["pre_ga90", "pre_xg90", "pre_minutes", "pass_accuracy",
             "successful_dribbles_percent", "duels_won_percent",
             "age_at_transfer", "position_group"],
        config.FIGURES_DIR / "similarity_pca.png")

    # ---- production artifact: explainable model with full context ------
    artifact = train_final_model(df, variant="D", model_name="lr")
    path = save_artifact(artifact, "transferfit_model.joblib")
    print(f"\nSaved production artifact: {path}")

    # small sanity: predictions on test for the artifact variant
    from src.similarity import compute_peer_features
    tr, te = compute_peer_features(train, test)
    from src.model import build_X
    Xte = build_X(te, "D")
    pipe = artifact["pipeline"]
    te_pred = te.copy()
    te_pred["pred_prob"] = pipe.predict_proba(Xte)[:, 1]
    te_pred[["player_name", "source_team_name", "target_team_name",
             "pre_minutes", "label", "pred_prob"]].to_csv(
        config.TABLES_DIR / "time_split_test_predictions.csv", index=False)
    print("\nSample test-set predictions (real computed outputs):")
    print(te_pred[["player_name", "source_team_name", "target_team_name",
                   "label", "pred_prob"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()