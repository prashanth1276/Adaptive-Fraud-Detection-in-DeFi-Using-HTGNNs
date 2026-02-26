"""
Baseline: XGBoost gradient-boosted tree classifier for DeFi fraud detection.

This module implements XGBoost as a non-structural, tree-based baseline to
benchmark the performance of gradient-boosted ensemble methods against GNN-based
approaches. XGBoost operates exclusively on tabular node features without any
graph topology awareness, providing a strong non-graph baseline commonly used
in production fraud detection systems. Class imbalance is handled via the
`scale_pos_weight` parameter, which upweights the minority fraud class by the
negative-to-positive training set ratio. This module is used solely for
comparative baseline evaluation.
"""

import xgboost as xgb
from sklearn.metrics import average_precision_score

from .utils import load_data


def main():
    """Trains and evaluates the XGBoost baseline on raw node features.

    Loads the pre-split node feature matrix and binary fraud labels using the
    same temporal split as the HTGNN training pipeline. Trains an XGBoost
    classifier with log-loss evaluation and class-imbalance-adjusted positive
    sample weighting, then reports AUPRC on validation and test partitions.
    """
    X, y, train_mask, val_mask, test_mask = load_data()
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # Compute the negative-to-positive class ratio to adjust the decision boundary
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    # Extract fraud class probability from the positive column of the output matrix
    y_prob = model.predict_proba(X_val)[:, 1]
    auprc = average_precision_score(y_val, y_prob)
    print(f"XGBoost Validation AUPRC: {auprc:.4f}")

    X_test, y_test = X[test_mask], y[test_mask]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    auprc_test = average_precision_score(y_test, y_prob_test)
    print(f"XGBoost Test AUPRC: {auprc_test:.4f}")


if __name__ == "__main__":
    main()
