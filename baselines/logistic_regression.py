"""
Baseline: Logistic Regression classifier for DeFi fraud detection.

This module implements logistic regression as a non-structural baseline to
quantify the predictive power of raw node features alone, independent of any
graph topology. By comparing this model against GNN-based approaches, the
marginal contribution of neighbourhood message passing to fraud detection
performance can be isolated. Class imbalance is addressed via the 'balanced'
class weight strategy, which is equivalent to instance-level re-weighting by
inverse class frequency. This module is exclusively used for comparative baseline
evaluation and does not interact with the graph construction or inference pipeline.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .utils import load_data


def main():
    """Trains and evaluates the Logistic Regression baseline on raw node features.

    Loads the pre-split node feature matrix and binary fraud labels produced by
    the shared `load_data` utility (which applies the same temporal split as the
    HTGNN training pipeline). Trains a logistic regression classifier with L2
    regularization and balanced class weighting on the training partition, then
    reports AUPRC on the validation and test partitions.
    """
    X, y, train_mask, val_mask, test_mask = load_data()
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # 'balanced' weighting reweights samples by inverse class frequency,
    # providing a simple but effective strategy for the severe fraud imbalance
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    # Predict fraud probability from the positive class column of the probability matrix
    y_prob = clf.predict_proba(X_val)[:, 1]
    auprc = average_precision_score(y_val, y_prob)
    print(f"Logistic Regression Validation AUPRC: {auprc:.4f}")

    X_test, y_test = X[test_mask], y[test_mask]
    y_prob_test = clf.predict_proba(X_test)[:, 1]
    auprc_test = average_precision_score(y_test, y_prob_test)
    print(f"Logistic Regression Test AUPRC: {auprc_test:.4f}")


if __name__ == "__main__":
    main()
