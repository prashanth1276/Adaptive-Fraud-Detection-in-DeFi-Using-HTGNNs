from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .utils import load_data


def main():
    X, y, train_mask, val_mask, test_mask = load_data()
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_val)[:, 1]
    auprc = average_precision_score(y_val, y_prob)
    print(f"Logistic Regression Validation AUPRC: {auprc:.4f}")

    X_test, y_test = X[test_mask], y[test_mask]
    y_prob_test = clf.predict_proba(X_test)[:, 1]
    auprc_test = average_precision_score(y_test, y_prob_test)
    print(f"Logistic Regression Test AUPRC: {auprc_test:.4f}")


if __name__ == "__main__":
    main()
