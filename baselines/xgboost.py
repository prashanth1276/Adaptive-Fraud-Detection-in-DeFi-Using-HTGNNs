import xgboost as xgb
from sklearn.metrics import average_precision_score

from .utils import load_data


def main():
    X, y, train_mask, val_mask, test_mask = load_data()
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_val)[:, 1]
    auprc = average_precision_score(y_val, y_prob)
    print(f"XGBoost Validation AUPRC: {auprc:.4f}")

    X_test, y_test = X[test_mask], y[test_mask]
    y_prob_test = model.predict_proba(X_test)[:, 1]
    auprc_test = average_precision_score(y_test, y_prob_test)
    print(f"XGBoost Test AUPRC: {auprc_test:.4f}")


if __name__ == "__main__":
    main()
