### Ingest the raw SCMS logistics dataset, clean missing and non-numeric values, engineer shipment delay targets (⁠Scheduled Delivery Date⁠ vs. ⁠Delivered to Client Date⁠), and train dual XGBoost models (delay classification and delay days regression).###








import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
import joblib

def load_and_preprocess_scms(csv_path: str = r"SCMS_Delivery_History_Dataset.csv"):
    print("[Week 1] Loading and preprocessing SCMS Delivery History Dataset...")
    df = pd.read_csv(csv_path)

    # 1. Calculate Target: Shipment Delay Days
    sched = pd.to_datetime(df['Scheduled Delivery Date'], errors='coerce')
    deliv = pd.to_datetime(df['Delivered to Client Date'], errors='coerce')
    df['delay_days'] = (deliv - sched).dt.days

    # Drop records where dates are missing
    df = df.dropna(subset=['delay_days']).copy()
    
    # Classification target: 1 if delayed past schedule (> 0 days), else 0
    df['is_delayed'] = (df['delay_days'] > 0).astype(int)

    # 2. Clean numeric feature columns
    for col in ['Weight (Kilograms)', 'Freight Cost (USD)', 'Line Item Insurance (USD)']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill numerical nulls with median values
    num_cols = [
        'Line Item Quantity', 'Line Item Value', 'Pack Price', 'Unit Price',
        'Weight (Kilograms)', 'Freight Cost (USD)', 'Line Item Insurance (USD)'
    ]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # 3. Categorical encoding for high-impact supply chain fields
    cat_cols = ['Country', 'Shipment Mode', 'Vendor INCO Term', 'Product Group', 'Fulfill Via']
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes

    feature_cols = num_cols + cat_cols
    return df, feature_cols

def train_xgboost_models():
    df, feature_cols = load_and_preprocess_scms()
    
    X = df[feature_cols]
    y_clf = df['is_delayed']
    y_reg = df['delay_days']

    X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf
    )

    # Train Classifier: Probability of delay
    print("[Week 1] Training XGBoost Classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=(len(y_clf_train) - sum(y_clf_train)) / sum(y_clf_train),
        random_state=42,
        eval_metric='logloss'
    )
    clf.fit(X_train, y_clf_train)
    clf_preds = clf.predict(X_test)
    print(classification_report(y_clf_test, clf_preds, target_names=["On-Time", "Delayed"]))

    # Train Regressor on delayed orders to predict exact delay magnitude
    print("[Week 1] Training XGBoost Regressor for delay duration...")
    delay_mask = y_clf_train == 1
    reg = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    reg.fit(X_train[delay_mask], y_reg_train[delay_mask])

    test_delay_mask = y_clf_test == 1
    reg_preds = reg.predict(X_test[test_delay_mask])
    mae = mean_absolute_error(y_reg_test[test_delay_mask], reg_preds)
    print(f"[Week 1] Delay Duration Regressor MAE: {mae:.2f} days")

    # Save artifacts
    joblib.dump(clf, "xgb_delay_classifier.pkl")
    joblib.dump(reg, "xgb_delay_regressor.pkl")
    joblib.dump(feature_cols, "feature_columns.pkl")
    print("[Week 1] Models successfully trained and serialized.")

if __name__ == "__main__":
    train_xgboost_models()
