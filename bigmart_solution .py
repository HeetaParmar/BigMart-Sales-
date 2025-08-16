#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BigMart Sales Prediction – Baseline, Reproducible Pipeline
---------------------------------------------------------
Reads train.csv and test.csv, performs cleaning + feature engineering,
runs several regression models with cross-validated RMSE,
selects the best model, and writes submission.csv.
"""

import os
import math
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ---------------------- Utility ----------------------
RANDOM_STATE = 42
N_SPLITS = 5
DATA_YEAR = 2013  # as per problem statement

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))

def log1p_transform(y):
    return np.log1p(y)

def expm1_transform(y):
    return np.expm1(y)

# ---------------------- Load ----------------------
train_path = r"C:\Users\DELL\Downloads\train_v9rqX0R.csv"
test_path  = r"C:\Users\DELL\Downloads\test_AbJTz2l.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError(
        "Missing 'train.csv' and/or 'test.csv'. Place them next to this script."
    )

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

if "Item_Outlet_Sales" not in train.columns:
    raise ValueError("train.csv must include 'Item_Outlet_Sales' target column.")

# ---------------------- Basic Cleaning ----------------------
def normalize_fat(x: pd.Series) -> pd.Series:
    repl = {
        "LF": "Low Fat",
        "low fat": "Low Fat",
        "Low Fat": "Low Fat",
        "reg": "Regular",
        "Regular": "Regular"
    }
    return x.replace(repl)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Item_Fat_Content" in out.columns:
        out["Item_Fat_Content"] = normalize_fat(out["Item_Fat_Content"].astype(str))
    if "Item_Visibility" in out.columns:
        out["Item_Visibility"] = out["Item_Visibility"].replace(0, np.nan)
    if "Outlet_Establishment_Year" in out.columns:
        out["Outlet_Age"] = DATA_YEAR - out["Outlet_Establishment_Year"]
    if "Item_Identifier" in out.columns:
        out["Item_Category"] = out["Item_Identifier"].str[:2]
    return out

train = add_features(train)
test  = add_features(test)

def impute_item_weight(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Item_Weight" in out.columns and "Item_Identifier" in out.columns:
        item_means = out.groupby("Item_Identifier")["Item_Weight"].transform("mean")
        out["Item_Weight"] = out["Item_Weight"].fillna(item_means)
    return out

def impute_item_visibility(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Item_Visibility" in out.columns and "Item_Identifier" in out.columns:
        item_medians = out.groupby("Item_Identifier")["Item_Visibility"].transform("median")
        out["Item_Visibility"] = out["Item_Visibility"].fillna(item_medians)
    return out

train = impute_item_weight(train)
test  = impute_item_weight(test)
train = impute_item_visibility(train)
test  = impute_item_visibility(test)

def add_visibility_ratio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Item_Identifier" in out.columns and "Item_Visibility" in out.columns:
        item_avg = out.groupby("Item_Identifier")["Item_Visibility"].transform("mean")
        out["Visibility_Avg"] = item_avg
        out["Visibility_Ratio"] = out["Item_Visibility"] / (item_avg.replace(0, np.nan))
    return out

train = add_visibility_ratio(train)
test  = add_visibility_ratio(test)

# ---------------------- Features / Target ----------------------
target = "Item_Outlet_Sales"
y = train[target].copy()
X = train.drop(columns=[target])

num_cols = [c for c in X.columns if X[c].dtype != "object"]
cat_cols = [c for c in X.columns if X[c].dtype == "object"]

# ---------------------- Preprocessor ----------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # fixed here
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# ---------------------- Models ----------------------
candidates = {
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=20000),
    "RandomForest": RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_split=4,
        min_samples_leaf=1, n_jobs=-1, random_state=RANDOM_STATE
    ),
    "GradientBoosting": GradientBoostingRegressor(
        random_state=RANDOM_STATE, learning_rate=0.05, n_estimators=800, max_depth=3
    )
}

def make_pipeline(model):
    return Pipeline([("prep", preprocessor), ("model", model)])

def cv_rmse_on_original_scale(model, X, y, n_splits=N_SPLITS, seed=RANDOM_STATE):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmses = []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        pipe = make_pipeline(model)
        y_tr_log = log1p_transform(y_tr)
        pipe.fit(X_tr, y_tr_log)

        y_pred_log = pipe.predict(X_va)
        y_pred = expm1_transform(y_pred_log)
        y_pred = np.clip(y_pred, 0, None)
        rmses.append(rmse(y_va, y_pred))

    return float(np.mean(rmses)), [float(r) for r in rmses]

cv_rows = []
best_name = None
best_score = float("inf")
best_model = None

for name, model in candidates.items():
    mean_rmse, folds = cv_rmse_on_original_scale(model, X, y)
    cv_rows.append({"model": name, "cv_rmse": mean_rmse, "fold_rmses": folds})
    if mean_rmse < best_score:
        best_score = mean_rmse
        best_name = name
        best_model = model

cv_df = pd.DataFrame(cv_rows).sort_values("cv_rmse")
cv_df.to_csv("cv_results.csv", index=False)
print("Cross-validated RMSEs:")
print(cv_df.to_string(index=False))
print(f"\nBest model: {best_name} (CV RMSE={best_score:.4f})")

# ---------------------- Fit Best on Full Train ----------------------
best_pipeline = make_pipeline(best_model)
y_log = log1p_transform(y)
best_pipeline.fit(X, y_log)

joblib.dump(best_pipeline.named_steps["prep"], "fitted_preprocessor.pkl")
joblib.dump(best_pipeline.named_steps["model"], "fitted_model.pkl")

# ---------------------- Predict Test & Write Submission ----------------------
test_pred_log = best_pipeline.predict(test)
test_pred = expm1_transform(test_pred_log)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({
    "Item_Identifier": test["Item_Identifier"],
    "Outlet_Identifier": test["Outlet_Identifier"],
    "Item_Outlet_Sales": test_pred
})

submission.to_csv("submission.csv", index=False)
print("\nWrote submission.csv")

summary = {
    "best_model": best_name,
    "cv_rmse": round(best_score, 6),
    "n_splits": N_SPLITS,
    "rows_train": int(len(train)),
    "rows_test": int(len(test))
}
with open("training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved training_summary.json and cv_results.csv")
