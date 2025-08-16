
# BigMart Sales Prediction — Reproducible Baseline

This baseline trains a few classic ML models (Ridge, Lasso, RandomForest, GradientBoosting) on the BigMart dataset, evaluates them via cross-validated RMSE (on original scale using a log1p target during training), picks the best, and writes `submission.csv`.

## Files
- `bigmart_solution.py` — main script (run this)
- `cv_results.csv` — cross-validated scores per model
- `training_summary.json` — brief metadata (best model, CV RMSE, row counts)
- `fitted_preprocessor.pkl` — trained `ColumnTransformer`
- `fitted_model.pkl` — best model (scikit-learn estimator)
- `submission.csv` — your uploadable file

## Expected Inputs
Place these files next to the script:
- `train.csv` — contains all features **plus** target column `Item_Outlet_Sales`
- `test.csv` — contains same features as train **without** the target

## Run
```bash
python bigmart_solution.py
```

## What the script does
1. **Cleans & engineers features**
   - Normalizes `Item_Fat_Content` labels (`LF`→`Low Fat`, `reg`→`Regular`, etc.).
   - Treats `Item_Visibility==0` as missing and imputes.
   - Adds `Outlet_Age = 2013 - Outlet_Establishment_Year`.
   - Adds `Item_Category` from the first two letters of `Item_Identifier`.
   - Item-level imputations for `Item_Weight` (mean) and `Item_Visibility` (median).
   - Adds `Visibility_Avg` and `Visibility_Ratio` features.

2. **Preprocesses**
   - Numeric: median impute + standardize.
   - Categorical: frequent impute + one-hot encode.

3. **Trains candidates (5-fold CV)**
   - Ridge, Lasso, RandomForest, GradientBoosting.
   - Trains on `log1p(Item_Outlet_Sales)` and reports RMSE after inverse transform.

4. **Fits the best model on full train and predicts test**
   - Writes `submission.csv` with the required three columns:
     `Item_Identifier`, `Outlet_Identifier`, `Item_Outlet_Sales`.

## Tips
- You can add XGBoost/LightGBM if available for stronger results.
- Try tuning model hyperparameters via `RandomizedSearchCV` or `GridSearchCV`.
- Consider interaction features, target encoding for high-cardinality fields, or
  grouping price bands from `Item_MRP`.
