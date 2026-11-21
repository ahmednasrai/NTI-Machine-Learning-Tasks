# House Rent Price Prediction

## Project Overview
The notebook explores a house-rent dataset and builds regression models to estimate monthly rent from property characteristics. It covers data cleaning, exploratory analysis, preprocessing, and comparison of several regression and ensemble models.

## Dataset
- **Notebook:** `House_Rent.ipynb`
- **Data file:** `House_Rent_Dataset.csv`
- **Size:** 4,746 rows × 12 columns before cleaning
- **Target:** `Rent`
- **Features:** BHK, size, floor, area type, area locality, city, furnishing status, tenant preference, bathroom count, and point of contact. `Posted On` was removed.
- **Source:** Not stated in the notebook.

## Libraries Used
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- SciPy
- tqdm
- XGBoost, CatBoost, LightGBM

## Work Done
- Loaded the CSV, inspected its structure, and removed the `Posted On` column.
- Removed 8 duplicate rows and 5 rows with missing rent values.
- Imputed missing `Size` values with the median, `Area Type` with the most frequent value, and `Bathroom` with the mean.
- Removed rent outliers using a z-score threshold of 3, reducing the data from 4,733 to 4,667 rows.
- Encoded categorical columns and explored the data with histograms, scatter plots, bar charts, and a BHK pie chart.
- Built a preprocessing pipeline with median imputation and scaling for numeric features, and most-frequent imputation and one-hot encoding for categorical features.
- Split the data into 80% training and 20% testing sets.
- Trained a Random Forest regressor and compared 15 regressors, simple/weighted voting regressors, and a stacking regressor using MAE, MSE, RMSE, and R².

## Results
- The Random Forest baseline (200 trees) achieved:
  - MAE: 8,347.91
  - MSE: 328,575,216.34
  - RMSE: 18,126.64
  - R²: 0.7634
- The notebook also produced a comparison of all models and actual-vs-predicted plots, but the saved output does not show the final metric table, so the best model score cannot be confirmed from the notebook.
