# Laptop Price Prediction

## Project Overview
This notebook analyzes laptop specifications and builds regression models to predict laptop prices. It covers missing-value handling, exploratory analysis, feature engineering, encoding, scaling, and a comparison of regression and ensemble models.

## Dataset
- **Notebook:** `laptop_price.ipynb`
- **Data file:** `laptopData.csv`
- **Size:** 1,303 rows × 12 columns
- **Target:** `Price`
- **Features:** company, laptop type, screen size, screen resolution, CPU, RAM, storage, GPU, operating system, and weight. The `Unnamed: 0` index column was removed.
- **Source:** Not stated in the notebook.

## Libraries Used
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- tqdm
- XGBoost, CatBoost, LightGBM

## Work Done
- Loaded the data, removed the `Unnamed: 0` column, and found 30 rows with missing values in every column.
- Imputed the categorical columns with their most frequent values and dropped the 30 rows with no `Price`.
- Explored the price distribution, the top 10 laptop companies, price by RAM size, the proportion of laptop types, and price vs. weight.
- Removed 28 price outliers with the IQR rule, leaving 1,245 rows.
- One-hot encoded `Company`, `TypeName`, and `OpSys`.
- Extracted numeric values from `Ram` and `Weight`, and label-encoded `ScreenResolution`, `Cpu`, `Memory`, and `Gpu`, producing 39 columns.
- Split the data into 80% training and 20% testing sets with `random_state=100`, giving 996 training rows and 249 test rows.
- Applied MinMax scaling fitted on the training set.
- Trained 15 regressors, including linear models, KNN, SVR, tree-based models, and boosting models.
- Added simple voting, weighted voting, and a stacking regressor.
- Evaluated models with MAE, MSE, RMSE, and R², and plotted actual vs. predicted prices for the best model.

## Results
- Hist Gradient Boosting was the best model: test R² 0.8488, RMSE 13,240.30, MAE 9,046.30, and MSE 175,305,508.94.
- LightGBM followed with test R² 0.8467 and RMSE 13,331.50.
- XGBoost reached test R² 0.8392, Gradient Boosting 0.8321, and Stacking 0.8224.
- SVM had the weakest result, with test R² 0.2902 and RMSE 28,687.48.
- Extra Trees overfitted heavily: training R² 0.9992 but test R² 0.8109.
