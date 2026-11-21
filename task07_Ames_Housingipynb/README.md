# Ames Housing Sale Price Prediction

## Project Overview
This notebook analyzes the Ames Housing dataset and builds regression models to predict house sale prices. It includes missing-value handling, exploratory analysis, feature encoding, scaling, and a comparison of regression and ensemble models.

## Dataset
- **Notebook:** `Ames_Housingipynb.ipynb`
- **Data file:** `Ames Housing Dataset.csv`
- **Size:** 1,460 rows × 81 columns
- **Target:** `SalePrice`
- **Features:** 79 explanatory variables describing the lot, dwelling type, construction year, exterior, basement, heating, garage, porch, pool, and sale details. `Id` was removed.
- **Source:** Not stated in the notebook.

## Libraries Used
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- tqdm
- XGBoost, CatBoost, LightGBM

## Work Done
- Loaded the data, inspected the columns, and removed `Id`.
- Measured missing values and filled 15 categorical columns (such as `Alley`, `PoolQC`, `Fence`, and `GarageType`) with `"None"`.
- Filled `LotFrontage` with the median, `MasVnrArea` and `GarageYrBlt` with 0, and `Electrical` with the most frequent value.
- Confirmed there were no duplicate rows.
- Explored the `SalePrice` distribution with a histogram and box plot.
- Ranked numeric features by correlation with `SalePrice` and plotted a heatmap of the top 15.
- One-hot encoded all categorical features with `drop_first=True`, producing 170 model features.
- Split the data into 80% training and 20% testing sets with `random_state=100`, giving 1,168 training rows and 292 test rows.
- Applied MinMax scaling fitted on the training set.
- Trained 15 regressors, including linear models, KNN, SVR, tree-based models, and boosting models.
- Added simple voting, weighted voting, and a stacking regressor.
- Evaluated models with MAE, MSE, RMSE, and R², and plotted actual vs. predicted prices for the first 20 test samples.

## Results
- Gradient Boosting had the best test performance: R² 0.9047, RMSE 24,753.39, and MAE 15,613.41.
- CatBoost followed closely with test R² 0.9042 and RMSE 24,816.91.
- Hist Gradient Boosting reached test R² 0.8976, LightGBM 0.8945, and XGBoost 0.8926.
- The stacking regressor reached test R² 0.8730.
- Linear Regression had the weakest result, with test R² 0.0589 and RMSE 77,784.65.
- Several tree ensembles overfitted: Extra Trees scored R² 1.0000 on training data but 0.8788 on the test set, and Random Forest scored 0.9725 on training data but 0.8772 on the test set.
