# Student Performance Index Prediction

## Project Overview
This notebook analyzes student study habits and builds regression models to predict the `Performance Index`. It includes missing-value imputation, duplicate removal, exploratory analysis, scaling, and a comparison of regression and ensemble models.

## Dataset
- **Notebook:** `Student_performance.ipynb`
- **Data file:** `Student_Performance.csv`
- **Size:** 10,000 rows × 6 columns
- **Target:** `Performance Index`
- **Features:** hours studied, previous scores, extracurricular activities, sleep hours, and sample question papers practiced.
- **Source:** Not stated in the notebook.

## Libraries Used
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- tqdm
- XGBoost, CatBoost, LightGBM

## Work Done
- Loaded the data and inspected its structure and missing values.
- Imputed 1,572 missing `Extracurricular Activities` values with the most frequent value.
- Imputed 2,856 missing `Sample Question Papers Practiced` values with the mean.
- Removed 151 duplicate rows, leaving 9,849 rows.
- Encoded `Extracurricular Activities` as a binary feature (`No` = 0, `Yes` = 1).
- Explored the target against sample papers, previous scores, and sleep hours, and plotted a correlation heatmap.
- Found that `Sample Question Papers Practiced` had a very weak linear correlation with the target (0.0308).
- MinMax-scaled `Previous Scores`, then split the data into 80% training and 20% testing sets with `random_state=100`, giving 7,879 training rows and 1,970 test rows.
- Applied MinMax scaling to the feature matrix after the split.
- Trained 15 regressors, including linear models, KNN, SVR, tree-based models, and boosting models.
- Added simple voting, weighted voting, and a stacking regressor.
- Evaluated models with MAE, MSE, RMSE, and R², and plotted actual vs. predicted values for the best model.

## Results
- Linear Regression was the best model: test R² 0.9881, RMSE 2.1216, MAE 1.6801, and MSE 4.5014.
- Ridge (R² 0.9881), SVM (0.9881), CatBoost (0.9878), Hist Gradient Boosting (0.9875), and LightGBM (0.9875) performed almost as well.
- Lasso had the weakest result, with test R² 0.9275 and RMSE 5.2466.
- Extra Trees overfitted: training R² 0.9992 but test R² 0.9812.
- The very high R² values show that the available features explain most of the variation in the performance index.
