# Car Price Prediction

## Project Overview
This notebook explores a car dataset and builds regression models to predict car prices from vehicle specifications. It compares a wide range of regression algorithms and ensemble methods.

## Dataset
- **Notebook:** `cars.ipynb`
- **Data file:** `cars_data.csv`
- **Size:** 1,000 rows × 8 columns
- **Target:** `Price`
- **Features:** brand, year, mileage, engine size, fuel type, transmission, and previous owners.
- **Source:** Not stated in the notebook.

## Libraries Used
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- tqdm
- XGBoost, CatBoost, LightGBM

## Work Done
- Loaded the data and inspected its structure, missing values, and duplicates.
- Imputed the 20 missing `Engine_Size` values with the median.
- Confirmed there were no duplicate rows.
- Explored numeric distributions, categorical value counts, and a correlation heatmap.
- Counted outliers using the IQR rule; 49 price outliers were detected but not removed.
- Built a preprocessing pipeline with median imputation and standard scaling for numeric features, and most-frequent imputation and one-hot encoding for categorical features.
- Transformed the data from 7 input columns to 18 processed features.
- Split the data into 80% training and 20% testing sets.
- Trained 15 regressors, including linear models, KNN, SVR, tree-based models, and boosting models.
- Added simple voting, weighted voting, and stacking regressors.
- Evaluated models with MAE, MSE, RMSE, and R², and plotted actual vs. predicted prices for the first 20 test samples.

## Results
- The notebook compares 15 regressors and ensemble models and produces R² and error comparison charts.
- The final metrics table was not printed in the saved notebook, so exact model scores cannot be reported.
