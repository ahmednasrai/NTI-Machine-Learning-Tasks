# Apartment Price Prediction

## Project Overview
This notebook analyzes apartment listing data and builds regression models to predict apartment prices from property characteristics. It covers data cleaning, exploratory analysis, preprocessing, and a comparison of many regression and ensemble models.

## Dataset
- **Notebook:** `Apartments_Prices.ipynb`
- **Data file:** `Apartments Prices Dataset.csv`
- **Size:** 5,578 rows × 11 columns
- **Target:** `price`
- **Features:** area, number of rooms, number of bathrooms, floor level, year built, seller type, view, price per square metre, payment method, and finishing type. Prices and several property fields are stored as text with units such as `EGP` and `m²`.
- **Source:** Not stated in the notebook.

## Libraries Used
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- tqdm
- XGBoost, CatBoost, LightGBM

## Work Done
- Loaded the data and inspected its structure; there were no missing values.
- Removed 1,489 duplicate rows, leaving 4,089 rows.
- Explored numeric distributions, categorical value counts, and a correlation heatmap.
- Checked the numeric columns with the IQR rule; no outliers were removed at this stage.
- Standardized the numeric feature (`year_built`) and one-hot encoded `seller_type`, `view`, `payment_method`, and `finishing_type`.
- Split the data into 80% training and 20% testing sets with `random_state=42`, giving 3,271 training rows and 818 test rows.
- Trained 15 regressors: Linear Regression, Lasso, Ridge, KNN, SVR, Decision Tree, Random Forest, Extra Trees, Bagging, AdaBoost, Gradient Boosting, Hist Gradient Boosting, XGBoost, CatBoost, and LightGBM.
- Added simple voting, weighted voting, and stacking regressors; the stacking model used Random Forest, Gradient Boosting, CatBoost, and LightGBM as base models with XGBoost as the meta-model.
- Evaluated models with MAE, MSE, RMSE, and R², and plotted actual vs. predicted prices for the first 20 test samples.

## Results
- The notebook builds a comparison table sorted by test R² and plots R², MAE, MSE, and RMSE comparisons, but the saved output does not display the final metrics table. The best model and exact scores therefore cannot be confirmed from the notebook.
- The saved outputs show all models training, but the code stored in the notebook does not convert the text-formatted columns (`price`, `area_sqm`, `rooms`, `bathrooms`, `floor_level`, and `price_per_sqm`) to numeric values before fitting. As a result, the saved results cannot be reproduced from the notebook code as-is.
