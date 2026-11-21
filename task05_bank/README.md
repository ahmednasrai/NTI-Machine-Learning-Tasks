# Bank Deposit Subscription Prediction

## Project Overview
The notebook analyzes a bank marketing dataset and builds classification models to predict whether a customer will subscribe to a term deposit. It includes data cleaning, exploratory analysis, model tuning, feature selection, and ensemble learning.

## Dataset
- **Notebook:** `bank.ipynb`
- **Data file:** `bank.csv`
- **Size:** 8,912 rows × 17 columns
- **Target:** `deposit` (`yes` / `no`)
- **Features:** customer age, job, marital status, education, default status, balance, housing loan, personal loan, contact type, day, month, call duration, campaign, previous contacts, and previous outcome.
- **Source:** Not stated in the notebook.

## Libraries Used
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- tqdm
- XGBoost, CatBoost, LightGBM

## Work Done
- Loaded the data and inspected its structure and missing values.
- Imputed numeric columns using the mean when skewness was between -0.5 and 0.5, otherwise the median.
- Imputed categorical columns with the most frequent value.
- Confirmed there were no duplicate rows.
- Explored the target balance, numeric distributions, job vs. deposit counts, and a correlation heatmap.
- Split the data into 80% training and 20% testing sets using stratification.
- One-hot encoded categorical features and standardized numeric features.
- Trained a Logistic Regression baseline and an SVC model.
- Tuned SVC with `GridSearchCV`; the best parameters were `C=1`, `gamma='scale'`, and `kernel='rbf'`.
- Tuned a Decision Tree; the best parameters were `max_depth=10`, `min_samples_leaf=5`, and `min_samples_split=2`, with a best cross-validation score of 0.8113.
- Used a Random Forest to select the top five features: `duration`, `balance`, `age`, `day`, and `campaign`.
- Trained a set of 13 classifiers, then added a soft-voting classifier and a stacking classifier.
- The main model loop stopped at XGBoost because the target labels were strings, so XGBoost, CatBoost, and LightGBM did not complete in that loop.
- Evaluated models with accuracy, precision, recall, F1-score, balanced accuracy, ROC-AUC, and PR-AUC.

## Results
- Logistic Regression baseline: accuracy 0.8149, precision 0.8071, recall 0.8014, F1 0.8043.
- Random Forest with the top five features: 76% accuracy.
- In the final comparison table, Gradient Boosting had the best test F1 score (0.7672) and test accuracy (0.7757), with ROC-AUC 0.8486 and PR-AUC 0.8020.
- Hist Gradient Boosting followed with test accuracy 0.7723 and F1 0.7659.
- The soft-voting classifier reached test accuracy 0.7723 and F1 0.7506.
- Tree-based boosting models performed better than the linear and distance-based models on this dataset.
