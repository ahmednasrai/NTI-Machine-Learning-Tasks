# Diabetes Patient Classification

## Project Overview
The notebook analyzes clinical measurements from diabetes patients and builds classification models to distinguish diabetic from non-diabetic patients. It includes data cleaning, exploratory analysis, model comparison, and ensemble learning.

## Dataset
- **Notebook:** `diabetes_patients.ipynb`
- **Data file:** `diabetes_patients_10000_full.csv`
- **Size:** 10,000 rows × 10 columns
- **Target:** `Class` (`Diabetic` / `Non-Diabetic`)
- **Features:** age, gender, blood sugar level, creatinine, BMI, urea, cholesterol, and HbA1c.
- **Source:** Not stated in the notebook.

## Libraries Used
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- tqdm
- XGBoost, CatBoost, LightGBM

## Work Done
- Loaded the data, inspected its structure, and removed `Patient_ID`.
- Checked for duplicates (none) and imputed missing numeric values with the median and missing `Gender` values with the most frequent value.
- Converted `Gender` to a binary feature (Male = 1, Female = 0).
- Explored the target distribution, feature histograms, and a correlation heatmap.
- Split the data into 80% training and 20% testing sets using stratification.
- Standardized the features with `StandardScaler`.
- Trained 13 classifiers: Logistic Regression, KNN, SVC, Decision Tree, Naive Bayes, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, Hist Gradient Boosting, XGBoost, CatBoost, and LightGBM.
- The main model loop stopped at XGBoost because the target labels were strings, so CatBoost and LightGBM were not completed in that loop.
- Built a soft-voting classifier and a stacking classifier with an XGBoost meta-model.
- Evaluated models with confusion matrices, classification reports, accuracy, balanced accuracy, ROC-AUC, and PR-AUC.

## Results
- Logistic Regression: 97% accuracy, ROC-AUC 0.9915, PR-AUC 0.9287.
- KNN: 91% accuracy, ROC-AUC 0.8473, PR-AUC 0.3093.
- SVC: 91% accuracy, ROC-AUC 0.9939, but it predicted no non-diabetic cases.
- Naive Bayes and Extra Trees: 99% accuracy.
- Decision Tree, Random Forest, AdaBoost, Gradient Boosting, and Hist Gradient Boosting: 100% accuracy, ROC-AUC, and PR-AUC on the test set.
- The voting classifier also reached 100% accuracy. The stacking classifier reached 100% accuracy, but its final PR-AUC plot failed because `pos_label` was not specified.
- The dataset is highly separable, which explains the very high scores.
