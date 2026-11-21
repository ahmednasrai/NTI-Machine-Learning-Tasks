# Adult Census Income Classification

## Project Overview
This notebook analyzes census data and builds classification models to predict whether a person earns more than 50K per year. It covers data cleaning, label encoding, outlier filtering, class balancing with SMOTE, exploratory analysis, and a comparison of classification and ensemble models.

## Dataset
- **Notebook:** `Adult_Census_Income.ipynb`
- **Data file:** `Adult Census Income.csv`
- **Size:** 32,561 rows × 15 columns
- **Target:** `income` (`<=50K` / `>50K`)
- **Features:** age, workclass, fnlwgt, education, education number, marital status, occupation, relationship, race, sex, capital gain, capital loss, hours per week, and native country.
- **Source:** Not stated in the notebook.

## Libraries Used
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- imbalanced-learn (`SMOTE`)
- tqdm
- XGBoost, CatBoost, LightGBM

## Work Done
- Loaded the data and inspected its structure; the dataset uses `?` instead of standard missing values.
- Filled the `?` entries in `workclass`, `occupation`, and `native.country` with the most frequent value.
- Removed 24 duplicate rows, leaving 32,537 rows.
- Label-encoded all categorical columns, including the target (`<=50K` = 0, `>50K` = 1).
- Applied the IQR rule to all numeric columns after encoding, which removed 21,762 rows and left 10,775 rows.
- Explored the income distribution, age distribution, hours worked by income, education by income, and age vs. hours worked.
- Split the data into 80% training and 20% testing sets with `random_state=42`, giving 8,620 training rows and 2,155 test rows.
- Standardized the numeric features.
- Applied SMOTE to the training set, increasing it to 13,482 rows with 6,741 rows in each class.
- Compared Logistic Regression, Naive Bayes, and a binarized Linear Regression baseline.
- Trained 13 classifiers, then added a soft-voting classifier and a stacking classifier with an XGBoost meta-model.
- Evaluated models with accuracy, precision, recall, F1-score, balanced accuracy, ROC-AUC, and PR-AUC, plus confusion matrices and ROC/PR curves.

## Results
- The best test F1 score came from Hist Gradient Boosting: F1 0.6418, accuracy 0.8260, precision 0.5884, recall 0.7059, ROC-AUC 0.8794, and PR-AUC 0.6547.
- Gradient Boosting followed with test F1 0.6367 and accuracy 0.8009.
- CatBoost had the highest test accuracy (0.8339) with F1 0.6216 and ROC-AUC 0.8830.
- AdaBoost had the highest test balanced accuracy (0.7985) with F1 0.6260.
- Decision Tree had the lowest test F1 (0.5050) and ROC-AUC (0.6846).
- The confusion-matrix and classification-report labels say "Not Stroke" and "Stroke" instead of the income classes, apparently copied from another notebook; the reported metrics still refer to the income target.
