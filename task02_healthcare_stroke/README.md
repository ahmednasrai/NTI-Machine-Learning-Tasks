# Healthcare Stroke Prediction

## Project Overview
This notebook studies a healthcare dataset to identify factors associated with stroke and builds classification models to predict whether a patient had a stroke. It also examines class-imbalance handling because stroke cases are rare.

## Dataset
- **Notebook:** `healthcare_stroke (1).ipynb`
- **Data file:** `stroke.csv`
- **Size:** 5,110 rows × 12 columns
- **Target:** `stroke` (0 = no stroke, 1 = stroke)
- **Features:** gender, age, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, and smoking status. The `id` column was removed.
- **Source:** Not stated in the notebook. The notebook loads `/content/healthcare-dataset-stroke-data.csv`; the repository includes the data as `stroke.csv`.

## Libraries Used
- pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- imbalanced-learn (`RandomUnderSampler`)
- tqdm
- XGBoost, CatBoost, LightGBM

## Work Done
- Loaded the data, inspected columns and types, and removed the `id` column.
- Checked for duplicates (none) and imputed the 201 missing `bmi` values with the median.
- Label-encoded the categorical features.
- Explored outliers in `age` and `avg_glucose_level` using box plots and IQR; the filtered dataframe was not used for model training.
- Visualized age distribution, stroke proportion, BMI vs. glucose level, stroke rate by smoking status, and BMI by work type.
- Split the data into 80% training and 20% testing sets.
- Trained a Logistic Regression baseline and a KNN model.
- Compared four imbalance strategies with Logistic Regression: normal training data, random under-sampling (labelled "SMOTE Over Sampling" in the notebook), balanced class weights, and custom class weights `{0: 1, 1: 10}`.
- Trained 13 classifiers, then added soft-voting and stacking classifiers.
- Evaluated models with confusion matrices, classification reports, accuracy, precision, recall, F1-score, balanced accuracy, ROC-AUC, and PR-AUC.

## Results
- The normal Logistic Regression model reached 93.93% accuracy, but it predicted only the majority class: 0 stroke cases were detected (stroke recall and F1 = 0).
- Imbalance handling improved stroke detection:
  - Random under-sampling: accuracy 73.68%, recall 75.81%, F1 25.90%
  - Balanced class weights: accuracy 74.95%, recall 75.81%, F1 26.86%
  - Custom class weights: accuracy 85.62%, recall 58.06%, F1 32.88%
- Across the 13 classifiers and ensembles, accuracy was often around 92–94%, but stroke recall and F1 remained low. The highest PR-AUC was about 0.26 (Logistic Regression), showing that the minority stroke class was still difficult to predict.
