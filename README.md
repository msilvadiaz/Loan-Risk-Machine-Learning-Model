# Loan Risk Machine Learning Model

### Project Overview

This project analyzes LendingClub accepted loan data to understand borrower and loan characteristics associated with loan default risk. The main goal is to predict whether a loan will be **Fully Paid** or **Charged Off** using selected borrower, credit, and loan-related features.

The target variable is `loan_status`, which is converted into a binary classification target:

- `0` = Fully Paid
- `1` = Charged Off

Because charged-off loans represent the minority class, this is an imbalanced binary classification problem. For that reason, the project focuses on metrics such as precision, recall, F1-score, average precision, and threshold optimization instead of relying only on accuracy.

## Dataset & Processing

The project uses a LendingClub accepted-loans sample dataset from Kaggle.

Original source:

https://www.kaggle.com/datasets/wordsforthewise/lending-club

Dataset is initially processed into a manageable sub-dataset with

```
df = pd.read_csv("accepted_2007_to_2018Q4.csv", low_memory=False)

# Keeps only clear final loan outcomes
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()

# Creates target variable
df["default"] = (df["loan_status"] == "Charged Off").astype(int)

# Creates a 500k stratified sample
df_sample, _ = train_test_split(
    df,
    train_size=500000,
    stratify=df["default"],
    random_state=42
)

df_sample.to_csv("accepted_loans_sample_500k.csv", index=False)

df_sample["default"].value_counts(normalize=True)
```
The code snippet takes the original LendingClub dataset, removes loans without final outcomes, creates a default target column, then creates a smaller 500,000-row sample while preserving the same default/non-default balance. Only loans with final outcomes of `Fully Paid` or `Charged Off` are used in the analysis.

### Selected Features

The notebook uses a selected group of features that are reasonable for predicting loan risk and avoiding obvious data leakage.

The selected features include:

- Loan characteristics:
  - `loan_amnt`
  - `term`
  - `installment`
  - `purpose`

- Borrower profile:
  - `emp_length`
  - `home_ownership`
  - `annual_inc`
  - `verification_status`
  - `dti`

- Credit score and credit history:
  - `fico_range_low`
  - `earliest_cr_line`
  - `delinq_2yrs`
  - `inq_last_6mths`
  - `mths_since_last_delinq`
  - `mths_since_last_record`

- Account and balance information:
  - `open_acc`
  - `pub_rec`
  - `revol_bal`
  - `revol_util`
  - `total_acc`
  - `mort_acc`
  - `acc_open_past_24mths`
  - `bc_util`

An additional risk-related feature `loan_to_income` is also feature engineered.

Some features like `int_rate` and `grade`, are useful for descriptive analysis but are removed before modeling because they may contain LendingClub's own risk assessment and could make the model less independent.

### Data Cleaning

The notebook performs several cleaning steps before modeling.

The main cleaning steps include:

- Filtering the dataset to keep only final loan outcomes:
  - `Fully Paid`
  - `Charged Off`

- One-Hot Encoding the target variable:
  - Fully Paid = `0`
  - Charged Off = `1`

- Converting text-based columns into numeric values:
  - `term`
  - `emp_length`
  - `revol_util`

- Creating an employment-length missingness flag:
  - `emp_length_missing`

- Converting `earliest_cr_line` into a new feature:
  - `credit_history_years`

- Replacing infinite values with missing values where needed.

## Exploratory Data Analysis
The notebook includes several charts and summary tables to understand the data before modeling.

### Status Count

| Loan Status | Count | Percentage |
|---|---:|---:|
| Fully Paid | 400,187 | 80.04% |
| Charged Off | 99,813 | 19.96% |

- Charged-off loans represent about 20% of the dataset.

### Interest rate & FICO Distributions

<img width="1359" height="419" alt="image" src="https://github.com/user-attachments/assets/14ef3475-da8d-4feb-afdd-31a286bd1a7d" />

- FICO Score Distribution: Charged-off loans are more concentrated at lower FICO scores, suggesting that FICO factors like lower credit scores are associated with higher default risk.
- Interest Rate Distribution: Charged-off loans appear more common at higher interest rates, showcasing that higher-risk loans were generally assigned higher borrowing costs.

### Default Rate by Loan Grade

<img width="690" height="490" alt="image" src="https://github.com/user-attachments/assets/6f252d33-9f32-42e8-bf28-772038c7c394" />

- Default risk increases clearly as loan grade worsens from A to G, showing that lower-quality grades are strongly associated with higher charge-off rates.

### Default rate by Loan Purpose

<img width="670" height="400" alt="image" src="https://github.com/user-attachments/assets/58d6b120-8e5e-4613-b235-4c3c786a487b" />

- Default risk varies by loan purpose, with small business loans having the highest charge-off rate and educational, wedding, and car loans having some of the lowest rates.

### Correlation Heatmap of Numeric Features

<img width="1311" height="989" alt="image" src="https://github.com/user-attachments/assets/9e688685-7498-430e-b4ce-b7de1f66b526" />

- The heatmap shows relationships between numeric features, with stronger correlations appearing between related variables such as ```loan_amnt``` and ```installment```, while most features have only weak-to-moderate relationships with ```loan_status```.

## Predictive Modeling Approach

The predictive analysis follows a structured workflow:

### Train/Test Split

The data is split into training and test sets using an 80/20 split.

Stratification is used so that the proportion of fully paid and charged-off loans remains similar in both the training and test sets.

### Train-Only Preprocessing

To avoid data leakage, preprocessing is fitted only on the training set and then applied to the test set.

The preprocessing steps include:

### Outlier Capping

Selected numeric features are capped using the 1st and 99th percentiles calculated from the training data only.

This reduces the effect of extreme values without removing rows from the dataset.

### Imputation

Missing numeric values are filled using training-set medians.

Missing categorical values are filled using training-set modes.

### One-Hot Encoding

Categorical variables are converted into dummy variables using one-hot encoding. The test set is then aligned to the training set so both datasets have the same columns.

## Class Imbalance Handling

The dataset has significantly more fully paid loans than charged-off loans.

To address this, the models use class imbalance handling. For XGBoost, `scale_pos_weight` is calculated as:

```python
negative_class_count / positive_class_count
```

This gives more weight to the minority positive class, which represents charged-off loans.

## Models Tested

The project tests three tuned machine learning models:

1. Logistic Regression
2. XGBoost
3. Random Forest

Each model is tuned using 3-fold stratified cross-validation.

The tuning process optimizes F1-score for the charged-off class because this metric balances precision and recall for the minority class.

### Probability Calibration

XGBoost and Random Forest probabilities are calibrated using isotonic calibration.

This is done because tree-based models can produce probability estimates that are not well calibrated. Since the project uses threshold optimization, better probability calibration helps make the selected threshold more meaningful.

### Threshold Optimization

Instead of using the default threshold of `0.50`, the notebook tests thresholds from `0.10` to `0.90`.

For each model, the threshold that gives the best F1-score for charged-off loans is selected.

This is important because in an imbalanced classification problem, the default threshold may not produce the best balance between precision and recall.

### Model Evaluation

The models are compared using the following metrics:

- Accuracy
- Precision for charged-off loans
- Recall for charged-off loans
- F1-score for charged-off loans
- Average precision
- Brier score

## Final Model Report

The best-performing model is tuned Logistic Regression.

| Model | Threshold | Accuracy | Precision Defaults | Recall Defaults | F1 Defaults | Average Precision | Brier Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.54 | 0.6898 | 0.3423 | 0.6011 | 0.4362 | 0.3839 | 0.2150 |
| XGBoost  | 0.23 | 0.6871 | 0.3392 | 0.5981 | 0.4329 | 0.3824 | 0.1440 |
| Random Forest  | 0.21 | 0.6581 | 0.3235 | 0.6532 | 0.4327 | 0.3803 | 0.1442 |


## Confusion Matrix Interpretation
Using Logistic Regression (highest F1) for a confusion matrix provides the following insights:

- The model correctly identifies 11,999 charged-off loans.
- The model misses 7,964 charged-off loans.
- The model incorrectly flags 23,051 fully paid loans as charged-off.
- The model catches about 60% of actual charged-off loans.

## Conclusion

This project developed a complete loan default prediction workflow using LendingClub accepted-loan data. The analysis included data cleaning, feature engineering, descriptive analysis, train-only preprocessing, model tuning, class imbalance handling, probability calibration, threshold optimization, and final business interpretation.

The final Logistic Regression model achieved the best F1-score for charged-off loans. While the model does not perfectly separate risky and safe loans, it provides useful predictive signal and demonstrates how machine learning can support loan risk analysis.
