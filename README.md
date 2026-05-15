# Loan Risk Machine Learning Model

## Project Overview

This project analyzes LendingClub accepted loan data to understand borrower and loan characteristics associated with loan default risk. The main goal is to predict whether a loan will be **Fully Paid** or **Charged Off** using selected borrower, credit, and loan-related features.

The target variable is `loan_status`, which is converted into a binary classification target:

- `0` = Fully Paid
- `1` = Charged Off

Because charged-off loans represent the minority class, this is an imbalanced binary classification problem. For that reason, the project focuses on metrics such as precision, recall, F1-score, average precision, and threshold optimization instead of relying only on accuracy.

## Dataset

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

The working dataset contains 500,000 sampled loan records. Only loans with final outcomes of `Fully Paid` or `Charged Off` are used in the analysis.

## Selected Features

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

Some features, such as `int_rate` and `grade`, are useful for descriptive analysis but are removed before modeling because they may contain LendingClub's own risk assessment and could make the model less independent.

## Data Cleaning

The notebook performs several cleaning steps before modeling.

The main cleaning steps include:

- Filtering the dataset to keep only final loan outcomes:
  - `Fully Paid`
  - `Charged Off`

- Encoding the target variable:
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

## Feature Engineering

The notebook creates an additional risk-related feature.

### `loan_to_income`

This feature divides the loan amount by the borrower's annual income:

```python
loan_to_income = loan_amnt / annual_inc
```

This helps capture how large the loan is relative to the borrower's income. A higher value may indicate higher financial pressure and higher default risk.

## Descriptive Analysis

The notebook includes several charts and summary tables to understand the data before modeling.

The descriptive analysis includes:

- Target distribution between fully paid and charged-off loans
- Loan amount distribution by loan status
- Interest rate distribution by loan status
- Default rate by loan grade
- FICO score distribution by loan status
- Default rate by loan purpose
- Correlation heatmap of numeric features

Key insights include:

- Charged-off loans represent about 20% of the dataset.
- Default risk increases as loan grade worsens.
- Small business loans show one of the highest default rates by purpose.
- Lower credit scores are more common among charged-off loans.
- Some financial and credit-related features show useful relationships with default risk.

## Predictive Modeling Approach

The predictive analysis follows a structured machine learning workflow.

The main steps are:

1. Split the data into training and test sets.
2. Apply preprocessing using training data only.
3. Handle outliers using percentile capping.
4. Impute missing values.
5. One-hot encode categorical variables.
6. Tune selected machine learning models using cross-validation.
7. Handle class imbalance.
8. Calibrate probabilities for selected models.
9. Optimize the classification threshold.
10. Evaluate final performance on the test set.

## Train/Test Split

The data is split into training and test sets using an 80/20 split.

Stratification is used so that the proportion of fully paid and charged-off loans remains similar in both the training and test sets.

## Train-Only Preprocessing

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

## Probability Calibration

XGBoost and Random Forest probabilities are calibrated using isotonic calibration.

This is done because tree-based models can produce probability estimates that are not well calibrated. Since the project uses threshold optimization, better probability calibration helps make the selected threshold more meaningful.

## Threshold Optimization

Instead of using the default threshold of `0.50`, the notebook tests thresholds from `0.10` to `0.90`.

For each model, the threshold that gives the best F1-score for charged-off loans is selected.

This is important because in an imbalanced classification problem, the default threshold may not produce the best balance between precision and recall.

## Model Evaluation

The models are compared using the following metrics:

- Accuracy
- Precision for charged-off loans
- Recall for charged-off loans
- F1-score for charged-off loans
- Average precision
- Brier score

The best model is selected based primarily on F1-score for the charged-off class.

## Final Model

The best-performing model is tuned Logistic Regression.

The final selected threshold is approximately:

```text
0.54
```

The final model performance is:

## Confusion Matrix Interpretation

The final confusion matrix shows:

This means:

- The model correctly identifies 11,999 charged-off loans.
- The model misses 7,964 charged-off loans.
- The model incorrectly flags 23,051 fully paid loans as charged-off.
- The model catches about 60% of actual charged-off loans.

## Business Interpretation

The final model is useful as a risk-screening tool, but it should not be used as an automatic loan rejection system.

The model has moderate recall, meaning it can identify a meaningful portion of charged-off loans. However, the precision is relatively low, meaning many loans flagged as risky would still be fully paid.

A practical use case would be to use the model as an early-warning system or as a decision-support tool for further manual review.

## Key Takeaways

- Loan default prediction is challenging because charged-off loans are the minority class.
- Accuracy alone is not a reliable metric for this project.
- F1-score, recall, precision, and average precision provide better insight into model performance.
- Train-only preprocessing helps prevent data leakage.
- Threshold optimization improves the usefulness of the final model.
- Logistic Regression performed slightly better than XGBoost and Random Forest in this version of the project.
- The final model is useful for identifying higher-risk loans but should be used to support, not replace, human decision-making.

## Conclusion

This project developed a complete loan default prediction workflow using LendingClub accepted-loan data. The analysis included data cleaning, feature engineering, descriptive analysis, train-only preprocessing, model tuning, class imbalance handling, probability calibration, threshold optimization, and final business interpretation.

The final Logistic Regression model achieved the best F1-score for charged-off loans. While the model does not perfectly separate risky and safe loans, it provides useful predictive signal and demonstrates how machine learning can support loan risk analysis.
