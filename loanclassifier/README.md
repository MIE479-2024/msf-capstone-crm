# LoanClassifier Package

## Overview

The **LoanClassifier** package is designed to help financial analysts and data scientists predict whether a loan is likely to become 90-day delinquent. The package is particularly useful for:

1. **Testing Your Labelled Data**  
   Evaluate the performance of pre-trained models on datasets with known performance variables.
   
2. **Generating Predictions for Unlabelled Data**  
   Predict delinquency probabilities for datasets where performance variables are unknown.

This package utilizes four pre-trained models for predictions, all trained on [Fannie Mae's Single-Family Loan Performance Data from 2021](https://capitalmarkets.fanniemae.com/credit-risk-transfer/single-family-credit-risk-transfer/fannie-mae-single-family-loan-performance-data).
- **Linear SVM (Weight of Evidence - WoE)**: `Linear_SVM_WoE.pkl`
- **Linear SVM**: `Linear_SVM.pkl`
- **Logistic Regression (WoE)**: `Logistic_Regression_WoE.pkl`
- **XGBoost**: `XGBoost.pkl`

---

## A note on WoE (Weight of Evidence)

**Weight of Evidence (WoE)** is a feature transformation technique often used in credit scoring. It involves the segmentation of loan acquisition variables into bins followed by determining the strength and relationship of each predictor with the loan outcome. In particular it is defined as,

$$WoE = ln(\frac{\text{Proportion of Good Loans}}{\text{Proportion of Good Loans}}) * 100$$

This feature engineering technique helps,
- Reduce the impact of outliers and overfitting.
- Improve model interpretability.
- Encode categorical variables for numerical models.

In this package, the models suffixed with `WoE` utilize features transformed using this technique.

---

## Key Features

### 1. **Evaluate Models**
Test all or specific models on your labelled dataset to identify the best-performing model. Evaluation includes:
- Predicted probabilities and classes.
- Classification metrics (e.g., accuracy, precision, recall, F1 score).
- A detailed classification report.
- Receiver Operating Characteristic (ROC) curves.

### 2. **Predict for Unlabelled Data**
Use pre-trained models to predict delinquency probabilities (and labels) for unlabelled datasets.

### 3. **Customizable Model Selection**
Run specific models or evaluate all models at once, using the following options:
- `"all"`: All models.
- `"woe"`: Models using WoE-transformed features.
- `"xgboost"`: XGBoost model.
- `"svm"`: Standard SVM.
- `"woe_lr"`: Logistic Regression (WoE).
- `"woe_svm"`: SVM (WoE).

### 4. **Export Predicted Classes to `csv`**
Save predicted deliquency labels and associated probabilities for each model to file, indexed by `LOAN_ID`.

---


## How to Use

### 1. **Initialization**
```python
from loan_classifier import LoanClassifier

# Initialize the classifier with your dataset
classifier = LoanClassifier(data_path="path/to/your_data.csv", labelled=True)
```

### 2. **Evaluate Models (For Labelled Data)**
```python
# Evaluate all models
classifier.evaluate(models="all")

# Access classification metrics
metrics = classifier.classification_metrics

# View the classification report
report = classifier.classification_report

# Access predictions
predictions = classifier.predictions

# Plot the ROC curve
roc_curve = classifier.roc_curve
```

### 3. **Generate Predictions (For Labelled or Unlabelled Data)**
```python
# Predict delinquency
classifier.evaluate(models="xgboost")  # Example: Using only XGBoost

# Access predictions
predictions = classifier.predictions

# Save predictions
classifier.save_predictions(output_path="path/to/save_predictions.csv")
```

---

## File Structure

- **`utils/non_woe_preparation.py`**: Preprocessing functions for models not using WoE.
- **`utils/woe_preparation.py`**: Preprocessing functions for models using WoE.
- **`utils/preprocess_general.py`**: General preprocessing functions.
- **`utils/predict.py`**: Functions for generating predictions and plotting ROC curves.
- **`utils/models/`**: Directory containing pre-trained model files (`*.pkl`).
- **`sample_data/`**: Example data following required format.
---

## Sample Outputs

1. **Metrics**: Evaluation metrics for labelled datasets.
   
   ![image](https://github.com/user-attachments/assets/c115db3f-e583-4db8-945e-2b28ac1f67dd)

2. **Classification Report**: Detailed model performance report.
   
   ![image](https://github.com/user-attachments/assets/517cb460-a0f6-44cf-9dfa-07821dfeda4f)

3. **ROC Curve**: Visualization of model performance.
   ![image](https://github.com/user-attachments/assets/262c9171-46c9-4b3b-9cf4-2cb9ccf64041)
   
4. **Predictions**: Loan-level delinquency predictions merged with the input dataset.
   
   ![image](https://github.com/user-attachments/assets/95004c68-c58f-456f-a96e-ddff445a7cfa)

---
## Installation and Dependencies

Clone the repository and change your directory to be `LoanClassifier/`. Then run the following.

```bash
python setup.py sdist
pip install .
```

Your package is now ready to use. Import by writing,
```python
import loan_classifier
```

---

## Notes

- Ensure your input CSV file has a unique loan identifier column named `LOAN_ID`.
- Furthermore data _must_ contain the following columns for each classifier type:
    - WoE models:
    ```python
    ["ORIG_RATE","CSCORE_B","OLTV"]
    ```
    - NonWoE models
    ```python
    ['orig_rt', 'orig_amt', 'orig_trm', 'oltv', 'num_bo', 'dti',
    'CSCORE_B', 'FTHB_FLG', 'purpose', 'NUM_UNIT', 'occ_stat', 'mi_pct']
    ```
- See `sample_data/sample_woe.csv` and `sample_data/sample_non_woe.csv` for example datasets.
---
