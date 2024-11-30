from sklearn.svm import LinearSVC
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_score
)
import pandas as pd


def get_predictions(
    models: list,
    std_data: pd.DataFrame,
    woe_data: pd.DataFrame,
    labelled: bool,
) -> dict:
    results = {}
    for item in models:
        model_name, model = item["name"],item["model"]
        model_data = woe_data if "woe" in model_name else std_data
        if labelled:
            X, Y = model_data.drop(columns=["Y"]), model_data["Y"]
        else:
            X = model_data 
        if isinstance(model, LinearSVC):
            predicted_proba = model._predict_proba_lr(X)[:, 1]
        else:
            predicted_proba = model.predict_proba(X)[:, 1]
        predicted_classes = (predicted_proba >= 0.5).astype(int)
        
        auc = roc_auc_score(Y, predicted_proba) if labelled else None 
        accuracy = accuracy_score(Y, predicted_classes) if labelled else None 
        recall = recall_score(Y, predicted_classes) if labelled else None 
        precision = precision_score(Y, predicted_classes) if labelled else None 
        f1 = f1_score(Y, predicted_classes) if labelled else None  

        results[model_name] = {
            "AUC": auc,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "F1-Score": f1,
            "Predicted Probabilities": predicted_proba,
            "Predicted Classes": predicted_classes
        }

    return results