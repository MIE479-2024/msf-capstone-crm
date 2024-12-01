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
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams.update({
    'font.family': 'serif',  # Use serif fonts
    'font.size': 14,  # General font size
    'axes.titlesize': 18,  # Title font size
    'axes.labelsize': 14,  # Label font size
    'xtick.labelsize': 12,  # X-axis tick font size
    'ytick.labelsize': 12,  # Y-axis tick font size
    'legend.fontsize': 14,  # Legend font size
    'figure.dpi': 300,  # Higher resolution for reports
    'savefig.dpi': 300,  # Resolution for saved figures
})


def get_predictions(
    models: list,
    std_data: pd.DataFrame,
    woe_data: pd.DataFrame,
    labelled: bool,
) -> dict:
    results = []
    loan_results = pd.DataFrame(index=std_data.index)
    for item in models:
        model_name, model = item["name"],item["model"]
        model_data = woe_data if "woe" in model_name else std_data
        X = model_data.drop(columns=["DLQ_90_FLAG"])
        Y = model_data["DLQ_90_FLAG"]
        if isinstance(model, LinearSVC):
            predicted_proba = model._predict_proba_lr(X)[:, 1]
        else:
            predicted_proba = model.predict_proba(X)[:, 1]
        predicted_classes = (predicted_proba >= 0.5).astype(int)
        
        if labelled:
            auc = roc_auc_score(Y, predicted_proba)
            accuracy = accuracy_score(Y, predicted_classes)
            for class_label in [0, 1]:  
                class_indices = (Y == class_label)
                print(class_label, sum(class_indices))
                recall = recall_score(Y[class_indices], predicted_classes[class_indices], zero_division=0)
                precision = precision_score(Y[class_indices], predicted_classes[class_indices], zero_division=0)
                f1 = f1_score(Y[class_indices], predicted_classes[class_indices], zero_division=0)
                
                results.append({
                    "Model": model_name,
                    "Class": class_label,
                    "AUC": auc,   
                    "Accuracy": accuracy,
                    "Recall": recall,
                    "Precision": precision,
                    "F1-Score": f1
                })

        loan_results[f"{model_name}_Predicted_Probabilities"] = predicted_proba
        loan_results[f"{model_name}_Predicted_Classes"] = predicted_classes

    return results, loan_results

def plot_roc_curve(results, Y_true):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for model_name, result in results.items():
        y_pred_proba = result["Predicted Probabilities"]
        fpr, tpr, _ = roc_curve(Y_true, y_pred_proba)        
        plt.plot(fpr, tpr, label=f"{model_name}")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random guess")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    plt.show()  

    return fig, ax