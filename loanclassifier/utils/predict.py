from sklearn.svm import LinearSVC
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_score,
    classification_report
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
    report = []
    loan_results = pd.DataFrame(index=std_data.index)
    for item in models:
        model_name, model = item["name"],item["model"]
        model_data = woe_data if "woe" in model_name.lower() else std_data
        X = model_data.drop(columns=["DLQ_90_FLAG"])
        Y = model_data["DLQ_90_FLAG"]


        if isinstance(model, LinearSVC):
            predicted_proba = model._predict_proba_lr(X)[:, 1]
        else:
            predicted_proba = model.predict_proba(X)[:, 1]
        predicted_classes = (predicted_proba >= 0.5).astype(int)
        auc, accuracy = None, None
        if labelled:
            auc = roc_auc_score(Y, predicted_proba)
            accuracy = accuracy_score(Y, predicted_classes)
            class_report = classification_report(Y, predicted_classes, output_dict=True, zero_division=0)
            for class_label, metrics in class_report.items():
                if class_label in ["0.0", "1.0"]:
                    results.append({
                        "Model": model_name,
                        "Class": int(float(class_label)),
                        "AUC": auc,
                        "Accuracy": accuracy,
                        "Recall": metrics["recall"],
                        "Precision": metrics["precision"],
                        "F1-Score": metrics["f1-score"]
                    })
        
        total_loans = len(X)
        loans_accepted = (predicted_classes == 0).sum()
        loans_rejected = (predicted_classes == 1).sum()

        report.append({
            "Model": model_name,
            "Total Loans": total_loans,
            "Non-Default": loans_accepted,
            "Default": loans_rejected,
            "Default Percentage": round(loans_rejected/total_loans, 4)*100
        })
        loan_results[f"{model_name}_Predicted_Probabilities"] = predicted_proba
        loan_results[f"{model_name}_Predicted_Classes"] = predicted_classes

    return report, results, loan_results

def plot_roc_curve(models, results, Y_true):
    """Plot ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for model in models:
        model_name = model["name"]
        y_pred_proba = results[model_name+"_Predicted_Probabilities"]
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