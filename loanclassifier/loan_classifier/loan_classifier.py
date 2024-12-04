from utils.non_woe_preparation import preprocess_NoWoE
from utils.woe_preparation import preprocess_WoE
from utils.preprocess_general import preprocess
from utils.predict import get_predictions, plot_roc_curve
import pandas as pd
import pickle
import warnings 
import pkgutil

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="`Booster.save_model`")
warnings.filterwarnings("ignore", module="xgboost")


def load_model(model_name: str):
    """Reads in pickle model from file.
    """
    model_data = pkgutil.get_data('utils', f'models/{model_name}')
    return pickle.loads(model_data)


class LoanClassifier():
    """
    A classifier for loan data that preprocesses, evaluates models, and provides metrics and 
    predictions for performance evaluation.

    Attributes:
        data (pd.DataFrame): The loan data loaded from the provided path.
        labelled (bool): Indicates whether the data contains performance variables.
        std_process_data (pd.DataFrame): Data preprocessed without Weight of Evidence (WoE).
        woe_process_data (pd.DataFrame): Data preprocessed with WoE.
        _predictions (pd.DataFrame): The predictions made by evaluated models.
        models_evaluated (bool): Indicates whether models have been evaluated.
    """
    def __init__(self, data_path: str, labelled: bool):
        """
        Initialize the LoanClassifier class with provided data
        Args:
            data_path (str): Path to the CSV file containing loan data.
            labelled (bool): Whether the data contains labels for loan performance.
        """
        self.data = pd.read_csv(data_path, index_col="LOAN_ID", low_memory=False)
        self.labelled = labelled 
        self.std_process_data = preprocess_NoWoE(self.data, self.labelled)
        self.woe_process_data = preprocess_WoE(self.data, self.labelled)
        self._predictions = None  
        self.models_evaluated = False   
    

    def evaluate(self, models: str = "all") -> None:
        """
        Generate predictions and evaluate models against true labels if applicable.

        Args:
            models (str) default = "all"
            One of the following options: ["all", "woe", "xgboost", "svm", "woe_lr", "woe_svm"].
        """
        self._read_models(models=models)
        self._report, self._metrics, self._predictions = get_predictions(
            self.models_list,
            self.std_process_data,
            self.woe_process_data,
            self.labelled
        )
        self.models_evaluated = True


    @property
    def classification_metrics(self) -> pd.DataFrame:
        """
        Retrieve classification metrics for evaluated models.

        Returns:
            pd.DataFrame: Classification metrics for each model
            (i.e AUC, accuracy, recall, etc.)

        Raises:
            RuntimeError: If the data is not labelled or models are not evaluated.
        """
        if not self.labelled:
            raise RuntimeError("Classification metrics are only available for labelled data, i.e. loan data with performance variables.")
        if not self.models_evaluated:
            raise RuntimeError("Metrics are not available until models are evaluated.")
        return pd.DataFrame(self._metrics).set_index("Model")


    @property
    def classification_report(self) -> pd.DataFrame:
        """
        Returns the classification report for evaluated models.

        Returns:
            pd.DataFrame: The classification report for each model.
            Contains information about the number of loans predicted
            to default for each model.

        Raises:
            RuntimeError: If models are not evaluated.
        """
        if not self.models_evaluated:
            raise RuntimeError("Classification report is not available until models are evaluated.")
        return pd.DataFrame(self._report).set_index("Model")
    

    @property
    def predictions(self) -> pd.DataFrame:
        """
        Retrieve the predictions made by evaluated models.

        Returns:
            pd.DataFrame: The predictions for the loan data.

        Raises:
            RuntimeError: If models are not evaluated.
        """
        if not self.models_evaluated:
            raise RuntimeError("Predictions are not available until models are evaluated.")
        
        return self._predictions

    
    @property
    def roc_curve(self):
        """
        Plot the ROC curve for evaluated models.

        Returns:
            (matplotlib.figure.Figure, matplotlib.figure.Axis): The ROC curve plot.

        Raises:
            RuntimeError: If the data is not labelled or models are not evaluated.
        """
        if not self.labelled:
            raise RuntimeError("ROC Curve is only available for labelled data, i.e. loan data with performance variables.")

        if not self.models_evaluated:
            raise RuntimeError("ROC Curve is not available until models are evaluated. To evaluate selected classifiers, first call .evaluate().")
        
        return plot_roc_curve(self.models_list, self._predictions, self.std_process_data["DLQ_90_FLAG"])


    def _read_models(self, models: str) -> None:
        """
        Load scikit-learn models from pickle files.

        Args:
            models (str): The models to load. Options are:
                ["all", "woe", "xgboost", "svm", "woe_lr", "woe_svm"].

        Raises:
            RuntimeError: If an invalid model type is selected.
        """
        if models not in ["all", "woe", "xgboost", "svm", "woe_lr", "woe_svm"]:
            raise RuntimeError(
                ("Invalid model type selection. Please choose one of the following: ",
                 '["all", "woe", "xgboost", "svm", "woe_lr", "woe_svm"]')
            )
        if models == "all":
            model_names = ["Linear_SVM_WoE.pkl", "Linear_SVM.pkl",
                           "Logistic_Regression_WoE.pkl", "XGBoost.pkl"]
        if models == "woe": 
            model_names = ["Linear_SVM_WoE.pkl", "Logistic_Regression_WoE.pkl"]
        if models == "xgboost":
            model_names = ["XGBoost.pkl"]
        if models == "svm":
            model_names = ["Linear_SVM.pkl"]
        if models == "woe_svm":
            model_names = ["Linear_SVM_WoE.pkl"]
        if models == "woe_lr":
            model_names = ["Logistic_Regression_WoE.pkl"]
        self.models_list = []
        for model_name in model_names:
            model = load_model(model_name)
            self.models_list.append({'name': model_name[:-4], 'model': model})

    
    def save_predictions(self, output_path: str) -> None:
        """
        Save the original data combined with predictions to a specified file path.

        Args:
            output_path (str): Path to save the combined data with predictions.

        Raises:
            RuntimeError: If models are not evaluated.
        """
        if not self.models_evaluated:
            raise RuntimeError("You must evaluate models before saving predictions.")

        predictions_df = self.predictions

        combined_data = self.data.merge(predictions_df, on="LOAN_ID", how="left")

        combined_data.to_csv(output_path, index=True)
        print(f"Predictions saved successfully to {output_path}.")