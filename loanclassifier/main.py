from utils.non_woe_preparation import preprocess_NoWoE
from utils.woe_preparation import preprocess_WoE
from utils.preprocess_general import data_preprocess
from utils.predict import get_predictions, plot_roc_curve
import pandas as pd
import pickle

class LoanClassifier():
    """
    """
    def __init__(self, data_path: str, labelled: bool):
        self.data = data_preprocess(data_path)
        self.std_process_data = preprocess_NoWoE(self.data, labelled=labelled)
        self.woe_process_data = preprocess_WoE(self.data, labelled=labelled)
        self._predictions = None  
        self.models_evaluated = False   
        self.labelled = labelled 
        self.evaluation_results = None
    

    def evaluate(self, models: str = "all"):
        """
        Generate predictions and evaluate models against true labels if applicable.

        Args:
            models: str, default = "all"
            One of the following options: ["all", "woe", "xgboost", "svm", "woe_lr", "woe_svm"].
        """
        self.models = self._read_models(models=models)
        self._predictions = get_predictions(
            self.models,
            self.std_process_data,
            self.woe_process_data,
            self.labelled
        )
        self.models_evaluated = True


    @property
    def predictions_dict(self):
        if not self.models_evaluated:
            raise RuntimeError("Predictions are not available until models are evaluated.")
        
        return self._predictions


    @property
    def predicted_classes(self):
        if not self.models_evaluated:
            raise RuntimeError("Predictions are not available until models are evaluated.")
        
        return self._predictions['Predicted Classes']
    

    def roc_curve(self):
        if not self.labelled:
            raise RuntimeError("ROC Curve is only available for labelled data, i.e. loan data with performance variables.")

        if not self.models_evaluated:
            raise RuntimeError("ROC Curve is not available until models are evaluated. To evaluate selected classifiers, first call .evaluate().")
        
        return plot_roc_curve(self.models, self.std_process_data["Y"])

        
    def _read_models(self, models: str) -> list:
        """
        Load scikit-learn models from pkl.

        Args:
            models: str, default = "all"
            One of the following options: ["all", "woe", "xgboost", "svm", "woe_lr", "woe_svm"].
        """
        if models not in ["all", "woe", "xgboost", "svm", "woe_lr", "woe_svm"]:
            raise RuntimeError(
                ("Invalid model type selection. Please choose one of the following: ",
                 '["all", "woe", "xgboost", "svm", "woe_lr", "woe_svm"]')
            )
        if models == "all":
            model_names = ["linear_svm_3f_woe.pkl", "linear_svm_2021.pkl",
                           "log_reg_3f_woe.pkl", "xgboost_2021.pkl"]
        if models == "woe": 
            model_names = ["linear_svm_3f_woe.pkl", "log_reg_3f_woe.pkl"]
        if models == "xgboost":
            model_names = ["xgboost_2021.pkl"]
        if models == "svm":
            model_names = ["linear_svm_2021.pkl"]
        if models == "woe_svm":
            model_names = ["linear_svm_3f_woe.pkl"]
        if models == "woe_lr":
            model_names = ["log_reg_3f_woe.pkl"]
        self.models_list = []
        for model_name in model_names:
            file = "/models/" + model
            with open(file, 'rb') as f:
                model = pickle.load(f)
            self.models_list.append({'name': model_name, 'model': model})