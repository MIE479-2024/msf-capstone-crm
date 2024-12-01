from utils.non_woe_preparation import preprocess_NoWoE
from utils.woe_preparation import preprocess_WoE
from utils.preprocess_general import preprocess
from utils.predict import get_predictions, plot_roc_curve
import pandas as pd
import pickle
import warnings 


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="`Booster.save_model`")
warnings.filterwarnings("ignore", module="xgboost")
class LoanClassifier():
    """
    """
    def __init__(self, data_path: str, labelled: bool):
        self.data = pd.read_csv(data_path, index_col="LOAN_ID", low_memory=False)
        self.labelled = labelled 
        self.std_process_data = preprocess_NoWoE(self.data, self.labelled)
        self.woe_process_data = preprocess_WoE(self.data, self.labelled)
        self._predictions = None  
        self.models_evaluated = False   
    

    def evaluate(self, models: str = "all"):
        """
        Generate predictions and evaluate models against true labels if applicable.

        Args:
            models: str, default = "all"
            One of the following options: ["all", "woe", "xgboost", "svm", "woe_lr", "woe_svm"].
        """
        self._read_models(models=models)
        self._metrics, self._predictions = get_predictions(
            self.models_list,
            self.std_process_data,
            self.woe_process_data,
            self.labelled
        )
        self.models_evaluated = True


    @property
    def classification_report(self):
        if not self.models_evaluated:
            raise RuntimeError("Metrics are not available until models are evaluated.")
        return pd.DataFrame(self._metrics).set_index("Model")
    

    @property
    def predictions(self):
        if not self.models_evaluated:
            raise RuntimeError("Predictions are not available until models are evaluated.")
        
        return self._predictions

    
    @property
    def roc_curve(self):
        if not self.labelled:
            raise RuntimeError("ROC Curve is only available for labelled data, i.e. loan data with performance variables.")

        if not self.models_evaluated:
            raise RuntimeError("ROC Curve is not available until models are evaluated. To evaluate selected classifiers, first call .evaluate().")
        
        return plot_roc_curve(self.models_list, self._predictions, self.std_process_data["DLQ_90_FLAG"])


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
            file = "./models/" + model_name
            with open(file, 'rb') as f:
                model = pickle.load(f)
            self.models_list.append({'name': model_name[:-4], 'model': model})