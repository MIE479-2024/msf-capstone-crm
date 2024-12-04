# $\mathbb{MSF}$ Capstone - Credit Risk Management 

## Repo folder structure
### Data preparation
`Data Preparation/`
- `Data Preparation.ipynb`
- `HPI_preprocess.ipynb`
- `Preprocess.ipynb`
- `prepare_func.py`

### Model training and application
`loanclassifier/`: Contains all files related to the Python package LoanClassifier. See associated [README.md](loanclassifier/README.md) for more information.


`Demo`/: 
- `demo.ipynb`: Demonstrated use of LoanClassifier package using the following sample datasets.
- `sample_labelled_data.csv`: dataset with all populated fields
- `sample_unlabelled_data.csv`: dataset with missing performance variables

`default_modelling`: Contains all notebooks used to train all explored models.
- `Credit Risk Scorecard\`: Training notebook with weight of evidence based approach for Logistic Regression, SVM and XGBoost models
- `SVM Optimization\`: Training notebook related to the 2-Medians-SVM Model
- `Scikit SVM Models\`: Training notebooks for SVM models trained using the scikit-learn library
- `Thunder SVM Models\`: Training notebooks for SVM models trained using the thundersvm library (for GPU based training)
- `XGBoost\`: Training notebook for XGBoost model
- _`deprecated`_: Scripts and notebooks no longer in use.
  
### Exploratory analyses
`EDA`/: Contains early exploratory data analyses of Fannie Mae dataset as well as feature correlation analyses by year.

`Compare CSV`/ 
