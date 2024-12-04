# $\mathbb{MSF}$ Capstone - Credit Risk Management 

## Repo folder structure
### Data preparation
`Data Preparation/`
- `Data Preparation.ipynb`: Takes Fannie Mae Single-Family Loan Performance Data of a single quarter and returns a reformatted table with statistics required for this project. Sample input: `2021Q1.csv`. Sample output: `2021Q1_stat.csv`. Raw data download: https://datadynamics.fanniemae.com/data-dynamics/#/downloadLoanData/Single-Family
- `HPI_preprocess.ipynb`: Prepares data for models with House Price Index. Takes HPI at each three-digit zip code from 1995 to 2024 and returns a reformatted table. Sample input: `hpi_raw.csv`. Sample output: `HPI_reformat.csv`. Raw data download: https://www.fhfa.gov/data/hpi/datasets?tab=quarterly-data
- `Preprocess.ipynb`: Takes the prepared quarterly tables from `Data Preparation.ipynb` and returns a preprocessed annual table. Sample input: `2021Q1_stat.csv` to `2021Q4_stat.csv`. Sample output: `2021_stat.csv`
- `prepare_func.py`: Functions to prepare and preprocess the datasets

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
