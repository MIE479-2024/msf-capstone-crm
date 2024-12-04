import sys
import pandas as pd

sys.path.append('/Users/mengyanzhu/Documents/GitHub/msf-capstone-crm/')
from importlib import reload  # Python 3.4+
#importlib.reload(mymodulesys.path.append('/Users/mengyanzhu/Documents/GitHub/msf-capstone-crm/SVM_optimizer')
from DataPreparation.NonWoE_preparation import preprocess_NoWoE as pN

data = pN(["2023"], True)

print('hello')