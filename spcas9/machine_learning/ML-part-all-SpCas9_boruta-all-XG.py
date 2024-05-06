import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from BorutaShap import BorutaShap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import pickle

data=pd.read_csv('sp_emx1_all_features_withlabels.csv')
data=data.drop(['Label','System','sys-labels'], axis=1)

# Step 1: Prepare the data
# dropping the rows having NaN values
data = data.dropna()
 # To reset the indices
data = data.reset_index(drop=True)
X = data.iloc[:, 1:]  # Remove the first five columns as they are identifiers and not relevant features
y = data['States']    # Assuming 'states' is the target variable column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
## Feature selection using Boruta

print ('########### Feature selection begins###########')

model =  XGBClassifier()

# no model selected default is Random Forest, if classification is False it is a Regression problem
Feature_Selector = BorutaShap(model=model,
                              importance_measure='shap',
                              classification=True)

Feature_Selector.fit(X_train, y_train, n_trials=50, random_state=0)

with open('Feature_Selector_data_spcas9-XG','wb') as f: pickle.dump(Feature_Selector, f)
