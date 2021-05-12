import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from ray.tune.sklearn import TuneGridSearchCV
import time

covtype = fetch_covtype(as_frame=True)
data = covtype['data']
label = covtype['target']

x=data.head(int(len(data)*(50/100)))
y=label.head(int(len(label)*(50/100)))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]
ccp_alpha = [0.0]

# Create the param grid
param_grid = {#'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'ccp_alpha': ccp_alpha
             }
print(param_grid)
rf_Model = RandomForestClassifier()
tune_search = TuneGridSearchCV(rf_Model, param_grid, early_stopping=True, max_iters=10)
start = time.time()
tune_search.fit(X_train, y_train)
end = time.time()
print("Tune GridSearch Fit Time:", end - start)
tune_search.best_params_
