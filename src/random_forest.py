
# coding: utf-8

# In[5]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.grid_search as gs
import sklearn.cross_validation as cv
from sklearn import datasets
from sklearn import tree
import sklearn.grid_search as gs
import math
from time import time
from sklearn import ensemble
get_ipython().magic(u'matplotlib inline')
pd.options.display.max_columns = 200


# In[6]:

# import preprocessed data

test_basic_df = pd.read_csv('./test_basic.csv')
test_basic_df.drop(labels = 'Unnamed: 0', axis = 1, inplace = True)

train_basic_df = pd.read_csv('./train_basic.csv')
train_basic_df.drop(labels = 'Unnamed: 0', axis = 1, inplace = True)

test_components_df = pd.read_csv('./test_components.csv')
test_components_df.drop(labels = 'Unnamed: 0', axis = 1, inplace = True)

train_components_df = pd.read_csv('./train_components.csv')
train_components_df.drop(labels = 'Unnamed: 0', axis = 1, inplace = True)

test_id_df = pd.read_csv('./test_id.csv')
test_id_df.drop(labels = 'Unnamed: 0', axis = 1, inplace = True)

cost_df = pd.read_csv('./cost.csv')
cost_df.drop(labels = 'Unnamed: 0', axis = 1, inplace = True)


# # Random Forest on Basic

# In[4]:

# prepare basic training and validation arrays for fitting
X_basic = np.array(train_basic_df)
y = np.array(cost_df)


# In[35]:

# random forest initialization and fitting
# 
randomForest_basic = ensemble.RandomForestRegressor(n_estimators = 1000,
                                                    max_features = 'sqrt')
# try cross validation and grid search to improve forest
np.random.seed(0)
grid_para_forest_basic = [{'max_leaf_nodes': np.linspace(start = 2,
                                                        stop = 700,
                                                        num = 15)}]
grid_search_forest_basic = gs.GridSearchCV(estimator = randomForest_basic,
                                         param_grid = grid_para_forest_basic,
                                         cv = 10)
t0 = time()
grid_search_forest_basic.fit(X_basic, y.ravel())
t1 = time()
t1 - t0


# In[36]:

grid_search_forest_basic.best_params_


# In[37]:

grid_search_forest_basic.best_score_


# In[38]:

grid_search_forest_basic.score(X_basic, y)


# In[39]:

cost_basic_df = pd.DataFrame(data = grid_search_forest_basic.predict(np.array(test_basic_df)),
                            columns = ['cost'])
basic_predictions_df = pd.concat(objs = [test_id_df, cost_basic_df], axis = 1)
basic_predictions_df.to_csv('./random_forest_basic_predictions.csv',
                           index = False)
# paramters: {'min_samples_leaf': 1, 'min_samples_split': 2.0, 'n_estimators': 10}; test_size = 0.20; score: 0.359864; rank 981/1324
# paramters: {'min_samples_leaf': 1, 'min_samples_split': 2.0, 'n_estimators': 10}; test_size = 0.01; score: 0.351449; rank 976/1324
# started using all data to cross-validate parameters
# paramters: {'min_samples_leaf': 1, 'min_samples_split': 2.0, 'n_estimators': 100}; score: 0.326865; rank 953/1324
# paramters: {'max_leaf_nodes': 700, 'n_estimators': 1000}; score: 0.504153; rank 1134/1324


# # Random Forest on Components

# In[7]:

# prepare components training and validation arrays for fitting
X_components = np.array(train_components_df)
y = np.array(cost_df)


# In[18]:

# Start with a single decision tree
randomForest_components = ensemble.RandomForestRegressor(n_estimators = 1000,
                                                         max_features = 'sqrt')
# try cross validation and grid search to improve tree
np.random.seed(0)
grid_para_forest_components = [{'max_leaf_nodes': np.linspace(start = 2,
                                                        stop = 700,
                                                        num = 15).astype(dtype = int)}]
grid_search_forest_components = gs.GridSearchCV(estimator = randomForest_components,
                                         param_grid = grid_para_forest_components,
                                         cv = 10)
t0 = time()
grid_search_forest_components.fit(X_components, y.ravel())
t1 = time()
t1 - t0


# In[19]:

grid_search_forest_components.best_params_


# In[20]:

grid_search_forest_components.best_score_


# In[21]:

grid_search_forest_components.score(X_components, y)


# In[22]:

cost_components_df = pd.DataFrame(data = grid_search_forest_components.                                  predict(np.array(test_components_df)),
                            columns = ['cost'])
components_predictions_df = pd.concat(objs = [test_id_df, cost_components_df], axis = 1)
components_predictions_df.to_csv('./random_forest_components_predictions.csv',
                           index = False)
# paramters: {'min_samples_leaf': 1, 'min_samples_split': 2.0, 'n_estimators': 100}; score: 0.318257; rank 950/1324
# paramters: {'max_leaf_nodes': 650}; score: 0.490001; rank 1120/1324


# In[ ]:



