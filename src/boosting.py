
# coding: utf-8

# In[3]:

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
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
get_ipython().magic(u'matplotlib inline')
pd.options.display.max_columns = 200


# In[4]:

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


# # Boosting on Basic

# In[12]:

# prepare basic training and validation arrays for fitting
X_basic = np.array(train_basic_df)
y = np.array(cost_df)


# In[27]:

# initialize boosting tree object
gradientBoost = GradientBoostingRegressor(n_estimators = 1000,
                                          max_leaf_nodes = 2,
                                          loss = 'ls')

# try cross validation and grid search to improve boosting tree object
np.random.seed(0)
learning_rate_paramter_list = map(math.exp,
                                  np.linspace(start = math.log(0.001),
                                              stop = math.log(0.01),
                                              num = 3))
grid_para_boost_basic = [{'learning_rate': learning_rate_paramter_list}]
grid_search_boost_basic = gs.GridSearchCV(gradientBoost,
                                         grid_para_boost_basic,
                                         cv = 10)
t0 = time()
grid_search_boost_basic.fit(X_basic, y.ravel())
t1 = time()
t1 - t0


# In[28]:

grid_search_boost_basic.best_params_


# In[29]:

grid_search_boost_basic.best_score_


# In[30]:

grid_search_boost_basic.score(X_basic, y)


# In[31]:

cost_basic_df = pd.DataFrame(data = grid_search_boost_basic.predict(np.array(test_basic_df)),
                            columns = ['cost'])
basic_predictions_df = pd.concat(objs = [test_id_df, cost_basic_df], axis = 1)
basic_predictions_df.to_csv('./gradient_boost_basic_predictions.csv',
                           index = False)
# paramters: {'learning_rate': 0.010000000000000004,
# 'max_leaf_nodes': 4,
# 'n_estimators': 10}; score: 0.913520; rank 1236/1324
# paramters: {'learning_rate': 0.010000000000000004,
# 'max_leaf_nodes': 2,
# 'n_estimators': 100}; score: 0.822071; rank 1224/1324


# # Boosting on Components

# In[5]:

# prepare basic training and validation arrays for fitting
X_components = np.array(train_components_df)
y = np.array(cost_df)


# In[6]:

# initialize boosting tree object
gradientBoost = GradientBoostingRegressor(n_estimators = 1000,
                                          max_leaf_nodes = 2,
                                          loss = 'ls')

# try cross validation and grid search to improve boosting tree object
np.random.seed(0)
learning_rate_paramter_list = map(math.exp,
                                  np.linspace(start = math.log(0.001),
                                              stop = math.log(0.01),
                                              num = 3))
grid_para_boost_components = [{'learning_rate': learning_rate_paramter_list}]
grid_search_boost_components = gs.GridSearchCV(gradientBoost,
                                         grid_para_boost_components,
                                         cv = 10)
t0 = time()
grid_search_boost_components.fit(X_components, y.ravel())
t1 = time()
t1 - t0


# In[7]:

grid_search_boost_components.best_params_


# In[8]:

grid_search_boost_components.best_score_


# In[9]:

grid_search_boost_components.score(X_components, y)


# In[12]:

cost_components_df = pd.DataFrame(data = grid_search_boost_components.                                  predict(np.array(test_components_df)),
                            columns = ['cost'])
components_predictions_df = pd.concat(objs = [test_id_df, cost_components_df], axis = 1)
components_predictions_df.to_csv('./gradient_boost_components_predictions.csv',
                           index = False)
# paramters: {'learning_rate': 0.010000000000000004,
# 'max_leaf_nodes': 2,
# 'n_estimators': 100}; score: 0.822071; rank 1224/1324
# paramters: {'learning_rate': 0.010000000000000004}; score: 

