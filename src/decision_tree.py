
# coding: utf-8

# In[4]:

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
get_ipython().magic(u'matplotlib inline')
pd.options.display.max_columns = 200


# In[5]:

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


# # Decision Tree on Basic

# In[7]:

# prepare basic training and validation arrays for fitting
X_basic = np.array(train_basic_df)
y = np.array(cost_df)


# In[125]:

# Start with a single decision tree
tree_model_basic = tree.DecisionTreeRegressor()
# try cross validation and grid search to improve tree
np.random.seed(0)
grid_para_tree_basic = [{'max_leaf_nodes': range(2, 1800)}]
grid_search_tree_basic = gs.GridSearchCV(tree_model_basic,
                                         grid_para_tree_basic,
                                         cv = 10)
t0 = time()
grid_search_tree_basic.fit(X_basic, y)
t1 = time()
t1 - t0


# In[16]:

grid_search_tree_basic.best_params_


# In[127]:

grid_search_tree_basic.best_score_


# In[128]:

grid_search_tree_basic.score(X_basic, y)


# In[129]:

cost_basic_df = pd.DataFrame(data = grid_search_tree_basic.predict(np.array(test_basic_df)),
                            columns = ['cost'])
basic_predictions_df = pd.concat(objs = [test_id_df, cost_basic_df], axis = 1)
basic_predictions_df.to_csv('./decision_tree_basic_predictions.csv',
                           index = False)
# paramters: {'min_samples_leaf': 2, 'min_samples_split': 2.0}; score: 0.364173; test_size = 0.20; rank 999/1324
# paramters: {'min_samples_leaf': 4, 'min_samples_split': 7.333333333333333}; test_size = 0.20; score: 0.358763; rank 978/1324
# paramters: {'min_samples_leaf': 2, 'min_samples_split': 23.333333333333332}; test_size = 0.01; score: 0.344002; rank 970/1324
# started using all training data for cross-validation
# paramters: {'min_samples_leaf': 3, 'min_samples_split': 7.333333333333333}; score: 0.358205; rank 977/1324
# paramters: {'max_leaf_nodes': 86}; score: 0.467373; rank 1109/1324
# paramters: {'max_leaf_nodes': 280}; score: 0.404779; rank 1068/1324


# # Decision Tree on Components

# In[21]:

# prepare basic training and validation arrays for fitting
X_components = np.array(train_components_df)
y = np.array(cost_df)


# In[22]:

# Start with a single decision tree
tree_model_components = tree.DecisionTreeRegressor()
# try cross validation and grid search to improve tree
np.random.seed(0)
grid_para_tree_components = [{'max_leaf_nodes': range(2, 1000)}]
grid_search_tree_components = gs.GridSearchCV(tree_model_components,
                                              grid_para_tree_components,
                                              cv = 10)
t0 = time()
grid_search_tree_components.fit(X_components, y)
t1 = time()
t1 - t0


# In[23]:

grid_search_tree_components.best_params_


# In[24]:

grid_search_tree_components.best_score_


# In[25]:

grid_search_tree_components.score(X_components, y)


# In[26]:

cost_components_df = pd.DataFrame(data = grid_search_tree_components.                                  predict(np.array(test_components_df)),
                            columns = ['cost'])
components_predictions_df = pd.concat(objs = [test_id_df, cost_components_df], axis = 1)
components_predictions_df.to_csv('./decision_tree_components_predictions.csv',
                           index = False)
# parameters: {'min_samples_leaf': 4, 'min_samples_split': 2.0}; test_size = 0.20; score: 0.349347; rank 975/1324
# parameters: {'min_samples_leaf': 4, 'min_samples_split': 23.333333333333332}; test_size = 0.01; score: 0.333676; rank 958/1324
# started using all training data for cross-validation
# paramters: {'min_samples_leaf': 2, 'min_samples_split': 18.0}; score: 0.336359; rank 961/1324
# parameters: {'max_leaf_nodes': 9}; score: 0.630126; 1181/1324


# In[ ]:



