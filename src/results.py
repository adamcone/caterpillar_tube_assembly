
# coding: utf-8

# # Graphing Results of Models

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
get_ipython().magic(u'matplotlib inline')
pd.options.display.max_columns = 250


# ## Component dataframe vs Basic Dataframe

# In[7]:

# Decision Tree
N = 4
dt_basic = (1-(999/1324.0), 1-(970/1324.0), 1-(977/1324.0), 1-(1068/1324.0))

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, dt_basic, width, color='r')

dt_components = (1-(975/1324.0), 1-(958/1324.0), 1-(961/1324.0), 1-(1181/1324.0))

rects2 = ax.bar(ind + width, dt_components, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel('% Teams Outperformed')
ax.set_title('Decision Tree Performance')
ax.set_xticks(ind + width)
ax.set_xticklabels(('A1', 'A2', 'A3', 'A4'))

ax.legend((rects1[0], rects2[0]), ('Basic', 'Components'))


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                #'%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()


# In[9]:

# Random Forest
N = 2
rf_basic = (1-(953/1324.0), 1-(1134/1324.0))

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, rf_basic, width, color='r')

rf_components = (1-(950/1324.0), 1-(1120/1324.0))

rects2 = ax.bar(ind + width, rf_components, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel('% Teams Outperformed')
ax.set_title('Random Forest Performance')
ax.set_xticks(ind + width)
ax.set_xticklabels(('A1', 'A2'))

ax.legend((rects1[0], rects2[0]), ('Basic', 'Components'))


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                #'%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()


# In[10]:

# Gradient Boosting
N = 2
gb_basic = (1-(1224/1324.0), 0.0)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, gb_basic, width, color='r')

gb_components = (1-(1224/1324.0), 0.0)

rects2 = ax.bar(ind + width, gb_components, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel('% Teams Outperformed')
ax.set_title('Gradient Boosting Performance')
ax.set_xticks(ind + width)
ax.set_xticklabels(('A1', 'A2'))

ax.legend((rects1[0], rects2[0]), ('Basic', 'Components'))


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                #'%d' % int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.show()


# ## Model vs Model

# In[19]:

N = 3
best_models_components = (1-(958/1324.0), 1-(950/1324.0), 1-(1224/1324.0))

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, best_models_components, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel('% Teams Outperformed')
ax.set_title('Performance of Best Models')
ax.set_xticks(ind)
ax.set_xticklabels(('', '', ''))


#def autolabel(rects):
 #   # attach some text labels
#    for rect in rects:
#        height = rect.get_height()
 #       ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
#              #'%d' % int(height),
#                ha='center', va='bottom')

#autolabel(rects1)

plt.show()


# In[ ]:



