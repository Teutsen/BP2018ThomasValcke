
# coding: utf-8

# In[5]:


################################ IMPORTS ###################################
import os
import sys
import pandas as pd
import numpy as np
import urllib3
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, WeekdayLocator, DateFormatter
from dateutil.relativedelta import relativedelta
from pandas.core import datetools
################################ IMPORTS LINEAR ###################################
import statsmodels.api as sm
from sklearn import datasets ## imports datasets from scikit-learn


# In[6]:


################################ READ DATA ###################################
test1 = pd.read_csv('D:/School/STAGE_BP/Stage_BachlerProef/datasets/jams3.csv',low_memory=False,delimiter=";")


# In[7]:


test1


# In[4]:


# data cleaning: only Gent
test1 = test1.filter(test1["City"] == "Gent")
#test1 = test1.filter(test1["PubTime"] >= "2017-11-12") 


# In[8]:


##### EXAMPLE LINEAR ########
data = datasets.load_boston() ## loads Boston dataset from datasets library 
# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(data.data, columns=data.feature_names)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])

## Without a constant

import statsmodels.api as sm

X = df["RM"]
y = target["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[9]:


data


# In[10]:


################################ READ DATA ###################################
data = datasets.load_boston()
#df.to_csv('D:/School/STAGE_BP/Stage_BachlerProef/datasets/testPandaCsv.csv')
#test0 = pd.read_csv('D:/School/STAGE_BP/Stage_BachlerProef/datasets/testPandaCsv.csv')
test1 = pd.read_csv('D:/School/STAGE_BP/Stage_BachlerProef/datasets/jams3.csv',low_memory=False,delimiter=";",skip_blank_lines=True)
#test1 = SpSession.read.option("delimiter", ";").option("header", "true").csv("D:/School/STAGE_BP/Stage_BachlerProef/datasets/jams2.csv")


# In[11]:


test1.target


# In[12]:


# Assign feature columns as list featurecols
feature_cols=['Length','Speed','Level']
# Assign to X a subset of the data including only feature colums
x= test1[feature_cols]


# In[13]:


# Assign to y the response variable as pandas series
y = test1['RoadType']


# In[14]:


from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y)


# In[15]:


from sklearn.linear_model import LinearRegression
linreg= LinearRegression()
#linreg.fit(x_train,y_train)

import statsmodels.api as sm

# Note the difference in argument order
model = sm.OLS(y, x).fit()
predictions = model.predict(X)


# In[ ]:


# define the data/predictors as the pre-set feature names  
df = pd.DataFrame(test1, columns=City)

# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(test1, columns=["speed"])

## Without a constant

import statsmodels.api as sm

#X = df["RM"]
#y = target["MEDV"]

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()


# In[ ]:




        

