
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
from sklearn import preprocessing


# In[2]:


with open('train.dat','r') as f:
    Df = pd.DataFrame (l.strip ().split () for l in f)


# In[3]:


#normalizing the values to scale the values
x_train = preprocessing.normalize(Df, norm='l2')


# In[5]:


#plotting the normalized values
import matplotlib.pyplot as plt
plt.plot(x_train)
plt.xlabel('records')
plt.ylabel('Normalized values')
plt.title('Recs vs Norm vals')
plt.show()


# In[6]:


Y_train = pd.read_csv(filepath_or_buffer='train.labels',header=None, sep='\n')
Y1_train=np.ravel(Y_train)


# In[7]:


from sklearn.decomposition import PCA
pca = PCA(n_components=28)
x = pca.fit_transform(x_train,Y1_train)


# In[8]:


var  = pca.explained_variance_ratio_
plt.plot(var)
plt.xlabel('Number of estimators')
plt.ylabel('Variance')
plt.title('No. of Estimators vs Variance')
plt.show()


# In[9]:


with open('test.dat','r') as g1:
    Df_test = pd.DataFrame (h1.strip ().split () for h1 in g1)


# In[10]:


#x_test = preprocessing.normalize(Df_test, norm='l1')
x_test = preprocessing.normalize(Df_test, norm='l2')
#x_test = scaler.transform(Df_test)
x1_test = pca.transform(x_test)


# In[ ]:


#on trying with 1000 trees using random forest, got 0.81 on F1 score in CLP
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
rf = Pipeline([('PCA',PCA()),
               ('RandomForestCLassifier',RandomForestClassifier(n_estimators= 1000,criterion="entropy",random_state=0))])
rf.fit(x,Y1_train)
pred=rf.predict(x1_test3)


# In[ ]:


#this gave 0.8236% on F1 score in the CLP
import xgboost as xgb
from xgboost import XGBClassifier
rf= XGBClassifier(n_estimators=1000,learning_rate=0.4,max_depth=10, min_child_weight=2, objective= 'binary:logistic')


# In[12]:


#on decreasing the learning rate to 0.3 and the max_depth to 8, and increasing the min_child_weight to 5, 
#it resulted in 0.829 on F1 score in the CLP
import time
startTime = time.time()
import xgboost as xgb
from xgboost import XGBClassifier
rf_final= XGBClassifier(n_estimators=1000,learning_rate=0.3,max_depth=8,min_child_weight=5,objective= 'binary:logistic')
rf_final.fit(x,Y1_train)
pred=rf_final.predict(x1_test)
endTime = time.time()


# In[ ]:


duration_Time = endTime-start_time
print duration_Time


# In[ ]:


output=pd.DataFrame(data=pred)
output.to_csv("Result_PR2.dat",index=False,quoting=3,header=None)

