#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Following are the steps involved in creating a well-defined ML project:

1] Understand and define the problem

2] Prepare the data

3] Explore and Analyse the data

4] Apply the algorithms

5] Reduce the errors

6] Predict the result
"""


# In[45]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from urllib.request import urlretrieve


# In[46]:


iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urlretrieve(iris)
df = pd.read_csv(iris, sep=',')


# In[49]:


attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = attributes
df.head()


# In[50]:


df.info()


# In[51]:


df.describe()


# In[52]:


import seaborn as sns


# In[53]:


df.shape


# In[55]:


df['class'].value_counts().to_frame()


# In[56]:


df.columns


# In[57]:


df.values


# In[58]:


df.describe(include='all')


# In[59]:


X=df.iloc[:,:4]
X


# In[60]:


y = df.iloc[:,4:]
y


# In[62]:


X=preprocessing.StandardScaler().fit_transform(X)
X


# In[63]:


X.mean()


# In[64]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[65]:


y_train.shape


# In[66]:


x_train.shape


# In[67]:


x_test.shape


# In[73]:


knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
y_predict = knn.predict(x_test)


# In[74]:


print(y_predict)


# In[75]:


from sklearn.metrics import accuracy_score


# In[79]:


acc = accuracy_score(y_test, y_predict)
acc


# In[80]:


from sklearn.metrics import confusion_matrix


# In[81]:


cm=confusion_matrix(y_test, y_predict)
cm


# In[83]:


cm1 = pd.DataFrame(data=cm, index=['setosa', 'versicolor', 'virginica'], columns=['setosa', 'versicolor', 'virginica'])
cm1


# In[91]:


y_test


# In[102]:


y_predict
print(type(y_predict))


# In[123]:


prediction_output=pd.DataFrame(data=[y_test.values,y_predict],index=['y_test','y_predict1'])


# In[125]:


prediction_output.transpose()


# In[145]:


Ks=11
mean_acc=np.zeros((Ks-1))


#train and predict
for n in range(1,Ks):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(x_train,y_train.values.ravel())
    yhat=neigh.predict(x_test)
    mean_acc[n-1]=accuracy_score(y_test,yhat)


# In[146]:


print(mean_acc)


# In[147]:


print( "The best accuracy was with", mean_acc.max(), "with k = ", mean_acc.argmax()+1) 


# In[148]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.legend(('Accuracy '))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

