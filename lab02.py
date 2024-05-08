#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml


# In[2]:


from sklearn.model_selection import train_test_split


# In[3]:


import numpy as np


# In[4]:


import pandas as pd


# In[5]:


import pickle


# In[6]:


mnist=fetch_openml('mnist_784',version=1)


# In[7]:


#PRZYGOTOWANIE DANYCH


# In[8]:


mnist.frame.info()


# In[9]:


print(mnist.keys())


# In[10]:


X,y = mnist['data'],mnist['target'].astype(np.uint8)
print(X)
print(y)


# In[11]:


type_X=type(X)
type_y=type(y)
print("Typ X:")
print(type_X)
print("Typ y:")
print(type_y)
print("Czy uporzadkowany?")
uporzadkowany=all(y[i]<=y[i+1]for i in range(len(y)-1))
print(uporzadkowany)


# In[12]:


print((np.array(mnist.data.loc[42]).reshape(28,28)>0).astype(int))


# In[13]:


#ZBIOR UCZACY I TESTOWY


# In[14]:


print(y.index)


# In[15]:


X_df = pd.DataFrame(X)
y_series = pd.Series(y, name='label')

y_sorted = y_series.sort_values()

print(y_sorted.index)

X_sorted = X_df.reindex(y_sorted.index)

print(X_sorted.index)


# In[16]:


print(len(X))
print(len(y))


# In[17]:


X_train,X_test=X_sorted[:56000], X_sorted[56000:]
y_train,y_test=y_sorted[:56000], y_sorted[56000:]
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)


# In[18]:


unique_classes_train = np.unique(y_train)
unique_classes_test = np.unique(y_test)
print("y_train:",unique_classes_train)
print("y_test:",unique_classes_test)


# In[19]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)



# In[20]:


unique_classes_train = np.unique(y_train)
unique_classes_test = np.unique(y_test)
print("y_train:",unique_classes_train)
print("y_test:",unique_classes_test)


# In[21]:


#UCZENIE JEDNA KLASA


# In[22]:


from sklearn.linear_model import SGDClassifier


# In[23]:


y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)
print(y_train_0)
print(np.unique(y_train_0))


# In[24]:


sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_0)


# In[25]:


y_train_pred=sgd_clf.predict(X_train)
y_test_pred=sgd_clf.predict(X_test)


# In[26]:


acc_train=sum(y_train_pred==y_train_0)/len(y_train_0)
acc_test=sum(y_test_pred==y_test_0)/len(y_test_0)


# In[27]:


print(acc_train,acc_test)


# In[28]:


accuracy_list = [acc_train, acc_test]

with open('sgd_acc.pkl', 'wb') as f:
    pickle.dump(accuracy_list, f)


# In[29]:


from sklearn.model_selection import cross_val_score


# In[30]:


score=cross_val_score(sgd_clf,X_train,y_train_0,cv=3,scoring="accuracy",n_jobs=-1)


# In[31]:


with open("sgd_cva.pkl", "wb") as file:
    pickle.dump(score, file)


# In[32]:


#UCZENIE WIELE KLAS


# In[33]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict


# In[34]:


sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train)


# In[35]:


y_pred = cross_val_predict(sgd_clf,X_test,y_test,cv=3,n_jobs=-1)


# In[36]:


conf_matrix = confusion_matrix(y_test, y_pred)


# In[37]:


print(conf_matrix)


# In[38]:


with open("sgd_cmx.pkl", "wb") as file:
    pickle.dump(conf_matrix, file)


# In[ ]:




