#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.neighbors
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt


# In[2]:


size=300
X=np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) +w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X,'y':y})
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')
X=df[['x']]
y=df['y']


# In[3]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[4]:


#REGRESJA LINIOWA


# In[5]:


#ZBIOR TESTOWY
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

lin_reg_pred_train=lin_reg.predict(X_train)
MSE_train=mean_squared_error(y_train,lin_reg_pred_train)

lin_reg_pred=lin_reg.predict(X_test)
MSE_test=mean_squared_error(y_test,lin_reg_pred)
print(MSE_test,MSE_train)

#plt.clf()
#plt.scatter(X,y,c="blue")
#X_new = np.arange(-3, 3, 0.001).reshape(-1, 1)
#plt.plot(X_new,lin_reg.predict(X_new),c="purple",label="lin_reg")
#plt.legend()
#plt.xlabel("X")
#plt.ylabel("y")
#plt.title("Linear regression")
#plt.savefig('lin_reg.png')
#plt.show()


# In[6]:


#KNN


# In[7]:


knn_reg_3 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_reg_3.fit(X_train,y_train)
knn_MSE_train_3=knn_reg_3.predict(X_train)
MSE_train_3=mean_squared_error(y_train,knn_MSE_train_3)

knn_MSE_test_3=knn_reg_3.predict(X_test)
MSE_test_3=mean_squared_error(y_test,knn_MSE_test_3)


knn_reg_5 = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn_reg_5.fit(X_train,y_train)
knn_MSE_train_5=knn_reg_5.predict(X_train)
MSE_train_5=mean_squared_error(y_train,knn_MSE_train_5)

knn_MSE_test_5=knn_reg_5.predict(X_test)
MSE_test_5=mean_squared_error(y_test,knn_MSE_test_5)

#plt.clf()
#plt.scatter(X,y,c="blue")
#X_new = np.arange(-3, 3, 0.001).reshape(-1, 1)
#plt.plot(X_new,knn_reg_3.predict(X_new),c="purple",label="KNN (k=3)")
#plt.plot(X_new,knn_reg_5.predict(X_new),c="pink",label="KNN (k=5)")
#plt.legend()
#plt.xlabel("X")
#plt.ylabel("y")
#plt.title("KNN Regression")
#
#plt.savefig('knn.png')
#plt.show()


# In[8]:


#WIELOMIANOWA


# In[9]:


odp=[]

#2
poly_feature_2 = PolynomialFeatures(degree=2,include_bias=False)
X_train_poly_2 = poly_feature_2.fit_transform(X_train)
X_test_poly_2 = poly_feature_2.transform(X_test)
    
model_2 = LinearRegression()
model_2.fit(X_train_poly_2, y_train)

y_pred_poly_2 = model_2.predict(X_test_poly_2)
y_train_pred_poly_2 = model_2.predict(X_train_poly_2)

mse_test_poly_2 = mean_squared_error(y_test, y_pred_poly_2)

mse_train_poly_2 = mean_squared_error(y_train, y_train_pred_poly_2)

poly_2_reg=model_2

odp.append((mse_train_poly_2,mse_test_poly_2))


# In[10]:


#3
poly_feature_3 = PolynomialFeatures(degree=3,include_bias=False)
X_train_poly_3 = poly_feature_3.fit_transform(X_train)
X_test_poly_3 = poly_feature_3.transform(X_test)
    

model_3 = LinearRegression()
model_3.fit(X_train_poly_3, y_train)

y_pred_poly_3 = model_3.predict(X_test_poly_3)
y_train_pred_poly_3 = model_3.predict(X_train_poly_3)

mse_test_poly_3 = mean_squared_error(y_test, y_pred_poly_3)

mse_train_poly_3 = mean_squared_error(y_train, y_train_pred_poly_3)

poly_3_reg=model_3

odp.append((mse_train_poly_3,mse_test_poly_3))


# In[11]:


#4
poly_feature_4 = PolynomialFeatures(degree=4,include_bias=False)
X_train_poly_4 = poly_feature_4.fit_transform(X_train)
X_test_poly_4 = poly_feature_4.transform(X_test)
    

model_4 = LinearRegression()
model_4.fit(X_train_poly_4, y_train)

y_pred_poly_4 = model_4.predict(X_test_poly_4)
y_train_pred_poly_4 = model_4.predict(X_train_poly_4)

mse_test_poly_4 = mean_squared_error(y_test, y_pred_poly_4)

mse_train_poly_4 = mean_squared_error(y_train, y_train_pred_poly_4)

poly_4_reg=model_4

odp.append((mse_train_poly_4,mse_test_poly_4))


# In[12]:


#5
poly_feature_5 = PolynomialFeatures(degree=5,include_bias=False)
X_train_poly_5 = poly_feature_5.fit_transform(X_train)
X_test_poly_5 = poly_feature_5.transform(X_test)


model_5 = LinearRegression() 
model_5.fit(X_train_poly_5, y_train)

y_pred_poly_5 = model_5.predict(X_test_poly_5)
y_train_pred_poly_5 = model_5.predict(X_train_poly_5)

mse_test_poly_5 = mean_squared_error(y_test, y_pred_poly_5)
mse_train_poly_5 = mean_squared_error(y_train, y_train_pred_poly_5)

poly_5_reg=model_5

odp.append((mse_train_poly_5,mse_test_poly_5))
print(odp)
print(model_5)


# In[13]:


#X_new = np.arange(-3, 3, 0.001).reshape(-1, 1)
#
#plt.scatter(X, y, c="blue")
#
#X_new_poly_2 = poly_feature_2.transform(X_new)
#X_new_poly_3 = poly_feature_3.transform(X_new)
#X_new_poly_4 = poly_feature_4.transform(X_new)
#X_new_poly_5 = poly_feature_5.transform(X_new)
#
#
#plt.plot(X_new, model_2.predict(X_new_poly_2), c="green", label="PR 2")
#plt.plot(X_new, model_3.predict(X_new_poly_3), c="yellow", label="PR 3")
#plt.plot(X_new, model_4.predict(X_new_poly_4), c="pink", label="PR 4")
#plt.plot(X_new, model_5.predict(X_new_poly_5), c="purple", label="PR 5")
#
#plt.legend()
#
#plt.xlabel("X")
#plt.ylabel("y")
#plt.title("Polynomial Regression")
#
#plt.savefig('pr.png')
#
#plt.show()


# In[14]:


#Zapisanie danych do DataFrame


# In[15]:


mse_df=pd.DataFrame({
    'train_mse':[MSE_train,MSE_train_3,MSE_train_5,odp[0][0],odp[1][0],odp[2][0],odp[3][0]],
    'test_mse':[MSE_test,MSE_test_3,MSE_test_5,odp[0][1],odp[1][1],odp[2][1],odp[3][1]]},
    index=['lin_reg', 'knn_3_reg', 'knn_5_reg', 'poly_2_reg', 'poly_3_reg', 'poly_4_reg', 'poly_5_reg']
)


# In[16]:


mse_df.head()


# In[17]:


mse_df.to_pickle('mse.pkl')


# In[18]:


#Zapis regresorow


# In[19]:


import pickle
regressors_with_features = [
    (lin_reg, None),
    (knn_reg_3, None),
    (knn_reg_5, None),
    (poly_2_reg,poly_feature_2),
    (poly_3_reg,poly_feature_3),
    (poly_4_reg,poly_feature_4),
    (poly_5_reg,poly_feature_5)
]
print(regressors_with_features)

with open('reg.pkl', 'wb') as f:
    pickle.dump(regressors_with_features, f)


# In[ ]:




