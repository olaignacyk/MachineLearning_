#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn.tree import export_graphviz,DecisionTreeClassifier,DecisionTreeRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score, mean_squared_error
import pickle
import matplotlib.pyplot as plt


# In[2]:


import graphviz


# In[3]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[4]:


data_breast_cancer


# In[5]:


#ZADANIE 1


# In[27]:


X = data_breast_cancer.data[['mean texture', 'mean symmetry']].values
y = data_breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[28]:


best_f1_score_train = 0
best_f1_score_test = 0
best_depth=None

for depth in range(1, 21):
    model = DecisionTreeClassifier(max_depth=depth)

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)

    f1_train = f1_score(y_train, y_pred_train)

    y_pred_test = model.predict(X_test)

    f1_test = f1_score(y_test, y_pred_test)

    if f1_train > best_f1_score_train and f1_test > best_f1_score_test:
        best_f1_score_train = f1_train
        best_f1_score_test = f1_test
        best_depth = depth

    print(f"Dla {depth}:")
    print("F1 train:", f1_train)
    print("F1 test:" ,f1_test)
    
print("Najlepsza glebokosc:",best_depth)


# In[41]:


model = DecisionTreeClassifier(max_depth=2)
model.fit(X_train, y_train)


# In[30]:


str_dot = export_graphviz(model,
                          out_file=None,
                          feature_names=["mean texture", "mean symmetry"], 
                          class_names=[str(num)+", "+name 
                                       for num,name in 
                                       zip(set(data_breast_cancer.target),
                                        data_breast_cancer.target_names)], 
                          rounded=True, 
                          filled=True)

graph = graphviz.Source(str_dot)
graph.render(filename='bc', format='png')


# In[31]:


y_train_pred = model.predict(X_train)
f1_train = f1_score(y_train, y_train_pred)
accuracy_train = accuracy_score(y_train, y_train_pred)

y_test_pred = model.predict(X_test)
f1_test = f1_score(y_test, y_test_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

f1acc_info = [best_depth, f1_train, f1_test, accuracy_train, accuracy_test]

with open('f1acc_tree.pkl', 'wb') as f:
    pickle.dump(f1acc_info, f)


# In[32]:


#ZADANIE 2


# In[33]:


size=300
X=np.random.rand(size)*5-2.5
w4,w3,w2,w1,w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4 
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')
X=df[['x']]
y=df['y']


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[35]:


best_depth = None
best_mse_train = float('inf')
best_mse_test = float('inf')


# In[36]:


for depth in range(1, 21):
    model = DecisionTreeRegressor(max_depth=depth)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    
    y_test_pred = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_test_pred)

    print(f'Wartosci dla{depth}')
    print(mse_train)
    print(mse_test)
    if mse_test < best_mse_test and mse_train < best_mse_train:
        best_depth = depth
        best_mse_train = mse_train
        best_mse_test = mse_test

best_model = DecisionTreeRegressor(max_depth=best_depth)
best_model.fit(X_train, y_train)
y_train_pred = best_model.predict(X_train)
mse_train = mean_squared_error(y_train, y_train_pred)
    
y_test_pred = best_model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)

print("Najlepsza głębokość drzewa decyzyjnego:", best_depth)
print("MSE dla zbioru uczącego:", best_mse_train)
print("MSE dla zbioru testowego:", best_mse_test)


# In[37]:


X_range = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
y_range_pred = best_model.predict(X_range)


# In[38]:


plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Dane', alpha=0.5)
plt.plot(X_range, y_range_pred, color='red', label='Predykcja drzewa decyzyjnego')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Porównanie drzewa decyzyjnego z danymi')
plt.legend()
plt.grid(True)
plt.show()


# In[39]:


mse_info = [best_depth, best_mse_train, best_mse_test]
print(mse_info)
with open('mse_tree.pkl', 'wb') as f:
    pickle.dump(mse_info, f)


# In[40]:


str_dot = export_graphviz(best_model,
                          out_file=None,
                          feature_names=['x'], 
                          rounded=True, 
                          filled=True)

graph = graphviz.Source(str_dot)
graph.render(filename='reg', format='png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




