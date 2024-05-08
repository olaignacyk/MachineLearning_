#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import pickle


# In[2]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer()
X_bc = data_breast_cancer.data
y_bc =data_breast_cancer.target


# In[3]:


from sklearn.datasets import load_iris
data_iris = load_iris()
X_iris = data_iris.data
y_iris = data_iris.target


# In[4]:


#Zadanie 1


# In[5]:


pca_bc = PCA(n_components=0.9)
X_bc_pca = pca_bc.fit_transform(X_bc)

pca_iris = PCA(n_components=0.9)
X_iris_pca = pca_iris.fit_transform(X_iris)

scaler_bc = StandardScaler()
X_bc_scaled = scaler_bc.fit_transform(X_bc)
pca_bc_scaled = PCA(n_components=0.9)
X_bc_scaled_pca = pca_bc_scaled.fit_transform(X_bc_scaled)

scaler_iris = StandardScaler()
X_iris_scaled = scaler_iris.fit_transform(X_iris)
pca_iris_scaled = PCA(n_components=0.9)
X_iris_scaled_pca = pca_iris_scaled.fit_transform(X_iris_scaled)



# In[6]:


pca_bc.explained_variance_ratio_


# In[7]:


pca_bc_scaled.explained_variance_ratio_


# In[8]:


variance_ratios_bc_scaled = pca_bc_scaled.explained_variance_ratio_
variance_ratios_iris_scaled = pca_iris_scaled.explained_variance_ratio_

with open('pca_bc.pkl', 'wb') as f:
    pickle.dump(variance_ratios_bc_scaled.tolist(), f)

with open('pca_ir.pkl','wb') as f:
    pickle.dump(variance_ratios_iris_scaled.tolist(),f)


# In[9]:


print(variance_ratios_bc_scaled)
print(X_bc.shape,'-->', X_bc_pca.shape)


# In[10]:


print(variance_ratios_iris_scaled)
print(X_iris.shape,'-->', X_iris_pca.shape)


# In[11]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_bc[:, 0], X_bc[:, 1], c=data_breast_cancer.target, cmap='viridis', alpha=0.7)
plt.title('Breast Cancer Dataset przed redukcją wymiarowości')


plt.subplot(1, 2, 2)
plt.scatter(X_iris[:, 0], X_iris[:, 1], c=data_iris.target, cmap='viridis', alpha=0.7)
plt.title('Iris Dataset przed redukcją wymiarowości')


plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_bc_pca[:, 0], np.zeros_like(X_bc_pca[:, 0]), c=data_breast_cancer.target, cmap='viridis', alpha=0.7)
plt.title('Breast Cancer Dataset po redukcji wymiarowości')


plt.subplot(1, 2, 2)
plt.scatter(X_iris_pca[:, 0], np.zeros_like(X_iris_pca[:, 0]), c=data_iris.target, cmap='viridis', alpha=0.7)
plt.title('Iris Dataset po redukcji wymiarowości')


plt.tight_layout()
plt.show()


# In[12]:


#Zadanie 2


# In[13]:


feature_indices_bc = []
for component in pca_bc_scaled.components_:
    max_abs_feature_index = np.argmax(np.abs(component))
    feature_indices_bc.append(max_abs_feature_index)
print(feature_indices_bc)


# In[14]:


with open('idx_bc.pkl', 'wb') as f:
    pickle.dump(feature_indices_bc, f)


# In[15]:


feature_indices = []
for component in pca_iris_scaled.components_:
    max_feature_index = np.argmax(np.abs(component))
    feature_indices.append(max_feature_index)
    
print(feature_indices)


# In[16]:


with open('idx_ir.pkl', 'wb') as f:
    pickle.dump(feature_indices, f)


# In[ ]:





# In[ ]:




