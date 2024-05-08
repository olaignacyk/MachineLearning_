#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN


# In[2]:


mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto') 
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# In[9]:


silhouette_scores = []
kmeans=KMeans(n_clusters=8,n_init=10)
kmeans.fit(X)
print(silhouette_score(X, kmeans.labels_))


# In[ ]:


#ZADANIE 2


# In[10]:


for k in range(8, 13):
    best_score = -1  
    best_model = None  
    for _ in range(10):  
        kmeans = KMeans(n_clusters=k, n_init=10).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_model = kmeans
        print(score)
    silhouette_scores.append(best_score)


# In[ ]:


print(silhouette_scores)


# In[ ]:


with open('kmeans_sil.pkl', 'wb') as f:
    pickle.dump(silhouette_scores, f)


# In[ ]:


#ZADANIE 5


# In[ ]:


kmeans = KMeans(n_clusters=10, n_init=10).fit(X)
cluster_labels = kmeans.labels_

conf_matrix = confusion_matrix(y, cluster_labels)

indices = [np.argmax(row) for row in conf_matrix]

sorted_indices = sorted(set(indices))

print(sorted_indices)
with open('kmeans_argmax.pkl', 'wb') as f:
    pickle.dump(sorted_indices, f)


# In[ ]:


#ZADANIE 6


# In[ ]:


distances = []
for i in range(300):
    for j in range(len(X)):
        if i != j:
            distance = np.linalg.norm(X[i] - X[j])
            if distance != 0:
                distances.append(distance)

sorted_distances = sorted(distances)[:10]

with open('dist.pkl', 'wb') as f:
    pickle.dump(sorted_distances, f)


# In[ ]:


s = np.mean(sorted_distances[:3])
dbscan_labels_count=[]
eps_values = np.arange(s, s + 0.1 * s, 0.04 * s)

for eps_ in eps_values:
    dbscan = DBSCAN(eps=eps_)
    labels = dbscan.fit_predict(X)

    unique_labels_count = len(set(labels))
    dbscan_labels_count.append(unique_labels_count)

with open('dbscan_len.pkl','wb') as f:
    pickle.dump(dbscan_labels_count,f)

    

