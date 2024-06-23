#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier,BaggingClassifier,RandomForestClassifier,GradientBoostingRegressor,AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import numpy as np


# In[2]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[3]:


data_breast_cancer


# In[4]:


X = data_breast_cancer.data[['mean texture', 'mean symmetry']].values
y = data_breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[5]:


log_clf = LogisticRegression()
knn_clf = KNeighborsClassifier()
tree_clf = DecisionTreeClassifier()


# In[6]:


voting_clf = VotingClassifier(
    estimators=[('lr', log_clf),
                ('knc', knn_clf),
                ('dtc', tree_clf)],
    voting='hard')
voting_clf.fit(X_train,y_train)


# In[ ]:





# In[7]:


hard=[]
for clf in (log_clf, knn_clf, tree_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc_test=accuracy_score(y_test, y_pred)
    y_pred = clf.predict(X_train)
    acc_train=accuracy_score(y_train, y_pred)
    hard.append((acc_train,acc_test))


# In[8]:


print(hard)


# In[9]:


voting_clf_soft = VotingClassifier(
    estimators=[('lr', log_clf),
                ('knc', knn_clf),
                ('dtc', tree_clf)],
    voting='soft')
voting_clf_soft.fit(X_train,y_train)


# In[10]:


soft=[]
for clf in (log_clf, knn_clf, tree_clf, voting_clf_soft):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc_test=accuracy_score(y_test, y_pred)
    y_pred = clf.predict(X_train)
    acc_train=accuracy_score(y_train, y_pred)
    soft.append((acc_train,acc_test))
print(soft)


# In[11]:


result=[]
for clf in (log_clf, knn_clf, tree_clf,voting_clf,voting_clf_soft):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc_test=accuracy_score(y_test, y_pred)
    y_pred = clf.predict(X_train)
    acc_train=accuracy_score(y_train, y_pred)
    result.append((acc_train,acc_test))
print(result)


# In[12]:


with open('acc_vote.pkl', 'wb') as f:
    pickle.dump(result, f)


# In[13]:


classifiers=[log_clf,knn_clf,tree_clf,voting_clf,voting_clf_soft]
print(classifiers)


# In[14]:


with open('vote.pkl', 'wb') as f:
    pickle.dump(classifiers, f)


# In[15]:


#Bagging
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,
                            bootstrap=True)
bag_clf.fit(X_train, y_train)

y_pred = bag_clf.predict(X_test)
acc_test=accuracy_score(y_test, y_pred)
y_pred = bag_clf.predict(X_train)
acc_train=accuracy_score(y_train, y_pred)
acc_bag_clf=(acc_train,acc_test)


# In[16]:


#Bagging z wykorzystaniem 50% instancji
bag_clf_50 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30,max_samples=0.5,
                            bootstrap=True)
bag_clf_50.fit(X_train, y_train)

y_pred = bag_clf_50.predict(X_test)
acc_test=accuracy_score(y_test, y_pred)
y_pred = bag_clf_50.predict(X_train)
acc_train=accuracy_score(y_train, y_pred)
acc_bag_clf_50=(acc_train,acc_test)


# In[17]:


#Pasting
pasting_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False)
pasting_clf.fit(X_train, y_train)
y_pred_pasting_clf = pasting_clf.predict(X_test)

y_pred = pasting_clf.predict(X_test)
acc_test=accuracy_score(y_test, y_pred)
y_pred = pasting_clf.predict(X_train)
acc_train=accuracy_score(y_train, y_pred)
acc_pasting_clf=(acc_train,acc_test)


# In[18]:


#Pasting z wykorzystaniem 50% instancji
pasting_clf_50 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5, bootstrap=False)
pasting_clf_50.fit(X_train, y_train)
y_pred_pasting_clf_50 = pasting_clf_50.predict(X_test)

y_pred = pasting_clf_50.predict(X_test)
acc_test=accuracy_score(y_test, y_pred)
y_pred = pasting_clf_50.predict(X_train)
acc_train=accuracy_score(y_train, y_pred)
acc_pasting_clf_50=(acc_train,acc_test)


# In[19]:


#Random Forest
rnd_clf = RandomForestClassifier(n_estimators=30)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

y_pred = rnd_clf.predict(X_test)
acc_test=accuracy_score(y_test, y_pred)
y_pred = rnd_clf.predict(X_train)
acc_train=accuracy_score(y_train, y_pred)
acc_rnd_clf=(acc_train,acc_test)


# In[20]:


#AdaBoost
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=30)
ada_clf.fit(X_train, y_train)
y_pred_ada = ada_clf.predict(X_test)

y_pred = ada_clf.predict(X_test)
acc_test=accuracy_score(y_test, y_pred)
y_pred = ada_clf.predict(X_train)
acc_train=accuracy_score(y_train, y_pred)
acc_ada_clf=(acc_train,acc_test)


# In[21]:


#Gradient Boosting.

gbrt = GradientBoostingRegressor(n_estimators=30)
gbrt.fit(X_train, y_train)

y_pred = gbrt.predict(X_test).round()
acc_test=accuracy_score(y_test, y_pred)
y_pred = gbrt.predict(X_train).round()
acc_train=accuracy_score(y_train, y_pred)
accuracy_gbrt = (acc_train, acc_test)


# In[22]:


acc_scores = [acc_bag_clf,acc_bag_clf_50,acc_pasting_clf,acc_pasting_clf_50,acc_rnd_clf,acc_ada_clf,accuracy_gbrt]
for i in acc_scores:
    print(i)


# In[23]:


with open('acc_bag.pkl', 'wb') as f:
    pickle.dump(acc_scores, f)


# In[24]:


classifiers2=[bag_clf,bag_clf_50,pasting_clf,pasting_clf_50,rnd_clf,ada_clf,gbrt]
print(classifiers2)


# In[25]:


with open('bag.pkl', 'wb') as f:
    pickle.dump(classifiers2, f)


# In[28]:


X_ = data_breast_cancer.data
y_ = data_breast_cancer.target
X_train_, X_test_, y_train_, y_test_ = train_test_split(X_, y_, test_size=0.2)

bag_clf_ = BaggingClassifier(max_features=2,
    n_estimators=30,
    max_samples=0.5
)

bag_clf_.fit(X_train_, y_train_)

y_pred_train_ = bag_clf_.predict(X_train_)
acc_train = accuracy_score(y_train_, y_pred_train_)

y_pred_test_ = bag_clf_.predict(X_test_)
acc_test = accuracy_score(y_test_, y_pred_test_)

accuracies = [acc_train, acc_test]
with open('acc_fea.pkl', 'wb') as f:
    pickle.dump(accuracies, f)
print(accuracies)
with open('fea.pkl', 'wb') as f:
    pickle.dump([bag_clf_], f)
print(bag_clf_)



# In[29]:


accuracies_train = []
accuracies_test = []
features_list = []

for i, estimator in enumerate(bag_clf.estimators_):
    y_pred_train = estimator.predict(X_train)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    y_pred_test = estimator.predict(X_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    features_idx = bag_clf.estimators_features_[i]
    features = [f"Feature_{idx}" for idx in features_idx]
    
    accuracies_train.append(acc_train)
    accuracies_test.append(acc_test)
    features_list.append(features)

df = pd.DataFrame({
    'Accuracy Train': accuracies_train,
    'Accuracy Test': accuracies_test,
    'Features': features_list
})

df_sorted = df.sort_values(by=['Accuracy Train', 'Accuracy Test'], ascending=False)


with open('acc_fea_rank.pkl', 'wb') as f:
    pickle.dump(df_sorted, f)
df_sorted


# In[ ]:





# In[ ]:




