#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import reciprocal
from scikeras.wrappers import KerasRegressor
import keras_tuner as kt


# In[2]:


housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,y_train_full, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# In[3]:


#BUDOWANIE MODELU


# In[4]:


def build_model(n_hidden, n_neurons, optimizer, learning_rate):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=X_train.shape[1:]))
    for _ in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))

    if optimizer == 'sgd':
        optimizer_instance = keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'nesterov':
        optimizer_instance = keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)
    elif optimizer == 'adam':
        optimizer_instance = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(loss="mse", optimizer=optimizer_instance)
    return model


# In[5]:


param_distribs = {
    "model__n_hidden": [0, 1, 2, 3],
    "model__n_neurons": np.arange(1, 101),
    "model__learning_rate": reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
    "model__optimizer": ['adam', 'sgd', 'nesterov']
}


# In[6]:


es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)

keras_reg = KerasRegressor(build_model, callbacks=[es])

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=5, cv=3, verbose=2)

rnd_search_cv.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), verbose=0)


# In[7]:


best_params = rnd_search_cv.best_params_
print(best_params)


# In[8]:


print(rnd_search_cv)


# In[9]:


with open("rnd_search_params.pkl", "wb") as f:
    pickle.dump(best_params, f)

with open("rnd_search_scikeras.pkl", "wb") as f:
    pickle.dump(rnd_search_cv, f)


# In[10]:


#KERAS TUNER


# In[18]:


def build_model_kt(hp):
    n_hidden = hp.Int('n_hidden', min_value=0, max_value=3, default=2)
    n_neurons = hp.Int('n_neurons', min_value=1, max_value=100)
    learning_rate = hp.Float('learning_rate', min_value=3e-4, max_value=3e-2, sampling='log')
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'nesterov'])
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=X_train.shape[1:]))
    for _ in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_neurons, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    
    if optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer == 'nesterov':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov=True)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(loss='mse', optimizer=opt, metrics=['mse'])
    return model


# In[19]:


random_search_tuner = kt.RandomSearch(
    build_model_kt, objective='val_mse', max_trials=10, overwrite=True,
    directory='my_california_housing', project_name='my_rnd_search', seed=42
)


# In[20]:


root_logdir = os.path.join(random_search_tuner.project_dir, 'tensorboard')
tb = tf.keras.callbacks.TensorBoard(root_logdir)


# In[21]:


es = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=1.0, verbose=1)

random_search_tuner.search(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid), callbacks=[es, tb])

best_hps = random_search_tuner.get_best_hyperparameters()[0]


# In[22]:


print(best_hps.values)


# In[ ]:





# In[23]:


import pickle
with open('kt_search_params.pkl', 'wb') as f:
    pickle.dump(best_hps.values, f)

best_model = random_search_tuner.get_best_models()[0]
best_model.save('kt_best_model.keras')


# In[ ]:





# In[ ]:





# In[ ]:




