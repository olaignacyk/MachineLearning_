#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping


# In[2]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() 
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

X_train = X_train / 255.0
X_test = X_test / 255.0


# In[3]:


import matplotlib.pyplot as plt 
plt.imshow(X_train[2137], cmap="binary") 
plt.axis('off')
plt.show()


# In[4]:


class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
               "sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[2137]]


# In[5]:


model = Sequential()

model.add(Flatten(input_shape=[28, 28]))

model.add(Dense(300, activation='relu'))

model.add(Dense(100, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
model.summary()




# In[6]:


log_dir = "image_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(X_train, y_train, epochs=20, validation_split=0.1, callbacks=[tensorboard_callback])


# In[7]:


image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()


# In[8]:


model.save('fashion_clf.keras')


# In[9]:


from sklearn.datasets import fetch_california_housing 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler


# In[10]:


housing = fetch_california_housing()


# In[11]:


X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)


# In[12]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# In[13]:


model = Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=X_train.shape[1:]))
model.add(tf.keras.layers.Normalization(axis=-1))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

model.compile(loss="mean_squared_error",
              optimizer="adam",
              metrics=[tf.keras.metrics.RootMeanSquaredError()])


# In[14]:


early_stopping_callback = EarlyStopping(patience=5, min_delta=0.01, verbose=1)
log_dir = "housing_logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# In[ ]:





# In[15]:


model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), 
          callbacks=[early_stopping_callback, tensorboard_callback])


# In[ ]:





# In[16]:


model.save('reg_housing_1.keras')


# In[17]:


def create_model():
    model = Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=X_train.shape[1:]))
    model.add(tf.keras.layers.Normalization(axis=-1))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error",
                  optimizer="adam",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

model2 = create_model()
model2.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), 
           callbacks=[early_stopping_callback, tensorboard_callback])


model3 = create_model()
model3.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), 
           callbacks=[early_stopping_callback, tensorboard_callback])


# In[18]:


model2.save('reg_housing_2.keras')


# In[19]:


model3.save('reg_housing_3.keras')


# In[ ]:





# In[ ]:




