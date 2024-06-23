#!/usr/bin/env python
# coding: utf-8

# In[16]:


# Ładowanie danych


# In[36]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

from tensorflow.keras.applications.xception import preprocess_input
import pickle
from functools import partial


# In[18]:


import tensorflow_datasets as tfds

[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
    "tf_flowers",
    split=['train[:10%]', "train[10%:25%]", "train[25%:]"],
    as_supervised=True,
    with_info=True)


# In[19]:


class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples


# In[20]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
index = 0
sample_images = train_set_raw.take(9)
for image, label in sample_images:
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis("off")

plt.show(block=False)


# In[21]:


# Budowanie sieci CNN


# In[22]:


# Przygotowanie danych


# In[23]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    return resized_image, label


# In[24]:


import tensorflow as tf

batch_size = 32

train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)

valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)


# In[25]:


# Wyświetl próbkę danych po przetworzeniu
plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index] / 255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")
plt.show()


# In[26]:


# Budownie sieci


# In[27]:


# Zbuduj model
DefaultConv2D = partial(layers.Conv2D,kernel_size=3,activation='relu',padding="SAME")

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(n_classes, activation='softmax')
])

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Wyświetlenie podsumowania modelu
model.summary()


# In[28]:


# Trenowanie modelu
history = model.fit(train_set, validation_data=valid_set, epochs=10)
acc_train, acc_valid = history.history['accuracy'], history.history['val_accuracy']
acc_test = model.evaluate(test_set)


# In[29]:


# Zapisanie wyników do pliku
import pickle

with open('simple_cnn_acc.pkl', 'wb') as f:
    pickle.dump((acc_train, acc_valid, acc_test), f)

model.save('simple_cnn_flowers.keras')


# In[32]:


# Uczenie transferowe


# In[33]:


# Przygotowenie danych


# In[34]:


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label


# In[37]:


#plt.figure(figsize=(8, 8))
#sample_batch = train_set.take(1).repeat()
#for X_batch, y_batch in sample_batch:
#    for index in range(12):
#       plt.subplot(3, 4, index + 1)
#        plt.imshow(X_batch[index] / 2 + 0.5)
#        plt.title("Class: {}".format(class_names[y_batch[index]]))
#        plt.axis("off")
#plt.show()


# In[38]:


# Utwórz model bazowy
base_model = tf.keras.applications.Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Zablokuj warstwy modelu bazowego
for layer in base_model.layers:
    layer.trainable = False

# Dodaj warstwy do modelu
average_pooling_layer = layers.GlobalAveragePooling2D()(base_model.output)
output_layer = layers.Dense(n_classes, activation='softmax')(average_pooling_layer)

# Utwórz model
model = models.Model(inputs=base_model.input, outputs=output_layer)

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Wyświetlenie podsumowania modelu
model.summary()


# In[39]:


# Pierwsza faza uczenia: warstwy bazowe zamrożone
history1 = model.fit(train_set, epochs=5, validation_data=valid_set)

# Odblokowanie warstw modelu bazowego
for layer in base_model.layers:
    layer.trainable = True

# Ponowna kompilacja modelu z mniejszym learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Druga faza uczenia: wszystkie warstwy odblokowane
history2 = model.fit(train_set, epochs=10, validation_data=valid_set)


# In[40]:


# Ewaluacja modelu
acc_train = model.evaluate(train_set)[1]
acc_valid = model.evaluate(valid_set)[1]
acc_test = model.evaluate(test_set)[1]

# Zapisanie wyników do pliku
results = (acc_train, acc_valid, acc_test)
with open('xception_acc.pkl', 'wb') as file:
    pickle.dump(results, file)

# Zapisanie modelu do pliku
model.save('xception_flowers.keras')


# In[ ]:




