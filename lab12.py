#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pobieranie danych


# In[2]:


import numpy as np
import tensorflow as tf
import pandas as pd
import pickle


# In[3]:


tf.keras.utils.get_file(
    "bike_sharing_dataset.zip",
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    cache_dir=".",
    extract=True
)


# In[4]:


# Przygotowanie danych


# In[5]:


df = pd.read_csv('datasets/hour.csv',
                 parse_dates={'datetime': ['dteday', 'hr']},
                 date_format='%Y-%m-%d %H',
                 index_col='datetime'
)


# In[6]:


print((df.index.min(), df.index.max()))


# In[7]:


(365 + 366) * 24 - len(df)


# In[8]:


# Resampling i uzupełnianie brakujących danych
df_resampled = df.resample('H').asfreq()

# Uzupełnianie kolumn z liczbą wypożyczeń zerami
df_resampled[['casual', 'registered', 'cnt']] = df_resampled[['casual', 'registered', 'cnt']].fillna(0)

# Interpolacja dla kolumn z danymi pogodowymi
df_resampled[['temp', 'atemp', 'hum', 'windspeed']] = df_resampled[['temp', 'atemp', 'hum', 'windspeed']].interpolate()

# Wypełnianie brakujących wartości dla kolumn kategoryzowanych
df_resampled[['holiday', 'weekday', 'workingday', 'weathersit']] = df_resampled[['holiday', 'weekday', 'workingday', 'weathersit']].fillna(method='ffill')

# Sprawdzenie, czy DataFrame ma odpowiednią strukturę i nie zawiera brakujących wartości
print(df_resampled.notna().sum())


# In[9]:


df[['casual', 'registered', 'cnt', 'weathersit']].describe()


# In[10]:


df.casual /= 1e3
df.registered /= 1e3
df.cnt /= 1e3
df.weathersit /= 4


# In[11]:


df_2weeks = df[:24 * 7 * 2]
df_2weeks[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))


# In[12]:


df_daily = df.resample('W').mean()
df_daily[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))


# In[13]:


# Wskaźniki bazowe


# In[14]:


mae_daily = df['cnt'].diff(24).abs().mean() * 1e3
mae_weekly = df['cnt'].diff(24*7).abs().mean() * 1e3
mae_baseline = (mae_daily, mae_weekly)
print(mae_baseline)
with open('mae_baseline.pkl', 'wb') as f:
    pickle.dump(mae_baseline, f)


# In[15]:


# Predykcja przy pomocy sieci gęstej 


# In[16]:


cnt_train = df['cnt']['2011-01-01 00:00':'2012-06-30 23:00']
cnt_valid = df['cnt']['2012-07-01 00:00':]


# In[17]:


seq_len = 1 * 24
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_train.to_numpy(),
    targets=cnt_train[seq_len:],
    sequence_length=seq_len,
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_valid.to_numpy(),
    targets=cnt_valid[seq_len:],
    sequence_length=seq_len,
    batch_size=32
)


# In[18]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[seq_len])
])


# In[19]:


# Kompilacja modelu
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)

# Trening modelu
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=20
)

# Zapisanie modelu
model.save('model_linear.keras')

# Obliczenie MAE dla zbioru walidacyjnego
mae_linear = model.evaluate(valid_ds, return_dict=True)['mae']
mae_linear_tuple = (mae_linear,)
print(mae_linear_tuple)

# Zapisanie MAE do pliku
with open('mae_linear.pkl', 'wb') as f:
    pickle.dump(mae_linear_tuple, f)


# In[20]:


# Prosta sieć rekurencyjna


# In[21]:


# Definicja modelu prostej sieci rekurencyjnej
model_rnn1 = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])
])

# Kompilacja modelu
model_rnn1.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)

# Trening modelu
history_rnn1 = model_rnn1.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=20
)

# Zapisanie modelu
model_rnn1.save('model_rnn1.keras')

# Obliczenie MAE dla zbioru walidacyjnego
mae_rnn1 = model_rnn1.evaluate(valid_ds, return_dict=True)['mae']
mae_rnn1_tuple = (mae_rnn1,)
print(mae_rnn1_tuple)

# Zapisanie MAE do pliku
with open('mae_rnn1.pkl', 'wb') as f:
    pickle.dump(mae_rnn1_tuple, f)


# In[22]:


# Definicja rozbudowanej sieci rekurencyjnej
model_rnn32 = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 1]),
    tf.keras.layers.Dense(1)
])

# Kompilacja modelu
model_rnn32.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)

# Trening modelu
history_rnn32 = model_rnn32.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=20
)

# Zapisanie modelu
model_rnn32.save('model_rnn32.keras')

# Obliczenie MAE dla zbioru walidacyjnego
mae_rnn32 = model_rnn32.evaluate(valid_ds, return_dict=True)['mae']
mae_rnn32_tuple = (mae_rnn32,)
print(mae_rnn32_tuple)

# Zapisanie MAE do pliku
with open('mae_rnn32.pkl', 'wb') as f:
    pickle.dump(mae_rnn32_tuple, f)


# In[23]:


# Głęboka RNN


# In[24]:


# Definicja głębokiego modelu RNN
model_rnn_deep = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(32),
    tf.keras.layers.Dense(1)
])

# Kompilacja modelu
model_rnn_deep.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)

# Trening modelu
history_rnn_deep = model_rnn_deep.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=20
)

# Zapisanie modelu
model_rnn_deep.save('model_rnn_deep.keras')

# Obliczenie MAE dla zbioru walidacyjnego
mae_rnn_deep = model_rnn_deep.evaluate(valid_ds, return_dict=True)['mae']
mae_rnn_deep_tuple = (mae_rnn_deep,)
print(mae_rnn_deep_tuple)

# Zapisanie MAE do pliku
with open('mae_rnn_deep.pkl', 'wb') as f:
    pickle.dump(mae_rnn_deep_tuple, f)


# In[25]:


# Model wielowymiarowy


# In[26]:


# Przygotowanie danych
df = pd.read_csv('datasets/hour.csv',
                 parse_dates={'datetime': ['dteday', 'hr']},
                 date_format='%Y-%m-%d %H',
                 index_col='datetime')

# Uzupełnianie brakujących wartości (procedura z wcześniejszej części)
df_resampled = df.resample('H').asfreq()
df_resampled['casual'].fillna(0, inplace=True)
df_resampled['registered'].fillna(0, inplace=True)
df_resampled['cnt'].fillna(0, inplace=True)
df_resampled[['temp', 'atemp', 'hum', 'windspeed']] = df_resampled[['temp', 'atemp', 'hum', 'windspeed']].interpolate()
df_resampled[['holiday', 'weekday', 'workingday', 'weathersit']] = df_resampled[['holiday', 'weekday', 'workingday', 'weathersit']].ffill()

# Normalizacja odpowiednich kolumn
df_resampled.casual /= 1e3
df_resampled.registered /= 1e3
df_resampled.cnt /= 1e3
df_resampled.weathersit /= 4

# Przygotowanie cech i targetu
features = ['cnt', 'weathersit', 'atemp', 'workingday']
target = 'cnt'

# Podział danych na zbiór uczący i walidacyjny
cnt_train = df_resampled[features]['2011-01-01 00:00':'2012-06-30 23:00']
cnt_valid = df_resampled[features]['2012-07-01 00:00':]

seq_len = 24
batch_size = 32

# Tworzenie zbiorów danych TensorFlow
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_train.to_numpy(),
    targets=cnt_train[target][seq_len:],
    sequence_length=seq_len,
    batch_size=batch_size,
    shuffle=True,
    seed=42
)

valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_valid.to_numpy(),
    targets=cnt_valid[target][seq_len:],
    sequence_length=seq_len,
    batch_size=batch_size
)

# Definicja modelu wielowymiarowego RNN
model_rnn_mv = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, len(features)]),
    tf.keras.layers.Dense(1)
])

# Kompilacja modelu
model_rnn_mv.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    loss=tf.keras.losses.Huber(),
    metrics=['mae']
)

# Trening modelu
history_rnn_mv = model_rnn_mv.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=20
)

# Zapisanie modelu
model_rnn_mv.save('model_rnn_mv.keras')

# Obliczenie MAE dla zbioru walidacyjnego
mae_rnn_mv = model_rnn_mv.evaluate(valid_ds, return_dict=True)['mae'] * 1e3
mae_rnn_mv_tuple = (mae_rnn_mv,)
print(mae_rnn_mv_tuple)

# Zapisanie MAE do pliku
with open('mae_rnn_mv.pkl', 'wb') as f:
    pickle.dump(mae_rnn_mv_tuple, f)


# In[ ]:




