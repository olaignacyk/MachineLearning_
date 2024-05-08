#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('housing.csv.gz')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


print(df['ocean_proximity'].dtype)


# In[6]:


print(df['ocean_proximity'].value_counts())


# In[7]:


print(df['ocean_proximity'].describe())


# In[8]:


df.hist(bins=50,figsize=(20,15))


# In[9]:


df.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1,figsize=(7,4))


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


df.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,figsize=(7,3),colorbar=True,s=df["population"]/100,label="population",c="median_house_value",cmap=plt.get_cmap("jet"))


# In[12]:


'''
DataFrame.hist(): Metoda ta generuje histogramy dla wszystkich numerycznych kolumn w ramce danych.
    -bins: Liczba przedziałów histogramu.
    -figsize: Rozmiar wygenerowanego wykresu.
    
DataFrame.plot(kind='scatter'): Metoda ta generuje wykres punktowy (scatter plot) na podstawie dwóch kolumn z ramki danych.
    -kind: Typ wykresu, w tym przypadku 'scatter'.
    -x, y: Nazwy kolumn, które będą używane jako wartości na osiach x i y.
    -alpha: Przezroczystość punktów (0 - całkowicie przeźroczyste, 1 - całkowicie nieprzezroczyste).
    -figsize: Rozmiar wygenerowanego wykresu.
    
DataFrame.plot(kind='scatter'): Również generuje wykres punktowy (scatter plot), ale z dodatkowymi opcjami.
kind: Typ wykresu, w tym przypadku 'scatter'.
    -x, y: Nazwy kolumn, które będą używane jako wartości na osiach x i y.
    -alpha: Przezroczystość punktów (0 - całkowicie przeźroczyste, 1 - całkowicie nieprzezroczyste).
    -figsize: Rozmiar wygenerowanego wykresu.
    -colorbar: Określa, czy wyświetlić pasek kolorów (tylko dla kolorowego scatter plot).
    -s: Rozmiar punktów (może być tablicą lub skalarem).
    -label: Etykieta dla punktów (tylko dla kolorowego scatter plot).
    -c: Nazwa kolumny, której wartości będą używane do ustalenia kolorów punktów.
    -cmap: Mapa kolorów do użycia dla punktów.
'''


# In[13]:


# Zapisujemy wykres 1
plt.figure(figsize=(20, 15))
df.hist(bins=50)
plt.savefig('obraz1.png')
plt.close()
# Zapisujemy wykres 2
plt.figure(figsize=(7, 4))
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.savefig('obraz2.png')
plt.close()
# Zapisujemy wykres 3
plt.figure(figsize=(7, 3))
plt.scatter(x=df["longitude"], y=df["latitude"], alpha=0.4, c=df["median_house_value"], cmap=plt.get_cmap("jet"))
plt.colorbar(label="Median House Value")
plt.savefig('obraz3.png')
plt.close()


# In[14]:


# Wybieramy tylko kolumny numeryczne
numerical_columns = df.select_dtypes(include=['number'])

# Obliczamy macierz korelacji
correlation_matrix = numerical_columns.corr()['median_house_value'].reset_index()

# Zmieniamy nazwy kolumn
correlation_matrix = correlation_matrix.rename(columns={'index': 'atrybut', 'median_house_value': 'współczynnik_korelacji'})

# Zapisujemy wyniki do pliku CSV
correlation_matrix.to_csv('korelacja.csv', index=False)

print(correlation_matrix)


# In[15]:


import seaborn as sns


# In[16]:


sns.pairplot(df)


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(train_set),len(test_set)


# In[19]:


'''
test_size: Określa procent danych, które mają być przypisane do zbioru testowego. Na przykład, test_size=0.2 oznacza, że 20% danych będzie przypisane do zbioru testowego, a 80% do zbioru treningowego.
random_state: Określa ziarno losowości używane podczas podziału danych. Ustawienie tego parametru na stałą liczbę zapewnia, że podział będzie deterministyczny
'''


# In[20]:


print(train_set.head())
print(test_set.head())


# In[21]:


numerical_columns_train = train_set.select_dtypes(include=['number'])
numerical_columns_test = test_set.select_dtypes(include=['number'])

# Obliczamy macierz korelacji dla zbioru uczącego
correlation_matrix_train = numerical_columns_train.corr()

# Obliczamy macierz korelacji dla zbioru testującego
correlation_matrix_test = numerical_columns_test.corr()

# Wyświetlamy macierze korelacji
print("Macierz korelacji dla zbioru uczącego:")
print(correlation_matrix_train)

print("\nMacierz korelacji dla zbioru testującego:")
print(correlation_matrix_test)


# In[22]:


# Zapisanie zbioru uczącego do pliku pickle
train_set.to_pickle('train_set.pkl')

# Zapisanie zbioru testowego do pliku pickle
test_set.to_pickle('test_set.pkl')


# In[ ]:




