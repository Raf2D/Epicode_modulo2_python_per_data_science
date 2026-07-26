"""
Esercizio 1 
Crea un dataframe con 10 righe e una colonna con valori numerici e alcuni NaN. 
Sostituisci i mancanti con valori costanti per esempio 0.
Esercizio 2 
Genera un dataframe con 3 colonne numeriche e alcuni valori mancanti. 
Sostituisci i mancanti della prima colonna con la media, quelli della seconda con la mediana, 
quelli della terza con la moda.
Esercizio 3
Usa KNNimputer dataset con almeno 2 colonne numeriche e valori mancanti. 
Prova diversi valori di n_neighbors e confronta i risultati ottenuti.
"""

import numpy as np 
import pandas as pd 


# Esercizio 1 

data = {
    'Stipendi':[12000,23000,34000,np.nan,66000,21000,np.nan,45000,np.nan,89000]
}

df = pd.DataFrame(data)

print("\nDataFrame prima della sostituzione.\n")
print(df)

df.fillna(0, inplace=True)


print("\nDataFrane dopo la sostituzione.\n")
print(df)

# Esercizio 2

data2 = {
    'Dipendenti':["Raffaele","Giuseppe","Roberto","Antonella","Alessia","Priscilla","Alessio","Gianluca","Pasquale","Alberto"],
    'Stipendi':[12000,23000,34000,np.nan,66000,21000,np.nan,45000,np.nan,89000],
    'Età':[23,24,25,np.nan,34,31,np.nan,43,np.nan,27],
    'Categoria_protetta':["SI","NO","SI","NO",np.nan,"NO","NO",np.nan,"SI","NO"]
}

df2 = pd.DataFrame(data2)

print("\nDataFrame prima della sostituzione.\n")
print(df2)

df2['Età'] = df2['Età'].fillna(df2['Età'].mean())
df2['Stipendi'] = df2['Stipendi'].fillna(df2['Stipendi'].median())
df2['Categoria_protetta'] = df2['Categoria_protetta'].fillna(df2['Categoria_protetta'].mode()[0])


print("\nDataFrane dopo la sostituzione.\n")
print(df2)

# Esercizio 3 

from sklearn.impute import KNNImputer 

data3 = pd.DataFrame({
    'Feature1':[1,2,np.nan,6,8],
    'Feature2':["Giovanni","Teresa","Giuseppe","Alessia","Giacoma"],
    'Feature3':[10.5,np.nan,12.34,np.nan,123.12345],
    'Feature4':["Roma","Alessandria","Bisceglie","Ruvo","Giovinazzo"],
    'Feature5':[1000,2000,np.nan,1500,1900],
    'Feature6':[10000000,7000000,np.nan,20000000,np.nan],
    'Feature7':[769, 876, np.nan, 876.542, 999.99]
})

# Selezioniamo solo le colonne numeriche 
numeric_cols = data3.select_dtypes(include = [np.number]).columns

# Creiamo una copia del dataframe originale 
data_imputed = data3.copy()

print("\n__________ Algoritmo KNN ______________\n")

# iniziamo l'imputer con 2 vicini 

print("\n---- Iniziamo l'imputer con 2 vicini ----\n")
imputer = KNNImputer(n_neighbors = 2)

# Applichiamo l'imputazione solo sulle colonne numeriche

data_imputed[numeric_cols] = imputer.fit_transform(data3[numeric_cols])

print(data_imputed)

# proseguiamo con 5 vicini 

print("\n---- Proseguiamo con 5 vicini ----\n")

imputer = KNNImputer(n_neighbors = 5)

# Applichiamo l'imputazione 

data_imputed[numeric_cols] = imputer.fit_transform(data3[numeric_cols])

print(data_imputed)

# proseguiamo con 7 vicini 

print("\n---- Proseguiamo con 7 vicini ----")

imputer = KNNImputer(n_neighbors= 7)

# Applichimo l'imputazione 

data_imputed[numeric_cols] = imputer.fit_transform(data3[numeric_cols])

print(data_imputed)

