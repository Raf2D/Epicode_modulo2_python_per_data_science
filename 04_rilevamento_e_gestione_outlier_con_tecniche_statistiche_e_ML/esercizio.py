"""
Esercizio 1: 
crea un dataframe con almento 10 valori numerici e individua gli outlier usando la deviazione standard
(soglia 2)

Esercizio 2:
genera un dataset di 20 valori, applica il medoto IQR e rimuovi le righe con outlier.
mostra il dataset pulito.

Esercizio 3: 
usa un dataset con due colonne numeriche (es altezza e peso). Individua gli outlier e visualizza
i risultati in un dataframe come una colonna aggiuntiva che indichi i valori anomali.
"""

import numpy as np 
import pandas as pd 

# Esercizio 1 

data = {"Valori": [100, 200, 150, 175, 210, 1500, 2000, 120, 245]}

df = pd.DataFrame(data)
media = df["Valori"].mean()
dev_std = df["Valori"].std()

df["Outlier"] = abs(df["Valori"] - media) > 2 * dev_std # per selezionare gli outlier

print("\n-------- Applicazione metodo Z - SCORE ---------\n")
print("Soglia: ", 2 * dev_std)
print(df)

# Esercizio 2 

# Dataset di 20 valori (18 dati "normali" + 2 outlier evidenti)
data2 = {
    "Valori": [
        45, 52, 48, 50, 55, 42, 49, 51, 53, 47, 
        46, 54, 50, 48, 52, 51, 49, 53,
        5,    # Outlier inferiore (troppo basso)
        120   # Outlier superiore (troppo alto)
    ]
}

df2 = pd.DataFrame(data2)

print("\n-------- Applicazione metodo IQR ---------\n")

Q1 = df2["Valori"].quantile(0.25)
Q3 = df2["Valori"].quantile(0.75)

IQR = Q3 - Q1 

limite_basso = Q1 - 1.5 * IQR 
limite_alto = Q3 + 1.5 * IQR 

df2["Outlier"] = (df2["Valori"] < limite_basso) | (df2["Valori"] > limite_alto)

print("\nDATA FRAME PRIMA DELLA PULIZIA OUTLIER\n")
print(df2)

df2_pulito = df2[df2["Outlier"] == False]
print("\nDATA FRAME DOPO DELLA PULIZIA OUTLIER\n")
print(df2_pulito)


# Esercizio 3

data3 = ({
    "Altezza":[144, 160.5, 172.5, 180, 182, 199, 230, 177],
    "Peso":[60, 70, 75, 80, 82, 110, 120, 74]
})

# Isolation Forest
 
print("\n-------- Applicazione Isolation Forest ---------\n")

from sklearn.ensemble import IsolationForest

df3 = pd.DataFrame(data3)
model = IsolationForest(contamination = "auto", random_state = 40) # impostato ad auto per rilevazione automatica
df3["Outlier"] = model.fit_predict(df3[["Altezza","Peso"]]) # un unico data frame con tutte le colonne

print(df3)