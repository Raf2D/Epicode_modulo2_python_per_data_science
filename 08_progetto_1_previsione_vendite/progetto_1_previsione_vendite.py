"""
Traccia:
Lavorare su un dataset di vendite storiche per comprendere i dati, pulirli, trasformarli e produrre delle

previsioni di base usando metodi semplici e direttamente implementabili con Pandas e Python “puro”.

Dataset (esempio di struttura)

Data — data della vendita

Prodotto — nome o codice prodotto

Vendite — quantità venduta quel giorno

Prezzo — prezzo unitario (opzionale)

I dati possono contenere valori mancanti, duplicati o inconsistenze.

Consegna

Parte 1 - Caricamento e esplorazione dati
1. Leggere il dataset con Pandas.
2. Visualizzare le prime righe, la struttura (.info()), e statistiche descrittive (.describe()).

Parte 2 - Pulizia
3. Gestire valori mancanti (es. sostituire con 0 o media).
4. Rimuovere duplicati.
5. Verificare che i tipi di dato siano corretti (date come datetime, quantità come numeri, ecc.).

Parte 3 - Analisi esplorativa
6. Calcolare vendite totali per prodotto.
7. Individuare il prodotto più venduto e quello meno venduto.
8. Calcolare vendite medie giornaliere.
"""

import pandas as pd 
import numpy as np 

#Creazione dataset fittizio con anomalie
np.random.seed(42)
date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

data = {
    "Data": np.random.choice(date_range, size=100),
    "Prodotto":np.random.choice(["Prodotto A", "Prodotto B", "Prodotto C"], size=100),
    "Vendite": np.random.choice([10, 20, 30, np.nan, 50, 100], size=100),
    "Prezzo": np.random.choice([15.0, 25.0, 50.0, np.nan], size=100)
}

df_raw = pd.DataFrame(data)

"""
Parte 1 - Caricamento e esplorazione dati
1. Leggere il dataset con Pandas.
2. Visualizzare le prime righe, la struttura (.info()), e statistiche descrittive (.describe()).
"""

# Prime 5 righe
print("--- PRIME RIGHE (head) ---")
print(df_raw.head())

# Informazioni sulla struttura del DataFrame
print("\n--- STRUTTURA DEL DATAFRAME (info) ---")
df_raw.info()

# Statistiche descrittive (include tutte le colonne, anche quelle non numeriche)
print("\n--- STATISTICHE DESCRITTIVE (describe) ---")
print(df_raw.describe(include="all"))

"""
Parte 2 - Pulizia
3. Gestire valori mancanti (es. sostituire con 0 o media).
4. Rimuovere duplicati.
5. Verificare che i tipi di dato siano corretti (date come datetime, quantità come numeri, ecc.).
"""

# copia pulita del dataframe originale per applicare pulizie e modifiche varie 

df_clean = df_raw.copy()

# ---------------------------------------------------------
# 3. GESTIONE DEI VALORI MANCANTI
# ---------------------------------------------------------

# Per Vendite sostituiamo i NaN con 0
df_clean["Vendite"] = df_clean["Vendite"].fillna(0)

# Per Prezzo sostituiamo i NaN con il prezzo medio dello specifico prodotto
df_clean["Prezzo"] = df_clean.groupby("Prodotto")["Prezzo"].transform(lambda x: x.fillna(x.mean()))

# ---------------------------------------------------------
# 4. RIMOZIONE DUPLICATI
# ---------------------------------------------------------

# Verifichiamo prima quanti duplicati ci sono
duplicati_iniziali = df_clean.duplicated().sum()

# Rimuoviamo i duplicati
df_clean = df_clean.drop_duplicates().reset_index(drop=True)

# ---------------------------------------------------------
# 5. VERIFICA E CORREZIONE DEI TIPI DI DATO (Dtypes)
# ---------------------------------------------------------

# Forziamo i tipi di dato corretti:
# - Data -> datetime64
# - Vendite -> numero intero (int64)
# - Prezzo -> numero decimale (float64)
# - Prodotto -> stringa / categoria

df_clean["Data"] = pd.to_datetime(df_clean["Data"])
df_clean["Vendite"] = df_clean["Vendite"].astype(int)
df_clean["Prezzo"] = df_clean["Prezzo"].round(2)  # Arrotondiamo a 2 decimali
df_clean["Prodotto"] = df_clean["Prodotto"].astype("category")

# Verifichiamo il risultato finale
print("\n--- INFO DEL DATAFRAME PULITO ---")
df_clean.info()

print("\n--- ANTEPRIMA DEL DATAFRAME PULITO ---")
print(df_clean.head())

"""
Parte 3 - Analisi esplorativa
6. Calcolare vendite totali per prodotto.
7. Individuare il prodotto più venduto e quello meno venduto.
8. Calcolare vendite medie giornaliere.
"""

# ---------------------------------------------------------
# 6. CALCOLARE VENDITE TOTALI PER PRODOTTO
# ---------------------------------------------------------
# Raggruppiamo per 'Prodotto' e sommiamo la colonna 'Vendite'

vendite_per_prodotto = (
    df_clean.groupby("Prodotto")["Vendite"].sum().reset_index()
)

print("--- VENDITE TOTALI PER PRODOTTO ---")
print(vendite_per_prodotto.to_string(index=False))

# ---------------------------------------------------------
# 7. INDIVIDUARE IL PRODOTTO PIÙ E MENO VENDUTO
# ---------------------------------------------------------
# Ordiniamo il risultato per trovare i due estremi

vendite_ordinate = vendite_per_prodotto.sort_values(
    by="Vendite", ascending=False
)
prodotto_piu_venduto = vendite_ordinate.iloc[0]
prodotto_meno_venduto = vendite_ordinate.iloc[-1]

print("\n--- PRODOTTO PIÙ E MENO VENDUTO ---")
print(
    f"Più venduto:  {prodotto_piu_venduto['Prodotto']} con {prodotto_piu_venduto['Vendite']} unità"
)
print(
    f"Meno venduto: {prodotto_meno_venduto['Prodotto']} con {prodotto_meno_venduto['Vendite']} unità"
)

# ---------------------------------------------------------
# 8. CALCOLARE VENDITE MEDIE GIORNALIERE
# ---------------------------------------------------------
# Metodo A: Media delle vendite totali per ciascun giorno

vendite_giornaliere_totali = df_clean.groupby("Data")["Vendite"].sum()
media_giornaliera = vendite_giornaliere_totali.mean()

# Metodo B: Media giornaliera suddivisa per singolo prodotto
media_giornaliera_per_prodotto = (
    df_clean.groupby(["Data","Prodotto"])["Vendite"].sum().groupby("Prodotto").mean()
)

print("\n--- VENDITE MEDIE GIORNALIERE ---")
print(f"Media vendite complessive al giorno: {media_giornaliera:.2f} unità")
print("\nMedia vendite giornaliere per ciascun prodotto:")
print(media_giornaliera_per_prodotto.round(2).to_string())