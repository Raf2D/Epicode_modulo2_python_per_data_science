"""
Esercizio 1
Crea una colonna di stringhe con nomi sporchi (spazi, maiuscole casuali). Puliscila rendendo tutti i nomi minuscoli e senza spazi extra.
Esercizio 2 
Genera un DataFrame con date di nascita in formato stringa. Convertile in datetime e calcola l’età in anni di ogni individuo.
Esercizio 3 
Crea un dataset giornaliero di valori numerici su due mesi. Usa .resample() per ottenere: media settimanale, massimo mensile e somma cumulativa.

"""

import numpy as np 
import pandas as pd 




# Esercizio 1

print("\n ---- Esecuzione Esercizio 1 ---- \n")

data = (
    {
        "Clienti": [
            " mArIo ROSSI",
            "luigi_verdi",
            "ANNA BIANCHI",
            "giovanni d'italia",
            " sofia neri ",
            "Luca Russo!"
        ]
    }
)

df = pd.DataFrame(data)
df['Clienti_puliti'] = df["Clienti"].str.replace(r"[^a-zA-ZàèéìòùÀÈÉÌÒÙ'\s]"," ",regex=True) # sostituisce caratteri speciali e trattini bassi con uno spazio
df["Clienti_puliti"] = df["Clienti_puliti"].str.replace(r"\s+"," ", regex=True, ).str.strip().str.lower() # rimuove spazi multipli interni e spazi di testa/cosa e tutto minuscolo

print(df)


# Esercizio 2

print("\n ---- Esecuzione Esercizio 2 ---- \n")

df_date_di_nascita = pd.DataFrame({
    'utenti':["Mario Rossi","Luigi Verdi","Anna Bianchi","Giovanni D'Italia","Sofia Neri","Luca Russo","Elena Romano","Marco Colombo","Giulia Ricci","Alessandro Marino","Chiara Bruno","Matteo Ferrari",],
    'date_di_nascita': pd.date_range('1990-01-01', periods =12, freq ='6MS') # genera date ogni 6 mesi calcolate al primo giorno del mese
})

df_date_di_nascita['date_di_nascita'] = pd.to_datetime(df_date_di_nascita['date_di_nascita'])
# df_date_di_nascita.set_index('date_di_nascita', inplace = True)

from datetime import datetime 
sysdate = pd.Timestamp.now() # prendo la data attuale 

df_date_di_nascita['Età'] = ((sysdate - df_date_di_nascita["date_di_nascita"]).dt.days / 365.25).astype(int)

print(df_date_di_nascita)


print("\n ---- Esecuzione Esercizio 3 ---- \n")

np.random.seed(42)

# creiamo il range di date giornaliere per 2 mesi completi

date_index = pd.date_range(start = '2026-01-01', end = '2026-02-28', freq = 'D')

vendite = np.random.randint(100, 500, size = len(date_index))
incasso = np.round(vendite * np.random.uniform(15.0, 25.0, size = len(date_index)), 2)

df_giornaliero = pd.DataFrame(
    {'vendite_unita': vendite, 'incasso_giornaliero': incasso}, index = date_index #imposta l'indice 
)

df_giornaliero.index.name = 'Data'

# Aggregazione settimanale con media e totale 

df_settimanale = df_giornaliero.resample('W-MON').agg(
    {
        'incasso_giornaliero':['mean', 'sum'],
        'vendite_unita':'sum'
    }
)

# Pulizia dei nomi delle colonne per massima chiarezza 

df_settimanale.columns = [
    'incasso_medio_giornaliero',
    'incasso_totale_settimanale',
    'vendite_totali_settimanali'
]

# Aggregazione Mensile con Media, Totale e MASSIMO

df_mensile = df_giornaliero.resample("ME").agg(
    {
        "incasso_giornaliero": ["mean", "sum", "max"],
        "vendite_unita": "sum",
    }
)

# Puliamo i nomi delle colonne
df_mensile.columns = [
    "incasso_medio_giornaliero",
    "incasso_totale_mese",
    "incasso_massimo_giornaliero",
    "vendite_totali_mese",
]

print(df_giornaliero)
print(df_settimanale)
print(df_mensile)

