"""
Progetto #2 - Analisi vendite realistica

Traccia:

Sei un data analyst in un’azienda di e-commerce. Ti vengono forniti:

o Un CSV con ordini clienti (ordini.csv) .
o Un JSON con informazioni prodotto (prodotti.json).
o Un CSV con dati clienti (clienti.csv).
Questo progetto simula una situazione reale in cui occorre integrare fonti dati multiple,
ottimizzare memoria, applicare filtri complessi, aggregazioni avanzate e serializzazione
e'iciente, come capita quotidianamente in un contesto lavorativo di Data Analyst o
Business Intelligence.

#########################################################################################

Consegna

Parte 1 – Crea i seguenti DataSet
1. Ordini.csv: 100.000 righe con ClienteID, ProdottoID, Quantità e DataOrdine.
2. prodotti.json: 20 prodotti con Categoria e Fornitore.
3. clienti.csv: 5.000 clienti con Regione e Segmento.

Parte 2 – Creare un DataFrame unificato
4. Unisci ordini.
5. Unisci prodotti.
6. Unisci clienti

Parte 3 – Ottimizzazione
7. Ottimizzare i tipi di dato.
8. Ottimizzare l’uso della memoria.

Parte 4 – Creare colonne e filtra i dati
9. Crea una colonna calcolata (ValoreTotale = Prezzo * Quantità).
10. Filtrare ordini con ValoreTotale > 100 e clienti
"""

##########################################################
# PARTE 1 
##########################################################

import pandas as pd
import numpy as np
from pathlib import Path
import time 

# Risolve il percorso corrente rendendolo un percorso assoluto sicuro
# Individua la cartella esatta in cui si trova questo file .py
percorso_cartella = Path(__file__).resolve().parent
np.random.seed(123) # seme per la riproducibilità



# prodotti.json: 20 prodotti con Categoria e Fornitore.

categorie = ["Elettronica", "Abbigliamento", "Casa e Cucina", "Sport", "Libri"]
fornitori = [
    "TechSupply Srl",
    "Global Goods Spa",
    "FastTrade Ltd",
    "StyleHub",
    "Omnia Logistics",
]

n_prodotti = 20

# creaimo il dataframe prodotti per poi salvarlo in json
prodotti = pd.DataFrame({
    "ProdottoID": np.random.choice(np.arange(1000, 10_000, 100),size=n_prodotti,replace=False), #id prodotti unici
    "Categoria": np.random.choice(categorie,n_prodotti),
    "Fornitore": np.random.choice(fornitori,n_prodotti),
    "Prezzo": np.round(np.random.uniform(1,1000, size=20),2)
})

# salvataggio prodotti.json in formato records
json_prodotti = prodotti.to_json(percorso_cartella/"prodotti.jsonl",orient="records",lines=True)


# clienti.csv: 5.000 clienti con Regione e Segmento.

n_clienti = 5_000

regioni_italiane = [
    "Abruzzo", "Basilicata", "Calabria", "Campania", "Emilia-Romagna",
    "Friuli-Venezia Giulia", "Lazio", "Liguria", "Lombardia", "Marche",
    "Molise", "Piemonte", "Puglia", "Sardegna", "Sicilia",
    "Toscana", "Trentino-Alto Adige", "Umbria", "Valle d'Aosta", "Veneto"
]

clienti = pd.DataFrame({
    "ClienteID":np.arange(1,n_clienti+1),
    "Nome":np.char.add("Cliente_",np.arange(1,n_clienti+1).astype(str)), #vettorializzazione
    "Segmento":np.random.choice(["Premium","Standard"],n_clienti),
    "Regione":np.random.choice(regioni_italiane,n_clienti)
})

clienti.to_csv(percorso_cartella/"clienti.csv",index=False)

# Ordini.csv: 100.000 righe con ClienteID, ProdottoID, Quantità e DataOrdine.

n_ordini = 100_000
date_ordini = pd.date_range(start="2025-01-01", end="2026-01-01")

# Estraiamo gli array 1D nativi per evitare che NumPy calcoli combinazioni giganti
array_clienti_id = clienti["ClienteID"].to_numpy()
array_prodotti_id = prodotti["ProdottoID"].to_numpy()

ordini = pd.DataFrame({
    "OrdineID": np.arange(1,n_ordini+1),
    "ClienteID":np.random.choice(array_clienti_id,n_ordini),
    "ProdottoID": np.random.choice(array_prodotti_id,n_ordini),
    "Quantita": np.random.randint(1,10,n_ordini),
    "DataOrdine": np.random.choice(date_ordini, n_ordini)
})

# salvataggio ordini.csv
ordini.to_csv(percorso_cartella/"ordini.csv",index=False)


##########################################################
# PARTE 2 
##########################################################

# lettura file 

clienti = pd.read_csv(percorso_cartella/"clienti.csv")
prodotti = pd.read_json(percorso_cartella/"prodotti.jsonl", orient="records",lines=True)
ordini = pd.read_csv(percorso_cartella/"ordini.csv",parse_dates=["DataOrdine"])

print("Esempio record clienti.csv\n")
print(clienti.head())
print("\nEsempio record prodotti.jsonl\n")
print(prodotti.head())
print("\nEsempio record ordini.csv\n")
print(ordini.head())

# Merge ordini + prodotti + clienti

resoconto = ordini.merge(prodotti, on="ProdottoID", how="left").merge(clienti, on="ClienteID",how="left")
print(f"\nLunghezza resoconto: {len(resoconto)}\n")

print("\nEsempio RESOCONTO: dataframe uniti")
print(resoconto.head())
print(resoconto.info())



##########################################################
# PARTE 3
# Ottimizzare i tipi di dato.
# Ottimizzare l’uso della memoria.
##########################################################

def misura_memoria(df, nome="DataFrame"):
    mem_mb = df.memory_usage(deep=True).sum()/1024**2
    print(f"Memoria occupata da {nome}: {mem_mb:.2f} MB")
    return mem_mb

mem_prima = misura_memoria(resoconto, "resoconto (Originale)")
print("Memoria occupata prima dell'ottimizzazione:",mem_prima)

resoconto["Regione"] = resoconto["Regione"].astype("category")
resoconto["Segmento"] = resoconto["Segmento"].astype("category")
resoconto["Categoria"] = resoconto["Categoria"].astype("category")
resoconto["Fornitore"] = resoconto["Fornitore"].astype("category")
resoconto["DataOrdine"] = resoconto["DataOrdine"].astype("datetime64[s]")

resoconto["Prezzo"] = resoconto["Prezzo"].astype("float32")
resoconto["Quantita"] = resoconto["Quantita"].astype("int16")
resoconto["OrdineID"] = resoconto["OrdineID"].astype("int32")
resoconto["ClienteID"] = resoconto["ClienteID"].astype("int32")
resoconto["ProdottoID"] = resoconto["ProdottoID"].astype("int16")

mem_dopo = misura_memoria(resoconto, "resoconto (Ottimizzato)")
print("Memoria occupata dopo dell'ottimizzazione:",mem_dopo)
print(f"Risparmio memoria : {((1 - mem_dopo/mem_prima) * 100):.2f}%")

##########################################################
# Parte 4 – Creare colonne e filtra i dati
# Crea una colonna calcolata (ValoreTotale = Prezzo * Quantità).
# Filtrare ordini con ValoreTotale > 100 e clienti
##########################################################
 
# creazione colonna ValoreTotale
resoconto["ValoreTotale"] = resoconto["Prezzo"] * resoconto["Quantita"]

subset_resoconto = resoconto.query("ValoreTotale > 100")[["OrdineID","ClienteID","ValoreTotale"]].copy()
print("\nReconto con solo il ValoreTotale > 100:\n")
print(len(subset_resoconto))
print(subset_resoconto.head())