"""
Esercizio Completo: Generazione, Ottimizzazione, Chunking, Formato Parquet e Predicate Pushdown
Questo script dimostra le best practices per gestire grandi volumi di dati in Python con Pandas e PyArrow,
evitando colli di bottiglia legati alla memoria RAM e ottimizzando le prestazioni di I/O.
"""

import pandas as pd 
import numpy as np
from pathlib import Path
import time
import pyarrow as pa
import pyarrow.parquet as pq
import os

# Determinazione dinamica del percorso della cartella di lavoro (compatibile con script o notebook)
try:
    percorso_cartella = Path(__file__).resolve().parent
except:
    percorso_cartella = Path.cwd()

# Inizializzazione del generatore di numeri casuali moderno di NumPy per garantire riproducibilità
rng = np.random.default_rng(111)
n_righe = 2_000_000  # Definizione del volume dei dati (2 milioni di righe)

print("Generazione dei dati in corso...")

# ---------------------------------------------------------
# FASE 1: Generazione del DataFrame di grandi dimensioni
# ---------------------------------------------------------
ids = np.arange(1, n_righe + 1)
prodotti = np.array(["Laptop", "Mouse", "Tastiera", "Monitor", "Stampante", "Cuffie"])
scelta_prodotti = rng.choice(prodotti, size=n_righe)
prezzi = np.round(rng.uniform(10.0, 1500.0, size=n_righe), 2)
quantita = rng.integers(1, 11, size=n_righe)
regioni = np.array(["Nord", "Centro", "Sud", "Isole"])
scelta_regioni = rng.choice(regioni, size=n_righe)
categorie = np.array(["Elettronica", "Informatica", "Accessori", "Office"])
scelta_categorie = np.random.choice(categorie, size=n_righe)

# Creazione del DataFrame iniziale con i dati grezzi
df = pd.DataFrame({
    "ID": ids,
    "Prodotto": scelta_prodotti,
    "Prezzo": prezzi,      
    "Quantita": quantita,    
    "Regione": scelta_regioni,
    "Categoria": scelta_categorie
})

print("DataFrame creato con successo!")
print(f"Dimensioni: {df.shape}")
print(df.head())

# ---------------------------------------------------------
# FASE 2: Ottimizzazione dei tipi di dato (Memory Footprint Reduction)
# ---------------------------------------------------------
# Misuriamo la memoria iniziale con deep=True per includere l'effettivo peso degli oggetti stringa
memoria_prima = np.round((df.memory_usage(deep=True) / 1024**2).sum(), 2)
print(f"\nUso memoria in MB prima dell'ottimizzazione dati: {memoria_prima} MB")

# Downcasting dei tipi numerici e conversione delle stringhe ripetute in 'category'
df["Prezzo"] = df["Prezzo"].astype("float32")         # Da float64 a float32 (dimezza lo spazio)
df["Quantita"] = df["Quantita"].astype("int16")       # Da int64/int32 a int16 (ottimale per numeri piccoli)
df["Regione"] = df["Regione"].astype("category")     # Convertito in indici interi interni (enorme risparmio)
df["Categoria"] = df["Categoria"].astype("category") # Convertito in categoria

# Misuriamo la memoria dopo l'ottimizzazione
memoria_dopo = np.round((df.memory_usage(deep=True) / 1024**2).sum(), 2)
print(f"Uso memoria in MB dopo l'ottimizzazione dati: {memoria_dopo} MB")

# ---------------------------------------------------------
# FASE 3: Calcoli vettorizzati e filtraggio con .copy()
# ---------------------------------------------------------
# Creazione della colonna Valore_totale tramite moltiplicazione vettorizzata
df["Valore_totale"] = df["Quantita"] * df["Prezzo"]

# Filtraggio delle righe con .copy() per evitare il SettingWithCopyWarning di Pandas
df_filtrato = df[df["Valore_totale"] > 500].copy() 
print(f"\nRighe con Valore totale > 500: {len(df_filtrato)}")

# Calcolo di metriche descrittive sul subset filtrato
media = df_filtrato["Valore_totale"].mean()
somma = df_filtrato["Valore_totale"].sum()
massimo = df_filtrato["Valore_totale"].max()

print(f"Media: {media:.2f} | Somma: {somma:.2f} | Massimo: {massimo:.2f}\n")

# ---------------------------------------------------------
# FASE 4: Salvataggio CSV e Lettura a Blocchi (Chunking)
# ---------------------------------------------------------
# Salvataggio del DataFrame in formato CSV su disco
df.to_csv(percorso_cartella / 'store.csv', index=False)

# Approccio a Chunking: Leggiamo il CSV in blocchi da 200.000 righe per volta per non sovraccaricare la RAM
start_time = time.time()
chunk_iter = pd.read_csv(percorso_cartella / 'store.csv', chunksize=200_000)

list_chunk = []
sconto = 1 / 10  # 10% di sconto

for chunk in chunk_iter:
    # Elaborazione locale sul singolo chunk
    chunk["Prezzo_scontato"] = (chunk["Prezzo"] * (1 - sconto)).round(2)
    chunk["Valore_totale_scontato"] = (chunk["Prezzo_scontato"] * chunk["Quantita"]).round(2)
    list_chunk.append(chunk)

# Ricomposizione finale dei chunk elaborati
df_finale = pd.concat(list_chunk, ignore_index=True)
end_time = time.time()
tempo_totale_chunking = end_time - start_time

print(f"TEMPO ELABORAZIONE CHUNKING (CSV): {tempo_totale_chunking:.2f} secondi.")

# Confronto con caricamento FULL in memoria in un'unica soluzione
start_time = time.time()
df_csv = pd.read_csv(percorso_cartella / 'store.csv')
df_csv["Prezzo_scontato"] = (df_csv["Prezzo"] * (1 - sconto)).round(2)
df_csv["Valore_totale_scontato"] = (df_csv["Prezzo_scontato"] * df_csv["Quantita"]).round(2)
end_time = time.time()
tempo_totale_full = end_time - start_time

print(f"TEMPO ELABORAZIONE FULL (CSV): {tempo_totale_full:.2f} secondi.")

# ---------------------------------------------------------
# FASE 5: Debug Step-by-Step sui Chunk
# ---------------------------------------------------------
for i, chunk in enumerate(list_chunk):
    print(f"\n--- CHUNK #{i+1} ---")
    print("\n--- INFO e MEMORIA ---")
    chunk.info()  # Visualizza struttura e consumo di memoria per colonna
    print("\n--- STATISTICHE DESCRITTIVE ---")
    print(chunk.describe())  # Statistiche numeriche di base
    print("=" * 50)

# ---------------------------------------------------------
# FASE 6: Salvataggio in Parquet Chunk-by-Chunk tramite ParquetWriter
# ---------------------------------------------------------
file_output = percorso_cartella / 'dataset_completo.parquet'
writer = None 

start_time = time.time()

for chunk in list_chunk:
    # Conversione del DataFrame Pandas in una Tabella PyArrow
    table = pa.Table.from_pandas(chunk, preserve_index=False)

    if writer is None:
        # Al primo giro inizializziamo il writer definendo lo schema basato sul primo blocco
        writer = pq.ParquetWriter(str(file_output), table.schema)

    # Scrittura sequenziale del blocco come 'row group' nel file Parquet unico
    writer.write_table(table)

# Chiusura del writer per finalizzare il file e scrivere i metadati
if writer:
    writer.close()

end_time = time.time()

print(f"\nTutti i chunk sono stati salvati in un unico file Parquet: \n{file_output}\n")

file_bytes = os.path.getsize(file_output)
print(f"Dimensione del file Parquet completo: {file_bytes / (1024**2):.2f} MB")
print(f"Tempo di scrittura finale: {end_time - start_time:.2f} secondi.")

# ---------------------------------------------------------
# FASE 7: Confronto di Prestazioni tra Lettura Totale e Predicate Pushdown
# ---------------------------------------------------------

# Approccio 1: Lettura completa e filtro applicato in post-elaborazione
start_normale = time.time()
df_totale = pd.read_parquet(file_output)
df_filtrato_normale = df_totale[
    (df_totale["Categoria"] == 'A') & 
    (df_filtrato_normale_cond := df_totale["Regione"] == 'Nord') # sintesi logica
]
end_normale = time.time()
tempo_normale = end_normale - start_normale

# Approccio 2: Lettura ottimizzata tramite Predicate Pushdown (filtri nativi a livello di lettura)
start_pushdown = time.time()
df_filtrato_pushdown = pd.read_parquet(
    file_output,
    filters=[
        ('Categoria', '=', 'A'),
        ('Regione', '=', 'Nord')
    ]
)
end_pushdown = time.time()
tempo_pushdown = end_pushdown - start_pushdown

print("\n--- CONFRONTO TEMPI DI ESECUZIONE (PARQUET) ---")
print(f"1. Lettura normale + filtro Pandas:  {tempo_normale:.4f} secondi")
print(f"2. Lettura con Predicate Pushdown:    {tempo_pushdown:.4f} secondi")

if tempo_pushdown < tempo_normale:
    fattore = tempo_normale / tempo_pushdown
    print(f"\n🚀 Il Predicate Pushdown è stato circa **{fattore:.1f} volte più veloce**!")