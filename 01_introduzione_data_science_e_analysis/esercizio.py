"""
Prova a modificare il codice:
- quale è la settimana con le vendite massime? 
- aggiungi un grafico a linee sopra quello a barre per visualizzare meglio il trend
- cambia i colori delle barre in base a una soglia: rosse se sotto la media, verde se sopra
"""

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

dati = {"Settimana": [1,2,3,4,5,6],
        "Vendite":[250, 300, 400, 350, 450, 500]
        }

df = pd.DataFrame(dati)

media_vendite = df["Vendite"].mean()
print("Media vendite: ", media_vendite)

# AGGIUNTA: Se le vendite sono maggiori della media assegna 'green', altrimenti 'red'
colori_barre = np.where(df["Vendite"] > media_vendite, 'green', 'red')

plt.figure(figsize=(8, 5))

# Disegniamo le barre passando la lista di colori appena creata
# alpha=0.6 rende i colori più pastello e piacevoli alla vista
plt.bar(df["Settimana"], df["Vendite"], color=colori_barre, alpha=0.6, label="Vendite")

# Grafico a linee sopra le barre per mostrare il trend
plt.plot(df["Settimana"], df["Vendite"], color="darkblue", marker="o", linewidth=2, label="Trend")

# Linea della media
plt.axhline(media_vendite, color='black', linestyle='--', linewidth=1.5, label=f"Media ({media_vendite})")


plt.xlabel("Settimana")
plt.ylabel("Vendite")
plt.title("Vendite settimanali (Verde > Media, Rosso < Media)")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# idxmax() trova l'indice della riga con il valore massimo nella colonna "Vendite"
riga_massima = df["Vendite"].idxmax()
settimana_max = df.loc[riga_massima, "Settimana"]
valore_max = df.loc[riga_massima, "Vendite"]

print(f"La settimana con le vendite massime è la {settimana_max} con ben {valore_max} vendite!")