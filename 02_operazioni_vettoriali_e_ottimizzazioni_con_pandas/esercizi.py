"""
Esercizio 1
Crea un dataframe con una colonna temperature in celsius e aggiungi una nuova colonna con la conversione in fahrenheit.
"""

import numpy as np
import pandas as pd


print("\n--------- Esercizio 1 ----------------------------------------------\n")

data = {

    "Città": [
        "Roma",
        "Catanzaro",
        "Firenze",
        "Cagliari"
    ],
    "Temperature_celsius":[
        30,31,34,33
    ]
}

df = pd.DataFrame(data)
df["Temperature_fahrenheit"] = (df["Temperature_celsius"]*1.8) + 32

print(df)



"""
Esercizio 2 
Hai una colonna con punteggi numerici. 
Crea una nuova colonna che indichi “promosso” se il punteggio è >= di 18 e “bocciato” altrimenti. Usa np.where().
"""

print("\n------ Esercizio 2 -------------------------------------------\n")

data2 = {

    "Voti": [
        20,22,18,17,15,19,24,19
    ]

}

df2 = pd.DataFrame(data2)
df2["Esito"] = np.where(df2["Voti"] >= 18, "Promosso","Bocciato")

print(df2)


"""
Esercizio 3 
Hai 2 colonne “ore lavorative” e “paga oraria”. Crea una nuova colonna che calcoli il salario settimanale,
applicando un bonus del 10% se le ore sono superiori a 40.
"""

print("\n--------- Esercizio 3 -----------------------------------------------------")

data3 = {

    "Ore_lavorative":[
        29, 34, 40, 44, 56, 78, 18, 21, 10
    ],
    "Paga_oraria":[ 
        10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5
    ]

}

df3 = pd.DataFrame(data3)

paga_base = (df3["Ore_lavorative"]*df3["Paga_oraria"])
paga_straordinario = (40 * df3["Paga_oraria"]) + ((df3["Ore_lavorative"] - 40) * df3["Paga_oraria"] * 1.10)

df3["Salario_settimanale"] = np.where(df3["Ore_lavorative"]>40, paga_straordinario, paga_base)

print(df3)

