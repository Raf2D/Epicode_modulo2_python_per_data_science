""" 
Progetto #3 — Creare una Dashboard Avanzata di Visualizzazione

Traccia:

Realizzare una dashboard interattiva per analizzare i dati di vendita e redditività di
un negozio online, usando Python (Pandas + Plotly Dash / Streamlit / Matplotlib / Seaborn).

Struttura del dataset:

• Order Date → Data dell’ordine
• Ship Date → Data di spedizione
• Category → Categoria prodotto (es. Furniture, O'ice Supplies, Technology)
• Sub-Category → Sottocategoria prodotto
• Sales → Vendite (€)
• Profit → Utile (€)
• Region → Area geografica
• State → Stato/Regione
• Quantity → Quantità venduta

Consegna

Parte 1 – Pulizia dati

1. Convertire le colonne data (Order Date, Ship Date) in formato datetime.
2. Controllare valori nulli e duplicati.
3. Creare una nuova colonna Year dall’Order Date.

Parte 2 – Analisi Esplorativa (EDA)

4. Totale vendite e profitti per anno.
5. Top 5 sottocategorie più vendute.
6. Mappa interattiva delle vendite
"""

import streamlit as st 
import pandas as pd 
import numpy as np
import plotly.express as px

# Configurazione della pagina della dashboard 
st.set_page_config(
    page_title="Dashboard Vendite e Redditività",
    page_icon="📊",
    layout="wide"
)

# Funzione che provvede alla creazione di un dataset di esempio
@st.cache_data
def carica_dati():
    np.random.seed(42)
    n = 500  # numero di righe dataframe
    date = pd.date_range(start='2023-01-01', end='2025-12-31', periods=n)

    # Categorie e sottocategorie collegate
    mappa_categorie = {
        "Elettronica": ["Smartphone", "Laptop", "Accessori"],
        "Abbigliamento": ["Magliette", "Pantaloni", "Scarpe"],
        "Casa e Cucina": ["Utensili", "Arredo", "Elettrodomestici"],
        "Sport": ["Abbigliamento Tecnico", "Attrezzatura", "Scarpe Sportive"]
    }

    categorie = np.random.choice(list(mappa_categorie.keys()), n)
    sotto_categorie = [np.random.choice(mappa_categorie[cat]) for cat in categorie]
    stati = ["Italia"]
    regioni = [
        "Abruzzo", "Basilicata", "Calabria", "Campania", "Emilia-Romagna", "Friuli-Venezia Giulia",
        "Lazio", "Liguria", "Lombardia", "Marche", "Molise", "Piemonte", "Puglia", "Sardegna", "Sicilia",
        "Toscana", "Trentino-Alto Adige", "Umbria", "Valle d'Aosta", "Veneto"
    ]

    df = pd.DataFrame({
        "order_date": date,
        "ship_date": date,
        "category": categorie,
        "sub-category": sotto_categorie,
        "sales": np.random.uniform(100, 1000, n).round(2),
        "profit": np.random.uniform(10, 200, n).round(2),
        "region": np.random.choice(regioni, n),
        "state": np.random.choice(stati, n),
        "quantity": np.random.randint(1, 10, n)
    })

    return df

# Istanzio il dataframe 
negozio_df = carica_dati()

# ====================================================================
# Parte 1 – Pulizia dati
# ====================================================================

# 1. Pulizia strutturale iniziale
negozio_df.drop_duplicates(inplace=True)
negozio_df.sort_values(by="order_date", ascending=True, inplace=True)
negozio_df.reset_index(drop=True, inplace=True)

# 2. Gestione delle date
negozio_df["order_date"] = pd.to_datetime(negozio_df["order_date"]).bfill().ffill().dt.round('s')
negozio_df["ship_date"] = pd.to_datetime(negozio_df["ship_date"]).bfill().ffill().dt.round('s')

# 3. Gestione dei testi / categorie
negozio_df["category"] = negozio_df["category"].fillna('Sconosciuto')
negozio_df["sub-category"] = negozio_df["sub-category"].fillna('Sconosciuto')
negozio_df["region"] = negozio_df["region"].fillna('Sconosciuto')
negozio_df["state"] = negozio_df["state"].fillna('Sconosciuto')

# 4. Gestione dei valori numerici
negozio_df["sales"] = negozio_df["sales"].fillna(negozio_df["sales"].median())
negozio_df["profit"] = negozio_df["profit"].fillna(negozio_df["profit"].median())
negozio_df["quantity"] = negozio_df["quantity"].fillna(negozio_df["quantity"].mean())

# Creazione della colonna Year dall’Order Date
negozio_df["year"] = negozio_df["order_date"].dt.year

# ====================================================================
# Parte 2 – Analisi Esplorativa (EDA) & Visualizzazione Streamlit
# ====================================================================

st.title("📊 Dashboard Analisi Vendite e Redditività")

# 4. Totale vendite e profitti per anno
totale_annuo = negozio_df.groupby(["year"])[["sales", "profit"]].sum().reset_index()

st.subheader("📅 Totale Vendite e Profitti per Anno")
st.dataframe(totale_annuo, use_container_width=True)

# 5. Top 5 sottocategorie più vendute
totale_categorie = negozio_df.groupby(["sub-category"])["sales"].sum().reset_index()
totale_categorie.sort_values(by="sales", ascending=False, inplace=True)

st.subheader("🔥 Top 5 Sottocategorie più Vendute")
st.dataframe(totale_categorie.head(5), use_container_width=True)

# 6. Mappa interattiva delle vendite
st.subheader("🗺️ Performance di Vendita per Regione Italiana")

# Dizionario con le coordinate dei capoluoghi di regione italiani
coordinate_regioni = {
    "Lombardia": {"lat": 45.4642, "lon": 9.1900},
    "Lazio": {"lat": 41.9028, "lon": 12.4964},
    "Veneto": {"lat": 45.4408, "lon": 12.3155},
    "Campania": {"lat": 40.8518, "lon": 14.2681},
    "Sicilia": {"lat": 37.5999, "lon": 14.0154},
    "Emilia-Romagna": {"lat": 44.4949, "lon": 11.3426},
    "Piemonte": {"lat": 45.0703, "lon": 7.6869},
    "Toscana": {"lat": 43.7696, "lon": 11.2558},
    "Puglia": {"lat": 41.1171, "lon": 16.8719},
    "Calabria": {"lat": 38.9059, "lon": 16.5944},
    "Sardegna": {"lat": 39.2238, "lon": 9.1217},
    "Liguria": {"lat": 44.4056, "lon": 8.9463},
    "Marche": {"lat": 43.6158, "lon": 13.5189},
    "Abruzzo": {"lat": 42.3539, "lon": 13.3975},
    "Friuli-Venezia Giulia": {"lat": 45.6495, "lon": 13.7768},
    "Trentino-Alto Adige": {"lat": 46.0679, "lon": 11.1211},
    "Umbria": {"lat": 43.1107, "lon": 12.3908},
    "Basilicata": {"lat": 40.6395, "lon": 15.8056},
    "Molise": {"lat": 41.5604, "lon": 14.6622},
    "Valle d'Aosta": {"lat": 45.7373, "lon": 7.3201}
}

# Aggregazione dati per regione
df_mappa = negozio_df.groupby("region")[["sales", "profit"]].sum().reset_index()

# Mappatura sicura delle coordinate
df_mappa["lat"] = df_mappa["region"].map(lambda x: coordinate_regioni.get(str(x), {}).get("lat") if pd.notna(x) else None)
df_mappa["lon"] = df_mappa["region"].map(lambda x: coordinate_regioni.get(str(x), {}).get("lon") if pd.notna(x) else None)
df_mappa.dropna(subset=["lat", "lon"], inplace=True)

# Creazione della mappa interattiva con Plotly
fig = px.scatter_map(
    df_mappa,
    lat="lat",
    lon="lon",
    size="sales",               
    color="profit",             
    hover_name="region",        
    hover_data={"sales": True, "profit": True, "lat": False, "lon": False},
    color_continuous_scale="Viridis", 
    zoom=5,                     
    center={"lat": 42.5, "lon": 12.5},
    height=600,
    size_max=25
)

fig.update_layout(map_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

# Visualizzazione della mappa su Streamlit
st.plotly_chart(fig, width="stretch")