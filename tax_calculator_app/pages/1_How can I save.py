import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# -----------------------------------------------------------
# 1) BFS API (vereinfachtes Beispiel)
# -----------------------------------------------------------
@st.cache_data
def load_bfs_data():
    url = "https://www.pxweb.bfs.admin.ch/api/v1/de/px-x-2103010000_103"
    query = {
        "query": [
            {"code": "Kanton", "selection": {"filter": "item", "values": ["17"]}},
            {"code": "Geschlecht", "selection": {"filter": "item", "values": ["1"]}},
            {"code": "Alter", "selection": {"filter": "all", "values": ["*"]}}
        ],
        "response": {"format": "JSON"}
    }

    try:
        r = requests.post(url, json=query)
        raw = r.json()
        df = pd.DataFrame(raw["data"])
        df["Wert"] = df["values"].astype(float)
        return df
    except:
        return None


# -----------------------------------------------------------
# 2) synthetische Daten erweitern: Abzug-HÖHEN
# -----------------------------------------------------------
def create_synthetic_data(n=400):
    np.random.seed(42)

    df = pd.DataFrame({
        "income": np.random.normal(72000, 18000, n).astype(int),
        "age": np.random.normal(43, 12, n).astype(int),
        "children": np.random.randint(0, 4, n),
        "commute_km": np.abs(np.random.normal(14, 7, n)),
        "insurance_costs": np.abs(np.random.normal(3500, 800, n)),
        "education_costs": np.abs(np.random.normal(1200, 900, n))
    })

    # Kategorien (binär)
    df["uses_3a"] = (df["income"] > 60000).astype(int)
    df["uses_commuter"] = (df["commute_km"] > 8).astype(int)
    df["uses_insurance"] = (df["insurance_costs"] > 3200).astype(int)
    df["uses_education"] = (df["education_costs"] > 1500).astype(int)
    df["uses_children"] = (df["children"] >= 1).astype(int)

    # Abzugshöhen (Regression Labels)
    df["amt_3a"] = np.clip(df["income"] * 0.07, 0, 7056)
    df["amt_commuter"] = df["commute_km"] * 220
    df["amt_insurance"] = df["insurance_costs"] * 0.8
    df["amt_education"] = df["education_costs"] * 0.6
    df["amt_children"] = df["children"] * 3000

    return df


# -----------------------------------------------------------
# 3) K-Means
# -----------------------------------------------------------
def train_kmeans(df):
    X = df[["income", "commute_km", "children"]]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = KMeans(n_clusters=3, random_state=42)
    df["cluster"] = model.fit_predict(Xs)

    return model, scaler


# -----------------------------------------------------------
# 4) Klassifizierer (Logistic Regression)
# -----------------------------------------------------------
def train_classifier(df, target_col):
    X = df[["income", "commute_km", "children", "age",
            "insurance_costs", "education_costs"]]
    y = df[target_col]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    xtr, xte, ytr, yte = train_test_split(Xs, y, test_size=0.25, random_state=42)

    model = LogisticRegression()
    model.fit(xtr, ytr)
    accuracy = model.score(xte, yte)

    return model, scaler, accuracy


# -----------------------------------------------------------
# 5) Regressionsmodell für Abzug-Höhe
# -----------------------------------------------------------
def train_regressor(df, target_col):
    X = df[["income", "commute_km", "children", "age",
            "insurance_costs", "education_costs"]]
    y = df[target_col]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(Xs, y)

    return model, scaler


# -----------------------------------------------------------
# 6) Streamlit UI
# -----------------------------------------------------------
st.title("Steuer-Abzugs-Assistent St. Gallen – ML-Modelle")
st.caption("Mit BFS-Daten, K-Means, Klassifikation & Regressionsprognosen")

# Benutzereingaben
st.header("🔧 Persönliche Angaben")

income = st.number_input("Einkommen (CHF)", 20000, 200000, 70000)
age = st.slider("Alter", 18, 75, 40)
children = st.number_input("Kinderanzahl", 0, 5, 1)
commute = st.number_input("Pendeldistanz (km)", 0.0, 100.0, 12.0)
insurance_costs = st.number_input("Versicherungskosten (CHF / Jahr)", 0, 15000, 3600)
education_costs = st.number_input("Kosten für Weiterbildung (CHF / Jahr)", 0, 15000, 1500)

# Daten laden
df = create_synthetic_data()

# --------------------------------------------------------------------------------
# CLUSTERING
# --------------------------------------------------------------------------------
st.header("🎯 Cluster-Zugehörigkeit (K-Means)")

kmeans, k_scaler = train_kmeans(df)
cluster_id = kmeans.predict(
    k_scaler.transform([[income, commute, children]])
)[0]

st.write(f"Du gehörst zu **Cluster {cluster_id}**.")

fig = px.scatter(
    df,
    x="income",
    y="commute_km",
    color="cluster",
    title="K-Means Cluster der Nutzer",
    labels={"income": "Einkommen", "commute_km": "Pendeldistanz"},
)
st.plotly_chart(fig)

# --------------------------------------------------------------------------------
# Klassifikation (Logistic Regression)
# --------------------------------------------------------------------------------
st.header("🔮 Abzug – Wahrscheinlichkeit (Logistic Regression)")

targets = {
    "uses_3a": ("Säule 3a", "amt_3a"),
    "uses_commuter": ("Pendlerabzug", "amt_commuter"),
    "uses_insurance": ("Versicherungsabzug", "amt_insurance"),
    "uses_education": ("Weiterbildungskosten", "amt_education"),
    "uses_children": ("Familienabzug", "amt_children")
}

user_vec = np.array([[income, commute, children, age, insurance_costs, education_costs]])

probs = {}
amounts = {}

for col, (label, amt_col) in targets.items():

    # Klassifikation
    model, scaler, acc = train_classifier(df, col)
    prob = model.predict_proba(scaler.transform(user_vec))[0][1]
    probs[label] = prob

    st.subheader(f"📌 {label}")
    st.write(f"Wahrscheinlichkeit: **{prob*100:.1f}%**")
    st.caption(f"Modell-Accuracy: {acc*100:.1f}%")

    # Regression – Abzugshöhe
    reg, scaler_r = train_regressor(df, amt_col)
    pred_amt = reg.predict(scaler_r.transform(user_vec))[0]
    amounts[label] = max(0, pred_amt)

    st.write(f"→ Voraussichtlicher Abzug: **{pred_amt:.2f} CHF**")


# --------------------------------------------------------------------------------
# VISUALISIERUNG DER ABZUGSHÖHEN
# --------------------------------------------------------------------------------
st.header("📊 Geschätzte Abzugshöhen (Regression)")

amt_df = pd.DataFrame({
    "Abzug": list(amounts.keys()),
    "CHF": list(amounts.values())
})

fig2 = px.bar(
    amt_df,
    x="Abzug",
    y="CHF",
    title="Geschätzte Abzugshöhen",
    labels={"CHF": "CHF"},
    text="CHF"
)
st.plotly_chart(fig2)
