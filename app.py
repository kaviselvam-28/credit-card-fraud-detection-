import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

st.title("💳 Credit Card Fraud Detection & Clustering")

uploaded_file = st.file_uploader("Upload amt.csv file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data", df.head())
    df = df.dropna()
    if 'trans_date' in df.columns:
        df['trans_date'] = pd.to_datetime(df['trans_date'], errors='coerce')

    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.select_dtypes(include='number')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    iso = IsolationForest(contamination=0.01, random_state=42)
    df['is_fraud_pred'] = iso.fit_predict(X_scaled)
    df['is_fraud_pred'] = df['is_fraud_pred'].map({1: 0, -1: 1})

    st.write("### Clustered Data", df[['amt', 'cluster', 'is_fraud_pred']].head())

    st.write("### Cluster Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='amt', y='cluster', hue='cluster', ax=ax)
    st.pyplot(fig)

    st.write("### 🚨 Fraudulent Transactions")
    st.dataframe(df[df['is_fraud_pred'] == 1])
with open("streamlit_app.py", "w") as f:
    f.write(code)
