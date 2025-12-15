import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("📊 Analisis Clustering & Regresi Siswa")

# =====================
# LOAD DATASET
# =====================
df = pd.read_csv("dataset tubes dt.csv")

st.subheader("Dataset Siswa")
st.write("Ukuran dataset:", df.shape)
st.dataframe(df.head())

# =====================
# K-MEANS CLUSTERING
# =====================
st.subheader("Proses Clustering (K-Means)")

X_cluster = df[["jam_belajar", "kehadiran", "nilai_tugas"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# =====================
# TABEL RINGKASAN CLUSTER
# =====================
cluster_summary = df["cluster"].value_counts().reset_index()
cluster_summary.columns = ["Cluster", "Jumlah Data"]
cluster_summary = cluster_summary.sort_values("Cluster")

st.write("Ringkasan Cluster")
st.dataframe(cluster_summary)

# =====================
# VISUALISASI CLUSTER
# =====================
st.subheader("Visualisasi Clustering")

fig1 = plt.figure()
plt.scatter(df["jam_belajar"], df["nilai_tugas"], c=df["cluster"])
plt.xlabel("Jam Belajar")
plt.ylabel("Nilai Tugas")
plt.title("Hasil Clustering Siswa")
st.pyplot(fig1)

# =====================
# REGRESI LINEAR
# =====================
st.subheader("Regresi Linear")

X = df[["jam_belajar", "kehadiran", "nilai_tugas"]]
y = df["nilai_akhir"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.write("### Hasil Regresi Linear")
st.write("MSE :", mse)
st.write("R²  :", r2)


