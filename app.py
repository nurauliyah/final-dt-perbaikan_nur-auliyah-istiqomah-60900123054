import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("📊 Clustering Aktivitas & Hasil Belajar Siswa")

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
# RINGKASAN JUMLAH CLUSTER
# =====================
st.subheader("Jumlah Data Tiap Cluster")

cluster_summary = df["cluster"].value_counts().reset_index()
cluster_summary.columns = ["Cluster", "Jumlah Data"]
cluster_summary = cluster_summary.sort_values("Cluster")

st.dataframe(cluster_summary)

# =====================
# INTI: AKTIVITAS & HASIL BELAJAR
# =====================
st.subheader("Karakteristik Aktivitas & Hasil Belajar per Cluster")

cluster_profile = (
    df.groupby("cluster")[["jam_belajar", "kehadiran", "nilai_tugas"]]
    .mean()
    .reset_index()
)

cluster_profile.columns = [
    "Cluster",
    "Rata-rata Jam Belajar",
    "Rata-rata Kehadiran",
    "Rata-rata Nilai Tugas"
]

st.dataframe(cluster_profile)

# =====================
# VISUALISASI CLUSTER
# =====================
st.subheader("Visualisasi Clustering")

fig = plt.figure()
plt.scatter(df["jam_belajar"], df["nilai_tugas"], c=df["cluster"])
plt.xlabel("Jam Belajar")
plt.ylabel("Nilai Tugas")
plt.title("Hasil Clustering Siswa")
st.pyplot(fig)



