import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("📊 Analisis Clustering & Regresi Siswa")

# =====================
# LOAD DATA
# =====================
df = pd.read_csv("dataset_tubes_dt.csv")

st.subheader("Dataset Siswa")
st.write("Ukuran dataset:", df.shape)
st.dataframe(df.head())

# =====================
# TABEL RINGKASAN CLUSTER
# =====================
st.subheader("Ringkasan Cluster")

cluster_summary = df["cluster"].value_counts().reset_index()
cluster_summary.columns = ["Cluster", "Jumlah Data"]
cluster_summary = cluster_summary.sort_values("Cluster")

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
# REGRESI LINEAR (TANPA GRAFIK)
# =====================
st.subheader("Regresi Linear")

X = df[["jam_belajar", "kehadiran", "nilai_tugas"]]
y = df["nilai_akhir"]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.write("### Hasil Evaluasi Model")
st.write("MSE :", mse)
st.write("R2  :", r2)
