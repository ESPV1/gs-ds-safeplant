import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# =====================
# CARREGAMENTO DOS DADOS
# =====================

with open("campinas_clima.csv", "r") as f:
    lines = f.readlines()

start = 0
for i, line in enumerate(lines):
    if line.strip().startswith("YEAR"):
        start = i
        break

clima = pd.read_csv(StringIO("".join(lines[start:])))
clima = clima.replace(-999, np.nan)

cafe = pd.read_csv("campinas_lavouras_permanentes.csv")

# Agregar clima por ano
clima_anual = clima.groupby("YEAR")[["T2M", "RH2M", "PRECTOTCORR"]].mean().reset_index()

# Merge dos dois datasets
df = pd.merge(clima_anual, cafe[["Ano", "Rendimento_medio_cafe_kg_ha", "Quantidade_produzida_cafe_t"]],
              left_on="YEAR", right_on="Ano")

# =====================
# SEÇÃO 5 — CORRELAÇÃO
# =====================

# Matriz de correlação
corr = df[["T2M", "RH2M", "PRECTOTCORR", "Rendimento_medio_cafe_kg_ha"]].corr()

pd.set_option('display.max_columns', None)

print("Correlação com Rendimento do café:")
print(corr["Rendimento_medio_cafe_kg_ha"].sort_values(ascending=False))

print("\nMatriz completa:")
print(corr.round(2))

# =====================
# HEATMAP
# =====================

fig, ax = plt.subplots(figsize=(8, 6))
mat = corr.values
cols = corr.columns

im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(["Temperatura", "Umidade", "Precipitação", "Rendimento café"], rotation=45, ha="right")
ax.set_yticklabels(["Temperatura", "Umidade", "Precipitação", "Rendimento café"])

for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=11)

ax.set_title("Heatmap de Correlação — Clima x Rendimento do Café")
plt.tight_layout()
plt.show()

# =====================
# SCATTERPLOTS
# =====================

for feature, label in [("T2M", "Temperatura média anual (°C)"),
                        ("RH2M", "Umidade relativa média anual (%)"),
                        ("PRECTOTCORR", "Precipitação média anual (mm/dia)")]:
    plt.figure(figsize=(7, 5))
    plt.scatter(df[feature], df["Rendimento_medio_cafe_kg_ha"], color="steelblue", alpha=0.8)
    plt.xlabel(label)
    plt.ylabel("Rendimento do café (kg/ha)")
    plt.title(f"Scatterplot: {label} vs Rendimento do Café")
    plt.tight_layout()
    plt.show()