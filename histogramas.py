import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

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

variaveis_clima = [
    ("T2M",          "Temperatura diária (°C)",      "#1565C0"),
    ("RH2M",         "Umidade relativa diária (%)",   "#2E7D32"),
    ("PRECTOTCORR",  "Precipitação diária (mm/dia)",  "#6A1B9A"),
]

variaveis_cafe = [
    ("Rendimento_medio_cafe_kg_ha", "Rendimento médio do café (kg/ha)", "#E53935"),
    ("Quantidade_produzida_cafe_t", "Quantidade produzida (t)",          "#F57F17"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Histogramas – Variáveis Climáticas (Campinas, diário)", fontsize=13, fontweight="bold")

for ax, (col, label, cor) in zip(axes, variaveis_clima):
    dados = clima[col].dropna()
    ax.hist(dados, bins=40, color=cor, edgecolor="white", alpha=0.85)
    ax.axvline(dados.mean(),   color="black",  linestyle="--", linewidth=1.4, label=f"Média {dados.mean():.1f}")
    ax.axvline(dados.median(), color="orange", linestyle=":",  linewidth=1.4, label=f"Mediana {dados.median():.1f}")
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Frequência", fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("hist_clima.png", dpi=150, bbox_inches="tight")
plt.show()

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Histogramas – Produção de Café (Campinas, anual)", fontsize=13, fontweight="bold")

for ax, (col, label, cor) in zip(axes2, variaveis_cafe):
    dados = cafe[col].dropna()
    ax.hist(dados, bins=10, color=cor, edgecolor="white", alpha=0.85)
    ax.axvline(dados.mean(),   color="black",  linestyle="--", linewidth=1.4, label=f"Média {dados.mean():.0f}")
    ax.axvline(dados.median(), color="navy",   linestyle=":",  linewidth=1.4, label=f"Mediana {dados.median():.0f}")
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel("Frequência", fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("hist_cafe.png", dpi=150, bbox_inches="tight")
plt.show()

print("Histogramas gerados: hist_clima.png, hist_cafe.png")
