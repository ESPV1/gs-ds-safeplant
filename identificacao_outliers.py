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

def iqr_outliers(series):
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    mask = (series < low) | (series > high)
    return mask, low, high

for col, label, fname in [
    ("T2M", "Temperatura diária (°C)", "boxplot_T2M"),
    ("RH2M", "Umidade relativa diária (%)", "boxplot_RH2M"),
    ("PRECTOTCORR", "Precipitação diária (mm/dia)", "boxplot_PRECTOTCORR")
]:
    mask, low, high = iqr_outliers(clima[col])
    print(f"\n{col} | Limites: [{low:.2f}, {high:.2f}]")
    print(f"Outliers: {mask.sum()} de {len(mask)} ({100*mask.sum()/len(mask):.1f}%)")

    plt.figure(figsize=(6, 5))
    plt.boxplot(clima[col].dropna().values, vert=True)
    plt.ylabel(label)
    plt.title(f"Boxplot: {label}")
    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.show()

for col, label, fname in [
    ("Rendimento_medio_cafe_kg_ha", "Rendimento médio do café (kg/ha)", "boxplot_rendimento"),
    ("Quantidade_produzida_cafe_t", "Quantidade produzida de café (t)", "boxplot_quantidade")
]:
    mask, low, high = iqr_outliers(cafe[col])
    print(f"\n{col} | Limites: [{low:.2f}, {high:.2f}]")
    print(f"Outliers: {mask.sum()} de {len(mask)}")
    if mask.sum() > 0:
        print(f"Anos: {cafe.loc[mask, 'Ano'].values}")
        print(f"Valores: {cafe.loc[mask, col].values}")

    plt.figure(figsize=(6, 5))
    plt.boxplot(cafe[col].dropna().values, vert=True)
    plt.ylabel(label)
    plt.title(f"Boxplot: {label}")
    plt.tight_layout()
    plt.savefig(f"{fname}.png", dpi=150, bbox_inches="tight")
    plt.show()