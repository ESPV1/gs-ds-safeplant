import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

cli_anual = clima.groupby("YEAR").agg(
    Temp_media=("T2M", "mean"),
    Umidade_media=("RH2M", "mean"),
    Chuva_total=("PRECTOTCORR", "sum")
).reset_index()

def ic_95(serie):
    s = serie.dropna()
    n = len(s)
    media = s.mean()
    ep = stats.sem(s)
    t_critico = stats.t.ppf(0.975, df=n - 1)
    margem = t_critico * ep
    return media, margem, media - margem, media + margem, n

print("=" * 60)
print("INTERVALOS DE CONFIANÇA (95%) – distribuição t de Student")
print("=" * 60)

variaveis = {
    "Temperatura média anual (°C)":        cli_anual["Temp_media"],
    "Umidade relativa média anual (%)":    cli_anual["Umidade_media"],
    "Precipitação acumulada anual (mm)":   cli_anual["Chuva_total"],
    "Rendimento médio do café (kg/ha)":    cafe["Rendimento_medio_cafe_kg_ha"],
    "Quantidade produzida de café (t)":    cafe["Quantidade_produzida_cafe_t"],
}

resultados = []
for nome, serie in variaveis.items():
    media, margem, li, ls, n = ic_95(serie)
    resultados.append((nome, media, li, ls, margem, n))
    print(f"\n{nome}")
    print(f"  n = {n} | Média = {media:.2f}")
    print(f"  IC 95%: [{li:.2f} ; {ls:.2f}]  (±{margem:.2f})")

fig, ax = plt.subplots(figsize=(10, 6))

nomes_curtos = ["Temperatura\n(°C)", "Umidade\n(%)", "Precipitação\n(mm/ano)", "Rendimento\ncafé (kg/ha)", "Qtd produzida\n(t)"]
medias  = [r[1] for r in resultados]
erros   = [r[4] for r in resultados]

cores = ["#1565C0", "#2E7D32", "#6A1B9A", "#E53935", "#F57F17"]

for i, (nome, media, li, ls, margem, n) in enumerate(resultados):
    ax.errorbar(i, media, yerr=margem, fmt="o", color=cores[i],
                capsize=6, capthick=2, elinewidth=2, markersize=8)

ax.set_xticks(range(len(nomes_curtos)))
ax.set_xticklabels(nomes_curtos, fontsize=9)
ax.set_ylabel("Valor médio com IC 95%", fontsize=11)
ax.set_title("Intervalos de Confiança (95%) por Variável", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

print("\nGráfico salvo: intervalos_confianca.png")
