import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

lav = pd.read_csv("campinas_lavouras_permanentes.csv")
cli = pd.read_csv("campinas_clima.csv")

cli["Ano"] = cli["YEAR"]

cli_anual = cli.groupby("Ano").agg(
    Temp_media = ("T2M", "mean"),
    Umidade_media = ("RH2M", "mean"),
    Chuva_total = ("PRECTOTCORR", "sum")
).reset_index()

df = pd.merge(lav, cli_anual, on = "Ano", how = "inner")

mediana_temp = df["Temp_media"].median()
grupo_quente = df.loc[df["Temp_media"] >= mediana_temp, "Rendimento_medio_cafe_kg_ha"]
grupo_frio = df.loc[df["Temp_media"] < mediana_temp, "Rendimento_medio_cafe_kg_ha"]

def cohens_d(a, b):
    n_a, n_b = len(a), len(b)
    dp_pooled = np.sqrt(((n_a - 1) * a.std(ddof = 1)**2 +
                         (n_b - 1) * b.std(ddof = 1)**2) /
                        (n_a + n_b - 2))
    return (a.mean() - b.mean()) / dp_pooled

def interpretar_d(d):
    d = abs(d)
    if d < 0.2:
        return "negligenciável"
    elif d < 0.5:
        return "pequeno"
    elif d < 0.8:
        return "médio"
    else:
        return "grande"

d1 = cohens_d(grupo_quente, grupo_frio)

CORES = ["#1565C0", "#E53935", "#2E7D32", "#F57F17"]

# Figura 1 – Boxplot comparativo (grupos de temperatura)
fig1, ax1 = plt.subplots(figsize = (7, 5))

dados_box = [grupo_frio.values, grupo_quente.values]
bp = ax1.boxplot(dados_box, patch_artist = True, widths = 0.5,
                 medianprops = dict(color = "black", linewidth = 2))

bp["boxes"][0].set_facecolor("#90CAF9")
bp["boxes"][1].set_facecolor("#EF9A9A")

ax1.set_xticks([1, 2])
ax1.set_xticklabels(["Anos mais frios\n(T < mediana)", "Anos mais quentes\n(T ≥ mediana)"], fontsize = 11)
ax1.set_ylabel("Rendimento médio (kg/ha)", fontsize = 11)
ax1.set_title(f"Rendimento do Café por Grupo de Temperatura\n"
              f"Cohen's d = {d1:.2f} (efeito {interpretar_d(d1)})", fontsize = 12)

for i, grupo in enumerate([grupo_frio, grupo_quente], start = 1):
    ax1.plot(i, grupo.mean(), "D", color = "navy", markersize = 7, zorder = 5, label = "Média" if i == 1 else "")
ax1.legend(fontsize = 10)
ax1.grid(axis = "y", alpha = 0.4)

plt.tight_layout()
plt.show()

# Figura 2 – Scatter: Temperatura x Rendimento + regressão
fig2, ax2 = plt.subplots(figsize = (7, 5))

x = df["Temp_media"]
y = df["Rendimento_medio_cafe_kg_ha"]

ax2.scatter(x, y, color = CORES[0], s = 70, zorder = 5, label = "Observações")

m, b, r, p_r, _ = stats.linregress(x, y)
x_linha = np.linspace(x.min(), x.max(), 100)
ax2.plot(x_linha, m * x_linha + b, color = CORES[1], linewidth = 2,
         label = f"Regressão  (r = {r:.2f}, p = {p_r:.3f})")

for _, row in df.iterrows():
    ax2.annotate(str(int(row["Ano"])),
                 xy = (row["Temp_media"], row["Rendimento_medio_cafe_kg_ha"]),
                 xytext = (3, 3), textcoords = "offset points", fontsize = 8, color = "gray")

ax2.set_xlabel("Temperatura média anual (°C)", fontsize = 11)
ax2.set_ylabel("Rendimento médio do café (kg/ha)", fontsize = 11)
ax2.set_title("Relação entre Temperatura e Rendimento do Café", fontsize = 12)
ax2.legend(fontsize = 10)
ax2.grid(alpha = 0.3)

plt.tight_layout()
plt.show()

# Figura 3 – Linha dupla: Rendimento + Chuva ao longo dos anos
fig3, ax_r = plt.subplots(figsize = (9, 5))
ax_c = ax_r.twinx()

ax_r.plot(df["Ano"], df["Rendimento_medio_cafe_kg_ha"],
          marker = "o", color = CORES[0], linewidth = 2, label = "Rendimento (kg/ha)")
ax_c.bar(df["Ano"], df["Chuva_total"], alpha = 0.35, color = CORES[2], label = "Chuva anual (mm)")

ax_r.set_xlabel("Ano", fontsize = 11)
ax_r.set_ylabel("Rendimento médio (kg/ha)", color = CORES[0], fontsize = 11)
ax_c.set_ylabel("Precipitação acumulada (mm/ano)", color = CORES[2], fontsize = 11)
ax_r.set_title("Evolução do Rendimento do Café e Precipitação – Campinas (2014-2024)", fontsize = 12)

lines_r, labels_r = ax_r.get_legend_handles_labels()
lines_c, labels_c = ax_c.get_legend_handles_labels()
ax_r.legend(lines_r + lines_c, labels_r + labels_c, loc = "upper left", fontsize = 10)
ax_r.set_xticks(df["Ano"])
ax_r.tick_params(axis = "x", rotation = 45)
ax_r.grid(alpha = 0.3)

plt.tight_layout()
plt.show()

# Figura 4 – Ilustração visual do Cohen's d
fig4, ax4 = plt.subplots(figsize = (8, 5))

mu1, sd1 = grupo_frio.mean(), grupo_frio.std(ddof = 1)
mu2, sd2 = grupo_quente.mean(), grupo_quente.std(ddof = 1)

x_range = np.linspace(min(mu1, mu2) - 3 * max(sd1, sd2),
                      max(mu1, mu2) + 3 * max(sd1, sd2), 300)

curva1 = stats.norm.pdf(x_range, mu1, sd1)
curva2 = stats.norm.pdf(x_range, mu2, sd2)

ax4.plot(x_range, curva1, color = "#1565C0", linewidth = 2, label = f"Anos frios  (μ = {mu1:.0f})")
ax4.plot(x_range, curva2, color = "#E53935", linewidth = 2, label = f"Anos quentes (μ = {mu2:.0f})")
ax4.fill_between(x_range, curva1, alpha = 0.25, color = "#1565C0")
ax4.fill_between(x_range, curva2, alpha = 0.25, color = "#E53935")

ax4.axvline(mu1, color = "#1565C0", linestyle = "--", alpha = 0.8)
ax4.axvline(mu2, color = "#E53935", linestyle = "--", alpha = 0.8)

ax4.set_xlabel("Rendimento médio do café (kg/ha)", fontsize = 11)
ax4.set_ylabel("Densidade de probabilidade", fontsize = 11)
ax4.set_title(f"Ilustração do Cohen's d = {abs(d1):.2f} (efeito {interpretar_d(d1)})\n"
              f"Diferença de rendimento entre anos quentes e frios", fontsize = 11)
ax4.legend(fontsize = 10)
ax4.grid(alpha = 0.3)

plt.tight_layout()
plt.show()