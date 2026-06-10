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
    Chuva_total=("PRECTOTCORR", "sum")
).reset_index().rename(columns={"YEAR": "Ano"})

df = pd.merge(cafe, cli_anual, on="Ano", how="inner")

mediana_temp = df["Temp_media"].median()
grupo_quente = df.loc[df["Temp_media"] >= mediana_temp, "Rendimento_medio_cafe_kg_ha"]
grupo_frio   = df.loc[df["Temp_media"] <  mediana_temp, "Rendimento_medio_cafe_kg_ha"]

print("=" * 60)
print("TESTE DE HIPÓTESE 1 – Temperatura e Rendimento do Café")
print("=" * 60)
print()
print("H0: O rendimento medio do cafe e igual entre anos mais")
print("    quentes e anos mais frios (media_quente = media_frio)")
print("H1: O rendimento medio difere entre os dois grupos")
print("    (media_quente != media_frio)  -- teste bicaudal, alpha = 0,05")
print()

stat_lev, p_lev = stats.levene(grupo_quente, grupo_frio)
igual_var = p_lev > 0.05
print(f"Levene (homogeneidade): W = {stat_lev:.4f} | p = {p_lev:.4f}")
print(f"-> Variancias {'homogeneas -> t-test padrao' if igual_var else 'diferentes -> Welch t-test'}")
print()

t_stat, p_valor = stats.ttest_ind(grupo_quente, grupo_frio, equal_var=igual_var)
print(f"Estatistica t = {t_stat:.4f}")
print(f"p-valor       = {p_valor:.4f}")
print()

alpha = 0.05
if p_valor < alpha:
    print(f"Decisao: p ({p_valor:.4f}) < alpha ({alpha}) -> REJEITA H0")
    print("Conclusao: ha diferenca estatisticamente significativa no")
    print("rendimento entre anos quentes e frios.")
else:
    print(f"Decisao: p ({p_valor:.4f}) >= alpha ({alpha}) -> NAO rejeita H0")
    print("Conclusao: nao ha evidencia suficiente de diferenca")
    print("significativa no rendimento entre anos quentes e frios.")

mediana_chuva   = df["Chuva_total"].median()
grupo_chuvoso   = df.loc[df["Chuva_total"] >= mediana_chuva, "Rendimento_medio_cafe_kg_ha"]
grupo_seco      = df.loc[df["Chuva_total"] <  mediana_chuva, "Rendimento_medio_cafe_kg_ha"]

print()
print("=" * 60)
print("TESTE DE HIPÓTESE 2 – Precipitação e Rendimento do Café")
print("=" * 60)
print()
print("H0: O rendimento medio do cafe e igual entre anos chuvosos")
print("    e anos secos (media_chuvoso = media_seco)")
print("H1: O rendimento medio difere entre os dois grupos")
print("    (media_chuvoso != media_seco)  -- teste bicaudal, alpha = 0,05")
print()

stat_lev2, p_lev2 = stats.levene(grupo_chuvoso, grupo_seco)
igual_var2 = p_lev2 > 0.05
print(f"Levene (homogeneidade): W = {stat_lev2:.4f} | p = {p_lev2:.4f}")
print(f"-> Variancias {'homogeneas -> t-test padrao' if igual_var2 else 'diferentes -> Welch t-test'}")
print()

t_stat2, p_valor2 = stats.ttest_ind(grupo_chuvoso, grupo_seco, equal_var=igual_var2)
print(f"Estatistica t = {t_stat2:.4f}")
print(f"p-valor       = {p_valor2:.4f}")
print()

if p_valor2 < alpha:
    print(f"Decisao: p ({p_valor2:.4f}) < alpha ({alpha}) -> REJEITA H0")
    print("Conclusao: ha diferenca estatisticamente significativa no")
    print("rendimento entre anos chuvosos e secos.")
else:
    print(f"Decisao: p ({p_valor2:.4f}) >= alpha ({alpha}) -> NAO rejeita H0")
    print("Conclusao: nao ha evidencia suficiente de diferenca")
    print("significativa no rendimento entre anos chuvosos e secos.")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Teste t – Comparação de Grupos de Rendimento do Café", fontsize=13, fontweight="bold")

ax1 = axes[0]
bp1 = ax1.boxplot([grupo_frio.values, grupo_quente.values], patch_artist=True,
                  medianprops=dict(color="black", linewidth=2))
bp1["boxes"][0].set_facecolor("#90CAF9")
bp1["boxes"][1].set_facecolor("#EF9A9A")
ax1.set_xticks([1, 2])
ax1.set_xticklabels(["Anos frios\n(T < mediana)", "Anos quentes\n(T >= mediana)"])
ax1.set_ylabel("Rendimento (kg/ha)")
ax1.set_title(f"Por Temperatura\nt = {t_stat:.3f} | p = {p_valor:.4f}")
ax1.grid(axis="y", alpha=0.3)

ax2 = axes[1]
bp2 = ax2.boxplot([grupo_seco.values, grupo_chuvoso.values], patch_artist=True,
                  medianprops=dict(color="black", linewidth=2))
bp2["boxes"][0].set_facecolor("#A5D6A7")
bp2["boxes"][1].set_facecolor("#FFE082")
ax2.set_xticks([1, 2])
ax2.set_xticklabels(["Anos secos\n(chuva < mediana)", "Anos chuvosos\n(chuva ≥ mediana)"])
ax2.set_ylabel("Rendimento (kg/ha)")
ax2.set_title(f"Por Precipitação\nt = {t_stat2:.3f} | p = {p_valor2:.4f}")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("teste_hipotese.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nGráfico salvo: teste_hipotese.png")
