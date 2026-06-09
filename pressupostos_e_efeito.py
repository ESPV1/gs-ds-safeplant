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

print("=" * 55)
print("1. VERIFICAÇÃO DE PRESSUPOSTOS")
print("=" * 55)

rendimento = df["Rendimento_medio_cafe_kg_ha"].dropna()
chuva = df["Chuva_total"].dropna()
temp = df["Temp_media"].dropna()

# Shapiro-Wilk: p > 0.05 → dados normais
print("\n--- Teste de Shapiro-Wilk (normalidade) ---")
variaveis = {
    "Rendimento café (kg/ha)": rendimento,
    "Chuva anual (mm)": chuva,
    "Temperatura média (°C)": temp,
}

for nome, serie in variaveis.items():
    stat, p = stats.shapiro(serie)
    resultado = "✓ Normal (p > 0.05)" if p > 0.05 else "✗ Não-normal (p ≤ 0.05)"
    print(f"  {nome}")
    print(f"    Estatística W = {stat:.4f} | p-valor = {p:.4f} → {resultado}")

# Levene – homogeneidade das variâncias entre dois grupos
print("\n--- Teste de Levene (homogeneidade de variâncias) ---")
mediana_temp = df["Temp_media"].median()
grupo_quente = df.loc[df["Temp_media"] >= mediana_temp, "Rendimento_medio_cafe_kg_ha"]
grupo_frio   = df.loc[df["Temp_media"] <  mediana_temp, "Rendimento_medio_cafe_kg_ha"]

stat_lev, p_lev = stats.levene(grupo_quente, grupo_frio)
resultado_lev = "✓ Variâncias homogêneas (p > 0.05)" if p_lev > 0.05 else "✗ Variâncias diferentes (p ≤ 0.05)"
print(f"  Grupos: anos mais quentes vs. anos mais frios")
print(f"  Estatística W = {stat_lev:.4f} | p-valor = {p_lev:.4f} → {resultado_lev}")
print()

# QQ-Plot – inspeção visual da normalidade
fig_qq, axes_qq = plt.subplots(1, 3, figsize = (14, 4))
fig_qq.suptitle("QQ-Plot – Verificação Visual de Normalidade", fontsize = 13, fontweight = "bold")

for ax, (nome, serie) in zip(axes_qq, variaveis.items()):
    stats.probplot(serie, dist = "norm", plot = ax)
    ax.set_title(nome, fontsize = 10)
    ax.get_lines()[0].set(markersize = 6, alpha = 0.8, color = "#2196F3")
    ax.get_lines()[1].set(color = "#E53935", linewidth = 1.5)

plt.tight_layout()
plt.show()

print("=" * 55)
print("2. TAMANHO DE EFEITO – COHEN'S d")
print("=" * 55)

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

# Comparação 1: rendimento em anos quentes vs.frios
d1 = cohens_d(grupo_quente, grupo_frio)
print(f"\nComparação: Rendimento café (anos mais quentes vs. mais frios)")
print(f"  Média quentes: {grupo_quente.mean():.1f} kg/ha | Média frios: {grupo_frio.mean():.1f} kg/ha")
print(f"  Cohen's d = {d1:.3f} → efeito {interpretar_d(d1)}")

# Comparação 2: rendimento em anos de muita vs. pouca chuva
mediana_chuva  = df["Chuva_total"].median()
rend_chuva_alt = df.loc[df["Chuva_total"] >= mediana_chuva, "Rendimento_medio_cafe_kg_ha"]
rend_chuva_bx  = df.loc[df["Chuva_total"] < mediana_chuva, "Rendimento_medio_cafe_kg_ha"]

d2 = cohens_d(rend_chuva_alt, rend_chuva_bx)
print(f"\nComparação: Rendimento café (anos chuvosos vs. secos)")
print(f"  Média chuvosos: {rend_chuva_alt.mean():.1f} kg/ha | Média secos: {rend_chuva_bx.mean():.1f} kg/ha")
print(f"  Cohen's d = {d2:.3f} → efeito {interpretar_d(d2)}")

t_stat, t_p = stats.ttest_ind(rend_chuva_alt, rend_chuva_bx)
print(f"  Teste t → t = {t_stat:.3f} | p = {t_p:.4f}")

print()
print("=" * 55)
print("RESUMO DOS RESULTADOS")
print("=" * 55)
print()
print("PRESSUPOSTOS:")
for nome, serie in variaveis.items():
    _, p = stats.shapiro(serie)
    tag = "Normal" if p > 0.05 else "Não-normal"
    print(f"  {nome:35s} → {tag} (p = {p:.4f})")

print(f"\n  Homogeneidade de variâncias (Levene): p = {p_lev:.4f}")
if p_lev > 0.05:
    print("  → Variâncias homogêneas ✓ (pressuposto do t-test atendido)")
else:
    print("  → Variâncias diferentes ✗ (usar Welch's t-test)")

print()
print("TAMANHO DE EFEITO:")
print(f"  Quente vs. Frio (rendimento):  d = {d1:.3f} → {interpretar_d(d1)}")
print(f"  Chuvoso vs. Seco (rendimento): d = {d2:.3f} → {interpretar_d(d2)}")