import pandas as pd
import numpy as np
from io import StringIO

# Carregando dataset de clima (pulando o cabeçalho da NASA)
with open("campinas_clima.csv", "r") as f:
    lines = f.readlines()

start = 0
for i, line in enumerate(lines):
    if line.strip().startswith("YEAR"):
        start = i
        break

clima = pd.read_csv(StringIO("".join(lines[start:])))
clima = clima.replace(-999, np.nan)

# Carregando dataset de café
cafe = pd.read_csv("campinas_lavouras_permanentes.csv")

# Função de resumo (amostral, ddof=1)
def resumo(s, nome):
    print(f"\n--- {nome} ---")
    print(f"Média:              {s.mean():.2f}")
    print(f"Mediana:            {s.median():.2f}")
    print(f"Moda:               {s.mode().iloc[0]:.2f}")
    print(f"Variância amostral: {s.var(ddof=1):.2f}")
    print(f"DP amostral:        {s.std(ddof=1):.2f}")
    print(f"Mínimo:             {s.min():.2f}")
    print(f"Máximo:             {s.max():.2f}")
    print(f"Q1:                 {s.quantile(0.25):.2f}")
    print(f"Q3:                 {s.quantile(0.75):.2f}")

resumo(clima["T2M"].dropna(),        "Temperatura (°C)")
resumo(clima["RH2M"].dropna(),       "Umidade relativa (%)")
resumo(clima["PRECTOTCORR"].dropna(),"Precipitação (mm/dia)")
resumo(cafe["Rendimento_medio_cafe_kg_ha"], "Rendimento do café (kg/ha)")
resumo(cafe["Quantidade_produzida_cafe_t"], "Quantidade produzida (t)")