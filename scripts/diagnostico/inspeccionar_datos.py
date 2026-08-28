import numpy as np
import pandas as pd

BASE = "/home/patto/tesis/datos/zenodo"

a = np.load(f"{BASE}/NVCh_10h_continuous_trace.npy", mmap_mode="r")
print("SHAPE npy:", a.shape, "| dtype:", a.dtype)
print("duracion estimada a 100 Hz:", a.shape[-1] / 100 / 3600, "horas")
print("=" * 70)

df = pd.read_csv(f"{BASE}/NVCh_10h_continuous_trace_reference.csv")
print("COLUMNAS:", list(df.columns))
print("FILAS:", len(df))
print("-" * 70)
print(df.head(10).to_string())
print("-" * 70)
print(df.dtypes)
print("=" * 70)

# Rango de columnas numericas -> permite inferir la unidad de los tiempos
print("RANGOS NUMERICOS (para deducir si son muestras, segundos o timestamps):")
for c in df.select_dtypes(include="number").columns:
    print(f"  {c:<25} min={df[c].min():<20} max={df[c].max():<20}")
print("=" * 70)

# Conteo de clases: detecta automaticamente columnas categoricas
print("COLUMNAS CATEGORICAS:")
for c in df.columns:
    if df[c].nunique() <= 20:
        print(f"\n  --- {c} ---")
        print(df[c].value_counts().to_string())
