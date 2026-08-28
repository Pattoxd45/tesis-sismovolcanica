import numpy as np
import pandas as pd

BASE = "/home/patto/tesis/datos/zenodo"
a = np.load(f"{BASE}/NVCh_10h_continuous_trace.npy", mmap_mode="r")
df = pd.read_csv(f"{BASE}/NVCh_10h_continuous_trace_reference.csv")

print("Perfil de cada fila (submuestreado 1:100 para ir rapido)")
print(f"{'fila':<5} {'min':>16} {'max':>16} {'n_unicos':>9} {'binaria?':>9} {'monotona?':>10}")
for i in range(a.shape[0]):
    r = np.asarray(a[i, ::100], dtype=np.float64)
    u = np.unique(r)
    binaria = set(np.round(u, 6)).issubset({0.0, 1.0})
    mono = bool(np.all(np.diff(r) > 0))
    print(f"{i:<5} {r.min():>16.4f} {r.max():>16.4f} {len(u):>9} {str(binaria):>9} {str(mono):>10}")

print("\nSi la fila 0 es tiempo Unix, esto debe dar una fecha real:")
print("  inicio:", pd.to_datetime(float(a[0, 0]), unit="s", errors="coerce"))
print("  fin   :", pd.to_datetime(float(a[0, -1]), unit="s", errors="coerce"))
print("  dt medio (s):", float(a[0, 1000]) - float(a[0, 999]))

# Validacion cruzada: la mascara one-hot debe activarse en los intervalos del CSV
print("\nValidacion mascaras vs CSV (primeros 3 eventos):")
for _, ev in df.head(3).iterrows():
    s, e = int(ev.idx_start), int(ev.idx_end)
    seg = np.asarray(a[9:15, s:e], dtype=np.float64)
    activas = [j for j in range(seg.shape[0]) if seg[j].mean() > 0.5]
    print(f"  {ev.event_type:<3} [{s}:{e}] -> filas 9-14 activas: {[9+j for j in activas]}")
