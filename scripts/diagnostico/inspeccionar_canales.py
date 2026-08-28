import numpy as np

BASE = "/home/patto/tesis/datos/zenodo"
a = np.load(f"{BASE}/NVCh_10h_continuous_trace.npy", mmap_mode="r")
t = np.asarray(a[0], dtype=np.float64)

# --- 1. Uniones entre los segmentos concatenados ---
d = np.diff(t)
saltos = np.where(np.abs(d - 0.01) > 1e-6)[0]
print(f"Discontinuidades temporales: {len(saltos)}")
for s in saltos:
    print(f"  muestra {s:>9} | salto de {d[s]:.1f} s")

# --- 2. Canales vacios / con huecos (senal completa) ---
print("\nCanal | frac. ceros | std")
for i in range(1, 9):
    r = np.asarray(a[i], dtype=np.float64)
    print(f"  {i}   |   {np.mean(r == 0):.4f}    | {r.std():.2f}")

# --- 3. Correlacion entre canales -> agrupacion por estacion ---
print("\nCorrelacion (bloque de 200k muestras con senal):")
blk = np.asarray(a[1:9, 500_000:700_000], dtype=np.float64)
C = np.corrcoef(blk)
print("     " + "".join(f"{j:>7}" for j in range(1, 9)))
for i in range(8):
    print(f"  {i+1}  " + "".join(f"{C[i,j]:>7.2f}" for j in range(8)))
