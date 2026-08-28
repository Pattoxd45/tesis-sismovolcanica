import numpy as np
import pandas as pd

BASE = "/home/patto/tesis/datos/zenodo"
a = np.load(f"{BASE}/NVCh_10h_continuous_trace.npy", mmap_mode="r")
df = pd.read_csv(f"{BASE}/NVCh_10h_continuous_trace_reference.csv")
CORTES = [2_160_000, 3_240_001]

# --- 1. Los huecos, son los MISMOS indices? ---
print("Comparacion exacta de patrones de hueco (bloque 0:1.000.000)")
z = {i: (np.asarray(a[i, :1_000_000]) == 0) for i in range(1, 8)}
print("     " + "".join(f"{j:>7}" for j in range(1, 8)))
for i in range(1, 8):
    fila = "".join(f"{np.mean(z[i] == z[j]):>7.3f}" for j in range(1, 8))
    print(f"  {i}  {fila}")
print("(1.000 = huecos identicos -> misma estacion)")

# --- 2. Correlacion de ENVOLVENTES sobre un evento real ---
ev = df.iloc[20]
s, e = int(ev.idx_start), int(ev.idx_end)
pad = (e - s) // 2
seg = np.asarray(a[1:8, max(0, s - pad):e + pad], dtype=np.float64)
env = np.abs(seg)
k = 201
env = np.apply_along_axis(lambda v: np.convolve(v, np.ones(k) / k, mode="same"), 1, env)
E = np.corrcoef(env)
print(f"\nCorrelacion de envolventes sobre evento {ev.event_type} [{s}:{e}]")
print("     " + "".join(f"{j:>7}" for j in range(1, 8)))
for i in range(7):
    print(f"  {i+1}  " + "".join(f"{E[i,j]:>7.2f}" for j in range(7)))
print("(>0.7 entre canales = misma estacion, 3 componentes)")

# --- 3. Eventos del catalogo por segmento y en los bordes ---
bordes = [0] + CORTES + [a.shape[1]]
print("\nEventos por segmento:")
for k_ in range(3):
    ini, fin = bordes[k_], bordes[k_ + 1]
    sub = df[(df.idx_start >= ini) & (df.idx_end <= fin)]
    print(f"  seg {k_+1} [{ini}:{fin}] -> {len(sub):>3} eventos | "
          f"{dict(sub.event_type.value_counts())}")

cruzan = df[[(any(s_ < c < e_ for c in CORTES))
             for s_, e_ in zip(df.idx_start, df.idx_end)]]
print(f"\nEventos que cruzan una union: {len(cruzan)}")
if len(cruzan):
    print(cruzan.to_string())

# --- 4. Duraciones (para dimensionar el clasificador) ---
dur = (df.idx_end - df.idx_start) / 100
print("\nDuracion de eventos [s] por clase:")
print(df.assign(dur=dur).groupby("event_type")["dur"]
        .agg(["count", "min", "median", "max"]).round(1).to_string())
