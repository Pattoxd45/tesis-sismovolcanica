# Tarea B-01: verifica si ventanas del pool de entrenamiento
# (datos/zenodo/NVCHVC_8192.zip) aparecen dentro de la traza continua de
# Zenodo. Solo mide: no excluye nada del entrenamiento.
#
# Pasos:
#   1. Inspeccion del zip sin supuestos: archivos, forma y dtype (leyendo la
#      cabecera de TODOS los .npy), patron de nombres, filas presentes,
#      normalizacion, mascaras y contenido espectral comparado con la traza.
#   2. Validacion del metodo con solucion conocida:
#        - positivos: fragmentos recortados de la propia traza (ventana
#          completa y segmento de evento) deben encontrarse con correlacion
#          cercana a 1 en el indice exacto;
#        - nulo: ventanas del pool de fechas que no pueden estar en la traza
#          (mes-dia a mas de un dia de las tres fechas de la traza) dan la
#          distribucion de la correlacion maxima por azar o por parecido de
#          forma de onda (multipletes). Con esto se fija el umbral.
#   3. Busqueda por correlacion cruzada normalizada (FFT) de cada ventana del
#      pool contra la fila de la misma estacion (fila i del pool <-> fila
#      i + 1 de la traza, ver inspeccion). Dos modos:
#        - "ventana": las 8.192 muestras completas;
#        - "evento": solo el segmento marcado como evento en la mascara del
#          pool. Detecta el caso en que el evento se haya insertado en la
#          traza con un ruido de fondo distinto.
#      Criterios de coincidencia: NCC media entre estaciones >= umbral, o
#      NCC de la mejor estacion sola >= UMBRAL_1EST (modo evento). El segundo
#      se agrego tras la primera corrida: hay ventanas que calzan casi
#      exactamente en una sola estacion y la media las diluye.
#      Orden: los 31 archivos que coinciden en mes, dia y hora con la traza;
#      los AV del pool; y, con --todo, el resto del pool.
#   4. Reporte de coincidencias: archivo, clase, indice, tramo, hora UTC,
#      correlacion, evento del catalogo con mayor IoU y si es uno de los 17 AV
#      del tramo 1 que no aparecen en el crudo de FRE.
#
# Uso:
#   python superposicion_pool_traza.py            # pasos 1-4 con 31 + AV
#   python superposicion_pool_traza.py --todo     # agrega el resto del pool
import os
import io
import re
import sys
import time
import zipfile
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy.fft as sf
from scipy.signal import butter, sosfiltfilt, welch

ZIP_POOL = os.path.expanduser("~/tesis/datos/zenodo/NVCHVC_8192.zip")
NPY_TRAZA = os.path.expanduser("~/tesis/datos/zenodo/NVCh_10h_continuous_trace.npy")
CSV_CATALOGO = os.path.expanduser("~/tesis/datos/zenodo/NVCh_10h_continuous_trace_reference.csv")
DIR_SALIDA = os.path.expanduser("~/tesis/resultados")

FS = 100.0
M = 8192
SOS = butter(5, [1.0, 15.0], btype="bandpass", fs=FS, output="sos")
CLASES_MASCARA = ("BG", "VT", "LP", "TR", "AV", "IC")   # filas 8-13 del pool, 9-14 de la traza

# Tramos de la traza (indices de muestra) y fecha UTC de cada uno
TRAMOS = [
    (1, 0, 2_160_000, "2018-02-09"),
    (2, 2_160_000, 3_240_001, "2020-01-14"),
    (3, 3_240_001, 3_664_193, "2017-01-10"),
]
# Criterio de los 31 candidatos: mes, dia y hora del nombre dentro de las
# horas cubiertas por cada tramo (horas UTC 04-10, 01-04 y 06-07)
HORAS_TRAMO = {("02", "09"): range(4, 11), ("01", "14"): range(1, 5), ("01", "10"): range(6, 8)}
# Para el nulo se excluyen ademas los dias vecinos, por si el nombre estuviera
# en hora local (Chile, UTC-3/-4) y cambiara de dia
FECHAS_EXCLUIDAS_NULO = {("02", "08"), ("02", "09"), ("02", "10"),
                         ("01", "13"), ("01", "14"), ("01", "15"),
                         ("01", "09"), ("01", "10"), ("01", "11")}

# Los 17 AV del tramo 1 sin contraparte en el crudo de FRE (idx_start del
# catalogo). Fuente: salida de scripts/diagnostico/verificar_crudos.py,
# paso 5, estado "no_coincide" (ejecutada el 1 de octubre de 2026)
AV_SIN_CRUDO = {9005, 169581, 176397, 187288, 242412, 273545, 433062, 455131, 478537,
                803794, 872366, 967042, 1142426, 1149460, 1425188, 1636140, 1722083}

UMBRAL_VENTANA = 0.90   # se justifica con el paso 2 (se imprime la comparacion)
UMBRAL_EVENTO = 0.90
# Criterio por estacion: la mejor estacion sola, en modo evento. Necesario
# porque una coincidencia en una sola estacion se diluye en la media
UMBRAL_1EST = 0.98
GUARDA_STD = 1e-3       # std local minima, relativa a la std global de la fila
N_NULO = 300
N_POSITIVOS = 24
SEMILLA = 0


def utc(t):
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def tramo_de(idx):
    for n, a, b, _ in TRAMOS:
        if a <= idx < b:
            return n
    return None


def partes_nombre(ruta):
    # carpeta/MMDDHHMM.9CLASE[_k].npy
    carpeta, base = ruta.split("/")
    m = re.fullmatch(r"(\d{2})(\d{2})(\d{2})(\d{2})\.(\d)([A-Z]{2})(?:_(\d+))?\.npy", base)
    return dict(archivo=ruta, carpeta=carpeta, mes=m.group(1), dia=m.group(2),
                hora=int(m.group(3)), minuto=int(m.group(4)), digito=m.group(5),
                clase_nombre=m.group(6), sufijo=m.group(7))


def es_candidato(p):
    horas = HORAS_TRAMO.get((p["mes"], p["dia"]))
    return horas is not None and p["hora"] in horas


def leer(z, ruta):
    return np.load(io.BytesIO(z.read(ruta)), allow_pickle=False)


def segmento_evento(w):
    # Tramo contiguo mas largo con mascara distinta de BG (fila 8 = BG)
    ev = w[8] < 0.5
    if not ev.any():
        return None
    d = np.diff(np.r_[0, ev.astype(np.int8), 0])
    ini, fin = np.where(d == 1)[0], np.where(d == -1)[0]
    k = int(np.argmax(fin - ini))
    return int(ini[k]), int(fin[k])


# ----------------------------------------------------------------------------
# Correlacion cruzada normalizada contra la traza
# ----------------------------------------------------------------------------
class Buscador:
    # Precalcula, por fila de la traza, su FFT y sus sumas acumuladas para
    # obtener la correlacion normalizada de una plantilla en todos los
    # desfases con una sola FFT inversa.
    def __init__(self, traza):
        self.N = traza.shape[1]
        self.nfft = sf.next_fast_len(self.N + M, real=True)
        self.x, self.X, self.S1, self.S2, self.std_global = {}, {}, {}, {}, {}
        for fila in range(1, 9):
            x = np.asarray(traza[fila], dtype=np.float64)
            nz = x[x != 0]
            if nz.size < M or nz.std() == 0:
                continue        # fila sin datos utilizables (la 8 es identicamente cero)
            self.x[fila] = x
            self.X[fila] = sf.rfft(x.astype(np.float32), self.nfft, workers=-1)
            self.S1[fila] = np.r_[0.0, np.cumsum(x)]
            self.S2[fila] = np.r_[0.0, np.cumsum(x * x)]
            self.std_global[fila] = nz.std()

    def ncc(self, plantilla, fila):
        # NCC(l) = sum_k t'[k] x[l+k] / (||t'|| * ||x[l:l+L] - media||), l = 0..N-L
        L = len(plantilla)
        t = plantilla - plantilla.mean()
        nt = np.sqrt(np.dot(t, t))
        T = sf.rfft(t.astype(np.float32), self.nfft, workers=-1)
        c = sf.irfft(self.X[fila] * np.conj(T), self.nfft, workers=-1)[: self.N - L + 1]
        s1 = self.S1[fila][L:] - self.S1[fila][:-L]
        s2 = self.S2[fila][L:] - self.S2[fila][:-L]
        var = np.maximum(s2 - s1 * s1 / L, 0.0)
        den = nt * np.sqrt(var)
        ok = var > L * (GUARDA_STD * self.std_global[fila]) ** 2
        r = np.zeros(self.N - L + 1)
        r[ok] = c[ok] / den[ok]
        return r

    def ncc_exacta(self, plantilla, fila, l):
        # Recalculo en float64 en un desfase dado (verifica el resultado FFT)
        L = len(plantilla)
        a = plantilla - plantilla.mean()
        b = self.x[fila][l:l + L] - self.x[fila][l:l + L].mean()
        d = np.sqrt(np.dot(a, a) * np.dot(b, b))
        return float(np.dot(a, b) / d) if d > 0 else 0.0

    def buscar(self, w, ini=0, fin=M):
        # w: ventana (14, 8192) con filas de estacion 0-7. Correlaciona el
        # segmento [ini, fin) de cada estacion presente contra la fila i + 1 y
        # promedia las NCC por desfase. Devuelve el indice de la traza donde cae
        # la muestra 0 de la ventana.
        filas = [i for i in range(8) if w[i, ini:fin].std() > 0 and (i + 1) in self.X]
        if not filas:
            return None
        rs = [self.ncc(w[i, ini:fin].astype(np.float64), i + 1) for i in filas]
        media = np.mean(rs, axis=0)
        l = int(np.argmax(media))
        por_est = [float(r[l]) for r in rs]
        # Mejor estacion individual (en su propio desfase)
        mejores = [(float(r.max()), int(r.argmax()), i) for r, i in zip(rs, filas)]
        r1, l1, i1 = max(mejores)
        r1 = self.ncc_exacta(w[i1, ini:fin].astype(np.float64), i1 + 1, l1)
        exacta = np.mean([self.ncc_exacta(w[i, ini:fin].astype(np.float64), i + 1, l)
                          for i in filas])
        return dict(ncc=float(exacta), ncc_fft=float(media[l]), idx=l - ini,
                    n_est=len(filas), filas=",".join(str(i) for i in filas),
                    ncc_por_est=",".join(f"{v:.3f}" for v in por_est),
                    ncc_1est=r1, idx_1est=l1 - ini, fila_1est=i1)


# ----------------------------------------------------------------------------
# 1. Inspeccion
# ----------------------------------------------------------------------------
def inspeccionar(z, nombres, traza, rng):
    print("=" * 78)
    print("1. INSPECCION DEL POOL (NVCHVC_8192.zip)")
    print("=" * 78)
    carpetas = sorted({n.split("/")[0] for n in z.namelist() if n.endswith("/")})
    print(f"Entradas: {len(z.namelist()):,} ({len(carpetas)} carpetas: {carpetas}; "
          f"{len(nombres):,} archivos)")
    print("Archivos por carpeta:", pd.Series([n.split("/")[0] for n in nombres]).value_counts().to_dict())

    # Cabecera de todos los .npy (sin leer los datos)
    formas = {}
    for n in nombres:
        with z.open(n) as f:
            ver = np.lib.format.read_magic(f)
            leer_cab = (np.lib.format.read_array_header_1_0 if ver == (1, 0)
                        else np.lib.format.read_array_header_2_0)
            forma, fortran, dtype = leer_cab(f)
            formas[(forma, str(dtype), fortran)] = formas.get((forma, str(dtype), fortran), 0) + 1
    print("Cabeceras .npy (forma, dtype, fortran_order) -> n archivos:", formas)

    partes = pd.DataFrame([partes_nombre(n) for n in nombres])
    print("\nPatron de nombre: carpeta/MMDDHHMM.<digito><CLASE>[_<k>].npy")
    print("  digito tras el punto:", partes.digito.value_counts().to_dict(),
          "(constante; significado desconocido, no es un anio)")
    print("  clase del nombre == carpeta en todos:", bool((partes.clase_nombre == partes.carpeta).all()))
    print("  sufijo _k (solo TR):", partes.groupby("carpeta").sufijo.value_counts().to_dict())
    print(f"  rangos: mes {partes.mes.min()}-{partes.mes.max()}, dia {partes.dia.min()}-{partes.dia.max()}, "
          f"hora {partes.hora.min()}-{partes.hora.max()}, minuto {partes.minuto.min()}-{partes.minuto.max()}")
    raiz = partes.archivo.str.split("/").str[1].str[:8]
    rep = raiz.value_counts()
    print(f"  MMDDHHMM repetidos en mas de un archivo: {int((rep > 1).sum())} "
          f"(el nombre no identifica de forma unica una ventana)")
    print("  El nombre NO trae anio ni zona horaria; tampoco hay fila de tiempo en el arreglo.")

    # Contenido de una muestra aleatoria
    muestra = rng.choice(nombres, 400, replace=False)
    maxabs, fila_max, presentes, uno_caliente, coincide_clase = [], [], np.zeros(8, int), [], []
    dur_ev, en_borde, refilt, esp_lo, esp_hi = [], [], [], [], []
    for n in muestra:
        w = leer(z, n)
        s = w[:8]
        maxabs.append(np.abs(s).max())
        fila_max.append(int(np.argmax(np.abs(s).max(1))))
        presentes += (s.std(1) > 0)
        m = w[8:]
        uno_caliente.append(bool(np.all((m == 0) | (m == 1)) and np.allclose(m.sum(0), 1)))
        activas = [CLASES_MASCARA[k] for k in range(1, 6) if m[k].any()]
        coincide_clase.append(activas == [n.split("/")[0]])
        seg = segmento_evento(w)
        if seg:
            dur_ev.append((seg[1] - seg[0]) / FS)
            en_borde.append(seg[0] == 0 or seg[1] == M)
        for i in np.where(s.std(1) > 0)[0]:
            f, p = welch(s[i], fs=FS, nperseg=1024)
            esp_lo.append(p[f < 0.8].sum() / p.sum())
            esp_hi.append(p[f > 16].sum() / p.sum())
            y = sosfiltfilt(SOS, s[i])
            refilt.append(np.corrcoef(s[i][500:-500], y[500:-500])[0, 1])
    print(f"\nMuestra aleatoria de {len(muestra)} ventanas (semilla {SEMILLA}):")
    print(f"  filas 0-7 = estaciones; filas 8-13 = mascaras {CLASES_MASCARA}")
    print(f"  max|x| de las filas 0-7 por ventana: min={np.min(maxabs):.4f} "
          f"mediana={np.median(maxabs):.4f} max={np.max(maxabs):.4f} "
          f"(ventanas con max|x| = 1 exacto: {np.mean(np.isclose(maxabs, 1)):.1%})")
    print("  fila que contiene el max|x|:", pd.Series(fila_max).value_counts().sort_index().to_dict())
    print("  ventanas con cada fila de estacion presente (std > 0):",
          {i: int(v) for i, v in enumerate(presentes)})
    print(f"  mascaras one-hot (0/1 y suman 1 por muestra): {np.mean(uno_caliente):.1%}")
    print(f"  la unica clase activa en la mascara es la de la carpeta: {np.mean(coincide_clase):.1%}")
    print(f"  duracion del evento en la mascara: min={np.min(dur_ev):.1f} s mediana={np.median(dur_ev):.1f} s "
          f"max={np.max(dur_ev):.1f} s; tocan el borde de la ventana: {np.mean(en_borde):.1%}")

    # Comparacion espectral y de re-filtrado con la traza
    t_lo, t_hi, t_ref = [], [], []
    for fila in range(1, 8):
        x = np.asarray(traza[fila])
        for _ in range(40):
            a = rng.integers(0, x.size - M)
            s = x[a:a + M]
            if (s == 0).mean() > 0.01 or s.std() == 0:
                continue
            f, p = welch(s, fs=FS, nperseg=1024)
            t_lo.append(p[f < 0.8].sum() / p.sum())
            t_hi.append(p[f > 16].sum() / p.sum())
            y = sosfiltfilt(SOS, s)
            t_ref.append(np.corrcoef(s[500:-500], y[500:-500])[0, 1])
    print("\nContenido espectral (fraccion de potencia, Welch 1024 muestras) y prueba de re-filtrado:")
    print("  re-filtrado: correlacion entre la senal y la misma senal pasada otra vez por el")
    print("  pasabanda 1-15 Hz de Ricardo. Cercana a 1 si la senal ya estaba filtrada.")
    print(f"  {'':8}{'<0,8 Hz med':>12}{'p95':>8}{'>16 Hz med':>12}{'p95':>8}{'r_refilt med':>14}{'p5':>8}")
    for nom, lo, hi, rf in (("pool", esp_lo, esp_hi, refilt), ("traza", t_lo, t_hi, t_ref)):
        print(f"  {nom:8}{np.median(lo):12.4f}{np.percentile(lo, 95):8.4f}{np.median(hi):12.4f}"
              f"{np.percentile(hi, 95):8.4f}{np.median(rf):14.4f}{np.percentile(rf, 5):8.4f}")
    return partes


# ----------------------------------------------------------------------------
# 2. Validacion con solucion conocida
# ----------------------------------------------------------------------------
def fragmento_traza(traza, a):
    # Construye una ventana con el formato del pool a partir de la traza:
    # filas 1-8 -> 0-7, mascaras 9-14 -> 8-13, normalizada por max|x| global
    w = np.asarray(traza[1:15, a:a + M], dtype=np.float64).copy()
    w[:8] /= np.abs(w[:8]).max()
    return w


def validar(z, partes, traza, cat, bus, rng):
    print("\n" + "=" * 78)
    print("2. VALIDACION DEL METODO CON SOLUCION CONOCIDA")
    print("=" * 78)
    # Positivos: posiciones aleatorias por tramo, la mitad centradas en eventos
    filas_pos = []
    for n_tr, a, b, _ in TRAMOS:
        ev = cat[(cat.idx_start >= a + M) & (cat.idx_end < b - M)]
        for k in range(N_POSITIVOS // 3):
            if k % 2 == 0 and len(ev):
                e = ev.iloc[rng.integers(len(ev))]
                p = int((e.idx_start + e.idx_end) // 2 - M // 2 + rng.integers(-1000, 1000))
            else:
                p = int(rng.integers(a, b - M))
            filas_pos.append((n_tr, p))
    print(f"Positivos: {len(filas_pos)} fragmentos de {M} muestras recortados de la traza "
          f"(formato del pool, normalizados por max|x|).")
    res = []
    for n_tr, p in filas_pos:
        w = fragmento_traza(traza, p)
        rv = bus.buscar(w)
        fila = dict(tramo=n_tr, idx_real=p, ncc_ventana=rv["ncc"], idx_ventana=rv["idx"],
                    n_est=rv["n_est"])
        # Version re-filtrada: muestra cuanto baja la NCC si el pool tuviera un
        # filtrado distinto del de la traza
        wf = w.copy()
        for i in range(8):
            if wf[i].std() > 0:
                wf[i] = sosfiltfilt(SOS, wf[i])
        fila["ncc_refiltrado"] = bus.buscar(wf)["ncc"]
        seg = segmento_evento(w)
        if seg and seg[1] - seg[0] >= 200:
            re_ = bus.buscar(w, *seg)
            fila.update(ncc_evento=re_["ncc"], idx_evento=re_["idx"], dur_ev_s=(seg[1] - seg[0]) / FS)
        res.append(fila)
    pos = pd.DataFrame(res)
    pos["ok_ventana"] = pos.idx_ventana == pos.idx_real
    if "idx_evento" in pos:
        pos["ok_evento"] = np.where(pos.idx_evento.isna(), np.nan, pos.idx_evento == pos.idx_real)
    print(pos.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"  indice exacto en modo ventana: {int(pos.ok_ventana.sum())}/{len(pos)}; "
          f"NCC min={pos.ncc_ventana.min():.6f}")
    if "ncc_evento" in pos:
        pe = pos.dropna(subset=["ncc_evento"])
        print(f"  indice exacto en modo evento: {int((pe.idx_evento == pe.idx_real).sum())}/{len(pe)}; "
              f"NCC min={pe.ncc_evento.min():.6f}")
    print(f"  NCC tras re-filtrar el fragmento: min={pos.ncc_refiltrado.min():.4f} "
          f"mediana={pos.ncc_refiltrado.median():.4f}")

    # Nulo: ventanas del pool de fechas ajenas a la traza
    ajenas = partes[[(m, d) not in FECHAS_EXCLUIDAS_NULO for m, d in zip(partes.mes, partes.dia)]]
    sel = ajenas.groupby("carpeta").sample(N_NULO // 5, random_state=SEMILLA)
    print(f"\nNulo: {len(sel)} ventanas del pool con mes-dia fuera de {sorted(FECHAS_EXCLUIDAS_NULO)} "
          f"({N_NULO // 5} por clase).")
    t0 = time.time()
    nulo = pd.DataFrame([medir(z, r, bus) for r in sel.itertuples()])
    print(f"  ({time.time() - t0:.0f} s)")
    nulo = anotar(nulo, traza, cat)
    nulo.to_csv(os.path.join(DIR_SALIDA, "superposicion_pool_traza_nulo.csv"), index=False)
    for modo in ("ventana", "evento"):
        v = nulo[f"ncc_{modo}"].dropna()
        v1 = nulo[f"ncc1_{modo}"].dropna()
        print(f"  modo {modo:7}: NCC media entre estaciones: max={v.max():.4f} p99={v.quantile(.99):.4f} "
              f"p95={v.quantile(.95):.4f} mediana={v.median():.4f} (n={len(v)})")
        print(f"  {'':13}NCC mejor estacion sola:     max={v1.max():.4f} p99={v1.quantile(.99):.4f} "
              f"p95={v1.quantile(.95):.4f} mediana={v1.median():.4f}")
    print("  NCC maxima del nulo por clase (modo evento, media entre estaciones):",
          nulo.groupby("clase").ncc_evento.max().round(4).to_dict())
    print("  NCC maxima del nulo por clase (modo ventana, media entre estaciones):",
          nulo.groupby("clase").ncc_ventana.max().round(4).to_dict())
    top = nulo.sort_values("ncc_evento", ascending=False).head(5)
    print("  5 ventanas del nulo con mayor NCC en modo evento:")
    print(top[["archivo", "clase", "dur_ev_s", "n_est", "ncc_evento", "ncc1_evento",
               "ncc_ventana"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    alto = nulo[nulo.ncc1_evento >= 0.95].sort_values("ncc1_evento", ascending=False)
    print(f"  Ventanas del nulo con alguna estacion sola >= 0,95 en modo evento: {len(alto)}")
    if len(alto):
        print(alto[["archivo", "clase", "dur_ev_s", "fila1_evento", "ncc1_evento", "idx_1est",
                    "tramo_1est", "utc_1est", "cat_tipo_1est", "cat_iou_1est", "av_sin_crudo_1est"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nUmbrales declarados: modo ventana {UMBRAL_VENTANA}, modo evento {UMBRAL_EVENTO} "
          f"(NCC media entre estaciones); mejor estacion sola {UMBRAL_1EST}.")
    for modo, u, col_pos in (("ventana", UMBRAL_VENTANA, "ncc_ventana"),
                             ("evento", UMBRAL_EVENTO, "ncc_evento")):
        mx = nulo[f"ncc_{modo}"].max()
        mn = pos[col_pos].min() if col_pos in pos else np.nan
        print(f"  {modo:7}: max del nulo {mx:.4f} < {u} <= min de positivos {mn:.4f}: "
              f"{'SEPARA' if mx < u <= mn else 'NO SEPARA, revisar umbral'}")
    # Para la estacion sola, el nulo solo es valido si sus valores altos no
    # caen sobre los AV sin crudo (si caen ahi, el nulo esta contaminado)
    limpio = nulo[~nulo.av_sin_crudo_1est]
    mx = limpio.ncc1_evento.max()
    print(f"  1 estac.: max del nulo fuera de los AV sin crudo {mx:.4f} < {UMBRAL_1EST} <= 1,0000 "
          f"(positivos): {'SEPARA' if mx < UMBRAL_1EST else 'NO SEPARA, revisar umbral'}; "
          f"ventanas del nulo que caen sobre AV sin crudo: {int(nulo.av_sin_crudo_1est.sum())}")
    return pos, nulo


# ----------------------------------------------------------------------------
# 3. Busqueda
# ----------------------------------------------------------------------------
def medir(z, r, bus):
    w = leer(z, r.archivo)
    fila = dict(archivo=r.archivo, clase=r.carpeta)
    rv = bus.buscar(w)
    if rv is None:
        fila.update(n_est=0)
        return fila
    fila.update(n_est=rv["n_est"], filas=rv["filas"], ncc_ventana=rv["ncc"], idx_ventana=rv["idx"],
                ncc_est_ventana=rv["ncc_por_est"], ncc1_ventana=rv["ncc_1est"],
                idx1_ventana=rv["idx_1est"], fila1_ventana=rv["fila_1est"])
    seg = segmento_evento(w)
    if seg:
        re_ = bus.buscar(w, *seg)
        fila.update(ev_ini=seg[0], ev_fin=seg[1], dur_ev_s=(seg[1] - seg[0]) / FS)
        if re_:
            fila.update(ncc_evento=re_["ncc"], idx_evento=re_["idx"], ncc_est_evento=re_["ncc_por_est"],
                        ncc1_evento=re_["ncc_1est"], idx1_evento=re_["idx_1est"],
                        fila1_evento=re_["fila_1est"])
    return fila


def buscar_conjunto(z, sel, bus, etiqueta):
    print(f"\n--- {etiqueta}: {len(sel):,} ventanas")
    t0 = time.time()
    out = []
    for k, r in enumerate(sel.itertuples(), 1):
        out.append(medir(z, r, bus))
        if k % 200 == 0:
            print(f"    {k:,}/{len(sel):,} ({time.time() - t0:.0f} s)", flush=True)
    print(f"    terminado en {time.time() - t0:.0f} s")
    df = pd.DataFrame(out)
    df["conjunto"] = etiqueta
    return df


# ----------------------------------------------------------------------------
# 4. Reporte
# ----------------------------------------------------------------------------
def iou(a0, a1, b0, b1):
    i = max(0, min(a1, b1) - max(a0, b0))
    u = max(a1, b1) - min(a0, b0)
    return i / u if u > 0 else 0.0


def anotar(df, traza, cat):
    # Para la mejor posicion de cada modo: tramo, hora UTC y evento del
    # catalogo con mayor IoU respecto del evento del pool trasladado a la traza
    # "1est": posicion de la mejor estacion sola en modo evento
    for modo, col in (("ventana", "idx_ventana"), ("evento", "idx_evento"), ("1est", "idx1_evento")):
        tr, hora, ev_cat, ev_tipo, ev_iou, sin_crudo = [], [], [], [], [], []
        for r in df.itertuples():
            idx = getattr(r, col, np.nan)
            if pd.isna(idx) or pd.isna(getattr(r, "ev_ini", np.nan)):
                tr.append(np.nan); hora.append(""); ev_cat.append(np.nan)
                ev_tipo.append(""); ev_iou.append(np.nan); sin_crudo.append(False)
                continue
            idx = int(idx)
            a, b = idx + int(r.ev_ini), idx + int(r.ev_fin)
            tr.append(tramo_de(max(idx, 0)))
            hora.append(utc(traza[0, min(max(idx, 0), traza.shape[1] - 1)]))
            ious = [(iou(a, b, e.idx_start, e.idx_end), e.idx_start, e.event_type)
                    for e in cat.itertuples()]
            best = max(ious)
            ev_iou.append(best[0]); ev_cat.append(best[1] if best[0] > 0 else np.nan)
            ev_tipo.append(best[2] if best[0] > 0 else "")
            sin_crudo.append(best[0] > 0 and best[1] in AV_SIN_CRUDO)
        df[f"tramo_{modo}"] = tr
        df[f"utc_{modo}"] = hora
        df[f"cat_idx_{modo}"] = ev_cat
        df[f"cat_tipo_{modo}"] = ev_tipo
        df[f"cat_iou_{modo}"] = ev_iou
        df[f"av_sin_crudo_{modo}"] = sin_crudo
    df["idx_1est"] = df.get("idx1_evento")
    return df


def reportar(df, traza, bus):
    print("\n" + "=" * 78)
    print("4. RESULTADOS")
    print("=" * 78)
    for c in df.conjunto.unique():
        d = df[df.conjunto == c]
        print(f"\n{c} (n={len(d):,}; sin estaciones utilizables: {int((d.n_est == 0).sum())})")
        for modo, u in (("ventana", UMBRAL_VENTANA), ("evento", UMBRAL_EVENTO)):
            v = d[f"ncc_{modo}"].dropna()
            print(f"  modo {modo:7}: NCC max={v.max():.4f} p99={v.quantile(.99):.4f} "
                  f"mediana={v.median():.4f}; >= {u}: {int((v >= u).sum())}")
        print("  NCC maxima por clase (ventana / evento):",
              {k: (round(g.ncc_ventana.max(), 3), round(g.ncc_evento.max(), 3))
               for k, g in d.groupby("clase")})

    hit = df[(df.ncc_ventana >= UMBRAL_VENTANA) | (df.ncc_evento >= UMBRAL_EVENTO)]
    print(f"\nCoincidencias por la media entre estaciones (cualquier modo): {len(hit)}")
    if len(hit):
        cols = ["conjunto", "archivo", "clase", "n_est", "ncc_ventana", "idx_ventana", "tramo_ventana",
                "utc_ventana", "cat_tipo_ventana", "cat_iou_ventana", "ncc_evento", "idx_evento",
                "tramo_evento", "utc_evento", "cat_idx_evento", "cat_tipo_evento", "cat_iou_evento",
                "av_sin_crudo_evento"]
        print(hit[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    h1 = df[(df.ncc1_evento >= UMBRAL_1EST) & ~df.archivo.isin(hit.archivo)]
    print(f"\nCoincidencias adicionales solo por estacion sola (NCC >= {UMBRAL_1EST} en modo evento): "
          f"{len(h1)}")
    if len(h1):
        cols = ["conjunto", "archivo", "clase", "dur_ev_s", "filas", "fila1_evento", "ncc1_evento",
                "ncc_est_evento", "ncc_evento", "idx_1est", "tramo_1est", "utc_1est", "cat_idx_1est",
                "cat_tipo_1est", "cat_iou_1est", "av_sin_crudo_1est"]
        print(h1.sort_values("idx_1est")[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Siempre: las 15 mayores NCC en cada modo, aunque no superen el umbral
    for modo in ("ventana", "evento"):
        print(f"\n15 mayores NCC en modo {modo} (sobre o bajo el umbral):")
        cols = ["conjunto", "archivo", "clase", "n_est", "dur_ev_s", f"ncc_{modo}", f"ncc_est_{modo}",
                f"ncc1_{modo}", f"idx_{modo}", f"tramo_{modo}", f"utc_{modo}", f"cat_tipo_{modo}",
                f"cat_iou_{modo}", f"av_sin_crudo_{modo}"]
        print(df.sort_values(f"ncc_{modo}", ascending=False).head(15)[cols]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Los 17 AV sin crudo: mejor ventana del pool que cae sobre cada uno
    print("\nLos 17 AV del tramo 1 sin contraparte en el crudo de FRE: ventana del pool con mayor")
    print("NCC de estacion sola (modo evento) cuya posicion se solapa con el AV (IoU > 0):")
    filas_av = []
    for e in sorted(AV_SIN_CRUDO):
        d = df[df.cat_idx_1est == e]
        if len(d):
            r = d.sort_values("ncc1_evento", ascending=False).iloc[0]
            filas_av.append(dict(idx_start=e, n_ventanas_solapan=len(d), mejor_archivo=r.archivo,
                                 clase_pool=r.clase, fila_pool=r.fila1_evento, ncc1=r.ncc1_evento,
                                 ncc_est=r.ncc_est_evento, ncc_media=r.ncc_evento, iou=r.cat_iou_1est,
                                 supera=r.ncc1_evento >= UMBRAL_1EST))
        else:
            filas_av.append(dict(idx_start=e, n_ventanas_solapan=0))
    print(pd.DataFrame(filas_av).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--todo", action="store_true", help="buscar tambien el resto del pool")
    args = ap.parse_args()
    rng = np.random.default_rng(SEMILLA)

    z = zipfile.ZipFile(ZIP_POOL)
    nombres = sorted(n for n in z.namelist() if not n.endswith("/"))
    traza = np.load(NPY_TRAZA, mmap_mode="r")
    cat = pd.read_csv(CSV_CATALOGO)
    print(f"Traza: forma={traza.shape} dtype={traza.dtype}; catalogo: {len(cat)} eventos")

    partes = inspeccionar(z, nombres, traza, rng)

    t0 = time.time()
    bus = Buscador(traza)
    print(f"\nFilas de la traza utilizables para la busqueda: {sorted(bus.X)} "
          f"(precalculo {time.time() - t0:.0f} s)")

    validar(z, partes, traza, cat, bus, rng)

    print("\n" + "=" * 78)
    print("3. BUSQUEDA DEL POOL EN LA TRAZA")
    print("=" * 78)
    print("Fila i del pool <-> fila i + 1 de la traza. NCC = media de las NCC por estacion en el")
    print("mismo desfase (estaciones presentes en la ventana del pool y utilizables en la traza).")
    cand = partes[partes.apply(es_candidato, axis=1)]
    av = partes[(partes.carpeta == "AV") & ~partes.archivo.isin(cand.archivo)]
    conjuntos = [(cand, "31 candidatos"), (av, "AV restantes")]
    if args.todo:
        resto = partes[~partes.archivo.isin(cand.archivo) & ~partes.archivo.isin(av.archivo)]
        conjuntos.append((resto, "resto del pool"))
    df = pd.concat([buscar_conjunto(z, s, bus, e) for s, e in conjuntos], ignore_index=True)
    df = anotar(df, traza, cat)
    os.makedirs(DIR_SALIDA, exist_ok=True)
    ruta = os.path.join(DIR_SALIDA, "superposicion_pool_traza.csv")
    df.to_csv(ruta, index=False)
    print(f"\nResultados por ventana guardados en {ruta}")
    reportar(df, traza, bus)


if __name__ == "__main__":
    sys.exit(main())
