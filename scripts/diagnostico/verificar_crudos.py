# Verifica los archivos crudos que envio Ricardo (datos/crudos/FRE_*.npy)
# contra la traza continua de Zenodo.
#
# Pasos:
#   1. Inspeccion sin supuestos: estructura, forma, dtype, rangos, ceros,
#      vectores de tiempo, huecos y componentes disponibles.
#   2. Validacion del filtro con un caso sintetico de solucion conocida
#      (senoides de 0,5, 5 y 30 Hz: tras el filtro debe quedar solo 5 Hz).
#   3. Filtrado de cada componente cruda con el filtro de Ricardo y
#      emparejamiento contra las filas 1 a 8 del tramo de Zenodo por
#      correlacion cruzada: fila que mejor corresponde, desfase, correlacion
#      maxima y error relativo tras alinear con factor de escala.
#   4. Desfase local por ventanas de 300 s, para ver si la alineacion se
#      mantiene a lo largo del tramo o cambia despues de los huecos (solo
#      para las componentes con correspondencia global, |r| >= 0,9).
#   5. Coincidencia evento por evento del catalogo de Zenodo entre la fila
#      emparejada y el crudo filtrado, para localizar donde difieren.
#
# El filtro NO se ajusta para forzar la coincidencia: es el que envio Ricardo.
#
# Nota de seguridad: los .npy crudos son objetos pickle (dict). Se revisaron
# sus opcodes antes de cargarlos: solo referencian clases de numpy.
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy.signal import butter, sosfiltfilt, correlate

DIR_CRUDOS = os.path.expanduser("~/tesis/datos/crudos")
NPY_ZENODO = os.path.expanduser("~/tesis/datos/zenodo/NVCh_10h_continuous_trace.npy")
CSV_CATALOGO = os.path.expanduser("~/tesis/datos/zenodo/NVCh_10h_continuous_trace_reference.csv")

FS = 100.0
SOS = butter(5, [1.0, 15.0], btype="bandpass", fs=FS, output="sos")

# Correspondencia indicada por Patricio: archivo crudo -> bloque de Zenodo
TRAMOS = [
    ("FRE_1518149218_1518170818.npy", 1, 0, 2_160_000),
    ("FRE_1578964018_1578974818.npy", 2, 2_160_000, 3_240_001),
    ("FRE_1484028418_1484032660.npy", 3, 3_240_001, 3_664_193),
]
COMPONENTES = ("Z", "N", "E")
FILAS_ZENODO = range(1, 9)

LAG_MAX = 3000          # busqueda de desfase global: +-30 s
MARGEN_HUECO = 2000     # muestras excluidas a cada lado de un hueco (transitorio del filtro)
VENTANA_LOCAL = 30000   # ventanas de 300 s para el desfase local
LAG_MAX_LOCAL = 500     # +-5 s alrededor del desfase global
R_MIN_CORRESPONDE = 0.9 # |r| global minimo para considerar que hay correspondencia
ERR_COINCIDE = 0.05     # err_rel maximo para declarar que un evento coincide
ERR_NO_COINCIDE = 0.5   # err_rel minimo para declarar que un evento no coincide


def utc(t):
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def huecos(t):
    # Indices donde el paso entre muestras no es 0,01 s
    dt = np.diff(t)
    return np.where(np.abs(dt - 1 / FS) > 1e-4)[0], dt


# ----------------------------------------------------------------------------
# 1. Inspeccion
# ----------------------------------------------------------------------------
def inspeccionar(nombre, d):
    print(f"\n### {nombre}")
    print(f"tipo del objeto cargado: {type(d).__name__}; claves: {list(d.keys())}")
    for k, v in d.items():
        if not isinstance(v, (np.ndarray, dict)):
            extra = f"  ({utc(v)} UTC)" if k.startswith("unix") else ""
            print(f"  {k}: {v!r}{extra}")
    print(f"  fs: {d['fs']}")
    nominal = int(round((d["unix_fin"] - d["unix_inicio"]) * FS))
    print(f"  muestras nominales (unix_fin - unix_inicio) * fs = {nominal:,}")
    for c in COMPONENTES:
        x = d[c]
        t = d["tiempos_unix"][c]
        idx, dt = huecos(t)
        faltan = int(round(np.sum(dt[idx] - 1 / FS) * FS))
        xf = x.astype(float)
        print(f"  [{c}] forma={x.shape} dtype={x.dtype} min={x.min():,} max={x.max():,} "
              f"media={xf.mean():.1f} std={xf.std():.1f} ceros={int((x == 0).sum()):,} "
              f"identicamente_cero={bool(np.all(x == 0))}")
        print(f"       tiempo: n={len(t):,} (igual a datos: {len(t) == len(x)}) "
              f"{utc(t[0])} -> {utc(t[-1])} UTC, monotono={bool(np.all(dt > 0))}, "
              f"paso mediano={np.median(dt):.6f} s")
        print(f"       huecos: {len(idx)}, muestras faltantes={faltan:,} "
              f"({faltan / FS:.0f} s), n + faltantes = {len(t) + faltan:,}")
        if len(idx):
            print("       detalle (muestra cruda, hora UTC, duracion del hueco):")
            for i in idx:
                print(f"         {i:>9,}  {utc(t[i])}  {dt[i] - 1 / FS:6.2f} s")


# ----------------------------------------------------------------------------
# 2. Validacion sintetica del filtro
# ----------------------------------------------------------------------------
def amplitud(x, f, t):
    # Amplitud de la componente de frecuencia f por proyeccion en seno y coseno
    return 2 * np.abs(np.mean(x * np.exp(-2j * np.pi * f * t)))


def validar_filtro():
    print("\n" + "=" * 78)
    print("2. VALIDACION SINTETICA DEL FILTRO")
    print("=" * 78)
    t = np.arange(0, 60, 1 / FS)
    x = np.sin(2 * np.pi * 0.5 * t) + np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 30 * t)
    y = sosfiltfilt(SOS, x)
    # Se evalua el tramo central de 40 s para excluir los bordes
    c = (t >= 10) & (t < 50)
    ok = True
    for f, esperado in ((0.5, 0.0), (5.0, 1.0), (30.0, 0.0)):
        a = amplitud(y[c], f, t[c])
        cumple = abs(a - esperado) < 0.01
        ok &= cumple
        print(f"  {f:>4} Hz: amplitud de entrada 1,000 -> salida {a:.5f} "
              f"(esperado {esperado:.0f}, tolerancia 0,01) {'OK' if cumple else 'FALLA'}")
    resto = y[c] - np.sin(2 * np.pi * 5 * t[c])
    rel = np.sqrt(np.mean(resto ** 2)) / np.sqrt(np.mean(np.sin(2 * np.pi * 5 * t[c]) ** 2))
    print(f"  RMS(salida - senoide 5 Hz) / RMS(senoide 5 Hz) = {rel:.5f}")
    ok &= rel < 0.01
    print(f"  resultado: {'el filtro se comporta como se espera' if ok else 'EL FILTRO NO PASA LA PRUEBA'}")
    return ok


# ----------------------------------------------------------------------------
# 3. Emparejamiento contra Zenodo
# ----------------------------------------------------------------------------
def a_grilla(x, t, t0_zen, n_zen):
    # Ubica cada muestra cruda en la grilla de Zenodo segun su tiempo Unix.
    # Los huecos se rellenan por interpolacion lineal (solo para que el filtro
    # no vea escalones) y se marcan como no validos, con un margen a cada lado.
    k = np.round((t - t0_zen) * FS).astype(np.int64)
    dentro = (k >= 0) & (k < n_zen)
    k, xv = k[dentro], x[dentro].astype(float)
    y = np.interp(np.arange(n_zen), k, xv)
    presente = np.zeros(n_zen, bool)
    presente[k] = True
    # Margen alrededor de cada hueco y de los extremos sin datos
    faltante = ~presente
    if faltante.any():
        ancho = np.convolve(faltante.astype(float), np.ones(2 * MARGEN_HUECO + 1), "same") > 0
    else:
        ancho = faltante
    return y, ~ancho, int(faltante.sum()), int((~dentro).sum())


def xcorr_lags(a, b, lag_max):
    # c[l] = sum a[n] * b[n + l] para l en [-lag_max, lag_max], via FFT.
    # Desfase positivo: la senal de Zenodo (b) va atrasada respecto del crudo (a).
    full = correlate(b, a, mode="full", method="fft")
    centro = len(a) - 1
    lags = np.arange(-lag_max, lag_max + 1)
    return lags, full[centro + lags]


def alinear(x, z, lag):
    # Devuelve los pares de muestras (x[n], z[n + lag]) en el solapamiento
    if lag >= 0:
        return x[: len(x) - lag], z[lag:]
    return x[-lag:], z[: len(z) + lag]


def metricas(x, z, vx, vz, lag):
    xa, za = alinear(x, z, lag)
    va, vb = alinear(vx, vz, lag)
    m = va & vb
    xa, za = xa[m], za[m]
    r = np.corrcoef(xa, za)[0, 1]
    # Factor de escala por minimos cuadrados (admite signo negativo)
    esc = np.dot(xa, za) / np.dot(xa, xa)
    err = np.sqrt(np.mean((za - esc * xa) ** 2)) / np.sqrt(np.mean(za ** 2))
    return r, esc, err, int(m.sum())


def emparejar(d, zen, n_tramo, ini, fin):
    print(f"\n### Tramo {n_tramo}: Zenodo [{ini:,}, {fin:,})  <->  crudo {d['estacion']} "
          f"{utc(d['unix_inicio'])[:19]} UTC")
    t_zen = np.asarray(zen[0, ini:fin])
    n = fin - ini
    print(f"  Zenodo: t0={t_zen[0]:.4f} ({utc(t_zen[0])}), n={n:,}")
    filas = {}
    for r in FILAS_ZENODO:
        zr = np.asarray(zen[r, ini:fin], dtype=float)
        vz = zr != 0
        filas[r] = (zr, vz)
    vacias = [r for r in FILAS_ZENODO if not filas[r][1].any()]
    if vacias:
        print(f"  filas de Zenodo identicamente cero en este tramo (se omiten): {vacias}")

    mejores = {}
    print(f"\n  Correlacion maxima |r| (desfase +-{LAG_MAX / FS:.0f} s, normalizada por "
          f"energia global) de cada componente cruda contra cada fila:")
    print("        " + "".join(f"  fila{r:<3}" for r in FILAS_ZENODO))
    for c in COMPONENTES:
        y, vx, n_falt, n_fuera = a_grilla(d[c], d["tiempos_unix"][c], t_zen[0], n)
        xf = sosfiltfilt(SOS, y)
        xm = np.where(vx, xf, 0.0)
        celdas, candidatos = [], []
        for r in FILAS_ZENODO:
            zr, vz = filas[r]
            if r in vacias:
                celdas.append("     --  ")
                continue
            zm = np.where(vz, zr, 0.0)
            lags, cc = xcorr_lags(xm, zm, LAG_MAX)
            cc = cc / np.sqrt(np.dot(xm, xm) * np.dot(zm, zm))
            i = int(np.argmax(np.abs(cc)))
            celdas.append(f"  {abs(cc[i]):7.4f}")
            candidatos.append((abs(cc[i]), r, int(lags[i])))
        print(f"    {c}   " + "".join(celdas))
        _, r_best, lag_best = max(candidatos)
        zr, vz = filas[r_best]
        rr, esc, err, n_val = metricas(xf, zr, vx, vz, lag_best)
        en_borde = abs(lag_best) == LAG_MAX
        mejores[c] = dict(fila=r_best, lag=lag_best, r=rr, esc=esc, err=err, n_val=n_val,
                          n_falt=n_falt, n_fuera=n_fuera, borde=en_borde, xf=xf, vx=vx)

    print("\n  Mejor correspondencia (metricas exactas en las muestras validas del solapamiento):")
    print("    comp  fila  desfase[muestras]  r_Pearson  escala     err_rel  muestras_validas")
    for c in COMPONENTES:
        m = mejores[c]
        aviso = "  <- desfase en el borde de la busqueda" if m["borde"] else ""
        print(f"    {c:>4}  {m['fila']:>4}  {m['lag']:>17}  {m['r']:9.5f}  {m['esc']:9.4g}  "
              f"{m['err']:8.5f}  {m['n_val']:,}{aviso}")
    print("    (muestras sin dato crudo en la grilla de Zenodo: "
          + ", ".join(f"{c}={mejores[c]['n_falt']:,}" for c in COMPONENTES) + ")")

    # Desfase local por ventanas, contra la fila elegida
    print(f"\n  Desfase local por ventanas de {VENTANA_LOCAL / FS:.0f} s "
          f"(busqueda +-{LAG_MAX_LOCAL / FS:.0f} s alrededor del global; "
          f"solo ventanas con >= 50 % de muestras validas):")
    for c in COMPONENTES:
        m = mejores[c]
        if abs(m["r"]) < R_MIN_CORRESPONDE:
            print(f"    {c}: se omite (|r| global = {abs(m['r']):.3f} < {R_MIN_CORRESPONDE}: "
                  f"sin correspondencia con ninguna fila)")
            continue
        zr, vz = filas[m["fila"]]
        res = []
        for a in range(0, n - VENTANA_LOCAL + 1, VENTANA_LOCAL):
            b = a + VENTANA_LOCAL
            vxa = m["vx"][a:b]
            # Se exige >= 50 % de muestras validas en el crudo y en Zenodo
            if a + m["lag"] < 0 or b + m["lag"] > n:
                continue
            if (vxa & vz[a + m["lag"]:b + m["lag"]]).mean() < 0.5:
                continue
            # Ventana del crudo fija; ventana de Zenodo ampliada en +-LAG_MAX_LOCAL
            za0 = max(0, a + m["lag"] - LAG_MAX_LOCAL)
            zb0 = min(n, b + m["lag"] + LAG_MAX_LOCAL)
            xw = np.where(vxa, m["xf"][a:b], 0.0)
            zw = np.where(vz[za0:zb0], zr[za0:zb0], 0.0)
            if not zw.any():
                continue
            full = correlate(zw, xw, mode="valid", method="fft")
            lags = np.arange(len(full)) + za0 - a
            i = int(np.argmax(np.abs(full)))
            lag_loc = int(lags[i])
            # Correlacion local exacta en el desfase encontrado
            xs = m["xf"][a:b]
            zs = zr[a + lag_loc:b + lag_loc] if 0 <= a + lag_loc and b + lag_loc <= n else None
            if zs is None:
                continue
            ok = vxa & vz[a + lag_loc:b + lag_loc]
            if ok.sum() < 100:
                continue
            r_loc = np.corrcoef(xs[ok], zs[ok])[0, 1]
            res.append((a, lag_loc, r_loc))
        lags_v = np.array([x[1] for x in res])
        r_v = np.array([x[2] for x in res])
        valores, cuentas = np.unique(lags_v, return_counts=True)
        print(f"    {c}: {len(res)} ventanas; desfases encontrados "
              + ", ".join(f"{v}({k})" for v, k in zip(valores, cuentas))
              + f"; r local min={r_v.min():.4f} mediana={np.median(r_v):.4f}")
        distintos = [x for x in res if x[1] != m["lag"]]
        for a, lg, rl in distintos:
            print(f"       ventana desde muestra {a:>9,} ({utc(t_zen[a])[:19]}): desfase {lg}, r={rl:.4f}")
    return mejores, filas


# ----------------------------------------------------------------------------
# 5. Coincidencia por evento del catalogo
# ----------------------------------------------------------------------------
def por_evento(cat, n_tramo, ini, fin, mejores, filas):
    # Compara, dentro de cada evento del catalogo, la fila de Zenodo emparejada
    # con la componente cruda filtrada, en el desfase global medido.
    res = []
    for c in COMPONENTES:
        m = mejores[c]
        if abs(m["r"]) < R_MIN_CORRESPONDE:
            continue
        zr, vz = filas[m["fila"]]
        n = fin - ini
        ev = cat[(cat.idx_start >= ini) & (cat.idx_start < fin)]
        for _, e in ev.iterrows():
            a, b = int(e.idx_start) - ini, min(int(e.idx_end) - ini, n)
            if a + m["lag"] < 0 or b + m["lag"] > n:
                continue
            xs = m["xf"][a:b]
            zs = zr[a + m["lag"]:b + m["lag"]]
            ok = vz[a + m["lag"]:b + m["lag"]]
            sin_crudo = 1 - m["vx"][a:b].mean()
            if ok.sum() < 50:
                estado, r, err = "zenodo_ceros", np.nan, np.nan
            else:
                r = np.corrcoef(xs[ok], zs[ok])[0, 1]
                err = np.sqrt(np.mean((zs[ok] - xs[ok]) ** 2)) / np.sqrt(np.mean(zs[ok] ** 2))
                estado = ("coincide" if err < ERR_COINCIDE else
                          "no_coincide" if err > ERR_NO_COINCIDE else "parcial")
            res.append(dict(tramo=n_tramo, comp=c, fila=m["fila"], idx_start=int(e.idx_start),
                            tipo=e.event_type, dur_s=(b - a) / FS, r=r, err_rel=err,
                            frac_cerca_hueco=sin_crudo, estado=estado))
    return res


def main():
    print("=" * 78)
    print("1. INSPECCION DE LOS ARCHIVOS CRUDOS")
    print("=" * 78)
    zen = np.load(NPY_ZENODO, mmap_mode="r")
    print(f"Zenodo: forma={zen.shape} dtype={zen.dtype}")
    crudos = {}
    for nombre, *_ in TRAMOS:
        ruta = os.path.join(DIR_CRUDOS, nombre)
        arr = np.load(ruta, allow_pickle=True)
        print(f"\n{nombre}: arreglo externo forma={arr.shape} dtype={arr.dtype} "
              f"({os.path.getsize(ruta):,} bytes)")
        crudos[nombre] = arr.item()
        inspeccionar(nombre, crudos[nombre])

    if not validar_filtro():
        print("Se detiene: el filtro no pasa la validacion sintetica.")
        return

    print("\n" + "=" * 78)
    print("3. EMPAREJAMIENTO DE CADA COMPONENTE FILTRADA CONTRA ZENODO")
    print("=" * 78)
    print(f"Convenciones: desfase > 0 significa que la muestra n del crudo corresponde a la "
          f"muestra n + desfase de Zenodo (ambos en la grilla de Zenodo).\n"
          f"Se excluyen de las metricas las muestras de Zenodo iguales a 0 y +-{MARGEN_HUECO / FS:.0f} s "
          f"alrededor de cada hueco del crudo.\n"
          f"err_rel = RMS(zenodo - escala * crudo_filtrado) / RMS(zenodo).")
    cat = pd.read_csv(CSV_CATALOGO)
    eventos = []
    for nombre, n_tramo, ini, fin in TRAMOS:
        mejores, filas = emparejar(crudos[nombre], zen, n_tramo, ini, fin)
        eventos += por_evento(cat, n_tramo, ini, fin, mejores, filas)

    print("\n" + "=" * 78)
    print("5. COINCIDENCIA POR EVENTO DEL CATALOGO (fila emparejada vs crudo filtrado)")
    print("=" * 78)
    print(f"coincide: err_rel < {ERR_COINCIDE}; no_coincide: err_rel > {ERR_NO_COINCIDE}; "
          f"parcial: entre ambos; zenodo_ceros: < 50 muestras de Zenodo distintas de 0.\n"
          f"frac_cerca_hueco: fraccion del evento a menos de {MARGEN_HUECO / FS:.0f} s de un hueco del crudo.")
    df = pd.DataFrame(eventos)
    print(pd.crosstab([df.tramo, df.comp, df.tipo], df.estado, margins=True).to_string())
    distintos = df[df.estado != "coincide"]
    print(f"\nEventos que no coinciden ({len(distintos)}):")
    print(distintos.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
