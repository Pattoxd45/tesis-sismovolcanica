#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
revisar_falsos_positivos.py
===========================

Pregunta una sola cosa, pero decisiva: los "falsos positivos" de los
detectores, son realmente falsos?

MOTIVO: en las figuras de validacion aparece energia sismica clara fuera
de los intervalos catalogados (por ejemplo un evento nitido justo antes
del tremor de referencia). Si el catalogo de 205 eventos no anota TODO lo
que ocurre en las 10 h, entonces parte de los falsos positivos son
eventos reales sin etiquetar, y la precision de los TRES detectores esta
subestimada de forma sistematica.

Esto no se puede decidir a ojo. La prueba es estadistica: si los falsos
positivos fueran ruido, su relacion senal/ruido deberia parecerse a la de
tramos de fondo tomados al azar. Si en cambio se parece a la de los
eventos catalogados, son eventos.

Metodo:
  1. Reproduce las detecciones de STA/LTA en su punto de operacion optimo.
  2. Separa verdaderos positivos, falsos positivos y tramos de fondo.
  3. Calcula la SNR de cada grupo (pico sobre RMS del fondo local).
  4. Compara las tres distribuciones y cuenta cuantos FP superan el
     percentil 10 de SNR de los eventos reales.

Uso:
    python revisar_falsos_positivos.py
    python revisar_falsos_positivos.py --canal 2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluar_detectores import (FS, CANALES, CORTES, GUARDA_S, NPY, CSV,
                                SALIDA, CLASES, Intervalo, segmentos,
                                fusionar_canal, coincidencia, emparejar,
                                STALTA)

IOU_MIN = 0.30
MIN_EST = 2


def snr(x_ev, x_fondo):
    """Pico del evento sobre el RMS del fondo local, en decibeles."""
    rms = float(np.sqrt(np.mean(x_fondo ** 2)))
    pico = float(np.max(np.abs(x_ev)))
    # un intervalo que cae en un hueco de datos tiene pico 0 y no admite
    # logaritmo: se descarta en vez de propagar -inf a las estadisticas
    if rms <= 0 or pico <= 0:
        return float("nan")
    return 20.0 * np.log10(pico / rms)


def snr_intervalo(a, canal, ini, fin, margen_s=60.0):
    """SNR de un intervalo respecto al fondo inmediatamente anterior."""
    m = int(margen_s * FS)
    f0 = max(0, ini - m)
    if ini - f0 < int(5 * FS):
        return float("nan")
    x_ev = np.asarray(a[canal, ini:fin], dtype=np.float64)
    x_bg = np.asarray(a[canal, f0:ini], dtype=np.float64)
    if x_ev.size == 0 or x_bg.size == 0 or np.allclose(x_bg, 0):
        return float("nan")
    return snr(x_ev, x_bg)


def describir(nombre, vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if v.size == 0:
        print(f"  {nombre:<26} sin datos validos")
        return v
    print(f"  {nombre:<26} n={v.size:>4}  "
          f"p10={np.percentile(v, 10):>6.1f}  "
          f"mediana={np.median(v):>6.1f}  "
          f"p90={np.percentile(v, 90):>6.1f}  dB")
    return v


def figuras_fp(a, canal, fp, catalogo, snrs, n_fig, destino_dir):
    """
    Grafica los falsos positivos con mayor SNR, para inspeccion visual.

    La estadistica dice que contienen senal; la vista dice de QUE tipo.
    Un sismo tectonico regional, ruido antropico y un evento volcanico
    pequeno tienen aspectos distintos, y esa distincion no la resuelve
    ningun numero de SNR.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    os.makedirs(destino_dir, exist_ok=True)
    orden = np.argsort([-s if np.isfinite(s) else 0 for s in snrs])
    inicios = np.array([ev.ini for ev in catalogo])

    generadas = []
    for rank, idx in enumerate(orden[:n_fig], start=1):
        d = fp[idx]
        pad = max(int(30 * FS), d.dur)
        ini, fin = max(0, d.ini - pad), min(a.shape[-1], d.fin + pad)
        x = np.asarray(a[canal, ini:fin], dtype=np.float64)
        if x.size < 512 or np.allclose(x, 0):
            continue

        j = int(np.argmin(np.abs(inicios - d.ini)))
        dist = (inicios[j] - d.ini) / FS

        fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1]})
        axes[0].plot(np.arange(len(x)) / FS, x, color="black", lw=0.6)
        axes[0].axvspan((d.ini - ini) / FS, (d.fin - ini) / FS,
                        color="#d62728", alpha=0.22,
                        label=f"deteccion no catalogada ({d.dur / FS:.1f} s)")
        axes[0].set_ylabel("amplitud [cuentas]")
        axes[0].set_title(f"FP #{rank} | SNR {snrs[idx]:.1f} dB | "
                          f"evento catalogado mas cercano: "
                          f"{catalogo[j].clase} a {dist:+.0f} s")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(alpha=0.3, ls="--")

        f, tt, Sxx = spectrogram(x, fs=FS, nperseg=256, noverlap=128)
        axes[1].pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-12),
                           shading="gouraud", cmap="viridis")
        axes[1].set_ylabel("frecuencia [Hz]")
        axes[1].set_xlabel("tiempo [s]")

        ruta = os.path.join(destino_dir, f"fp_{rank:02d}_snr{snrs[idx]:.0f}.png")
        fig.tight_layout()
        fig.savefig(ruta, dpi=140)
        plt.close(fig)
        generadas.append(ruta)

    print(f"\n{len(generadas)} figuras de falsos positivos en {destino_dir}")
    print("  Que mirar: contenido espectral y forma de la envolvente.")
    print("  Alta frecuencia y arribo impulsivo -> probable sismo tectonico.")
    print("  Baja frecuencia y arranque emergente -> probable evento")
    print("  volcanico pequeno no anotado por el analista.")
    return generadas


def clasificar_fp(fp, catalogo, margen_s=60.0):
    """
    Separa los falsos positivos por CAUSA. Tratarlos como una sola
    categoria oculta que se producen por mecanismos distintos:

      fragmento : se solapa con un evento catalogado. El detector SI vio
                  el evento pero lo recorto tan corto que el IoU quedo
                  bajo el umbral. Cuenta doble: falso negativo mas falso
                  positivo. Se corrige mejorando la delimitacion.
      adyacente : sin solape, pero a menos de `margen_s` de un evento.
                  Precursor, replica o coda separada del intervalo.
      novedoso  : aislado. Este si es candidato a evento no catalogado.
    """
    cat = sorted(catalogo, key=lambda x: x.ini)
    inis = np.array([c.ini for c in cat])
    fins = np.array([c.fin for c in cat])
    m = int(margen_s * FS)

    out = []
    for d in fp:
        solapa = np.maximum(0, np.minimum(d.fin, fins) -
                            np.maximum(d.ini, inis))
        if solapa.max() > 0:
            j = int(np.argmax(solapa))
            frac = solapa[j] / max(1, cat[j].dur)
            out.append(("fragmento", cat[j].clase, float(frac)))
        else:
            dist = np.minimum(np.abs(inis - d.fin), np.abs(fins - d.ini))
            j = int(np.argmin(dist))
            if dist[j] <= m:
                out.append(("adyacente", cat[j].clase,
                            float(dist[j]) / FS))
            else:
                out.append(("novedoso", cat[j].clase, float(dist[j]) / FS))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canal", type=int, default=1)
    ap.add_argument("--canales", nargs="+", type=int, default=CANALES)
    ap.add_argument("--figuras", type=int, default=0,
                    help="genera N figuras de los FP con mayor SNR")
    args = ap.parse_args()

    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    from obspy.signal.filter import bandpass

    a = np.load(NPY, mmap_mode="r")
    df = pd.read_csv(CSV)
    n = a.shape[-1]

    valida = np.ones(n, dtype=bool)
    g = int(GUARDA_S * FS)
    for c in CORTES:
        valida[max(0, c - g):min(n, c + g)] = False
    catalogo = [Intervalo(int(r.idx_start), int(r.idx_end), r.event_type)
                for r in df.itertuples()]
    catalogo = [e for e in catalogo if valida[e.ini] and valida[e.fin - 1]]

    # --- detecciones de STA/LTA en su punto optimo ------------------
    print("Ejecutando STA/LTA en el punto de operacion optimo ...")
    por_canal = {}
    for ch in args.canales:
        trigs = []
        for ini, fin in segmentos(n):
            x = np.asarray(a[ch, ini:fin], dtype=np.float64)
            if np.allclose(x, 0):
                continue
            xf = bandpass(x, STALTA["banda"][0], STALTA["banda"][1], df=FS,
                          corners=4, zerophase=True)
            cft = classic_sta_lta(xf, int(STALTA["sta_s"] * FS),
                                  int(STALTA["lta_s"] * FS))
            cft = np.nan_to_num(cft, nan=0.0, posinf=0.0, neginf=0.0)
            for t0, t1 in trigger_onset(cft, STALTA["thr_on"],
                                        STALTA["thr_off"]):
                trigs.append((ini + int(t0), ini + int(t1)))
        por_canal[ch] = {"intervalos": fusionar_canal(trigs, int(5.0 * FS))}
        print(f"  canal {ch}: {len(por_canal[ch]['intervalos'])} intervalos")

    det = coincidencia(por_canal, valida, MIN_EST)
    pares, perdidos, fp = emparejar(catalogo, det, IOU_MIN)
    print(f"\nDetecciones: {len(det)} | TP: {len(pares)} | FP: {len(fp)}")

    # --- control de duraciones patologicas -------------------------
    durs = np.array([d.dur / FS for d in det])
    print(f"\nDuracion de las detecciones [s]: mediana {np.median(durs):.1f} | "
          f"p90 {np.percentile(durs, 90):.1f} | max {durs.max():.1f}")
    largas = int((durs > 300).sum())
    if largas:
        print(f"  ATENCION: {largas} detecciones superan 300 s. El evento mas "
              f"largo del catalogo dura 241 s.")
        print("  Detecciones muy largas absorben eventos reales y los "
              "convierten en falsos negativos.")

    # --- de donde vienen los falsos positivos -----------------------
    clases_fp = clasificar_fp(fp, catalogo)
    print("\n" + "=" * 70)
    print("  ORIGEN DE LOS FALSOS POSITIVOS")
    print("=" * 70)
    for tipo in ["fragmento", "adyacente", "novedoso"]:
        sub = [c for c in clases_fp if c[0] == tipo]
        if not sub:
            continue
        print(f"  {tipo:<12} {len(sub):>4}  ({100 * len(sub) / len(fp):.1f} %)")
        por_clase = {}
        for _, cl, _ in sub:
            por_clase[cl] = por_clase.get(cl, 0) + 1
        print(f"    {'':<10}por clase del evento asociado: "
              + ", ".join(f"{k}:{v}" for k, v in
                          sorted(por_clase.items(), key=lambda kv: -kv[1])))
    n_frag = sum(1 for c in clases_fp if c[0] == "fragmento")
    if n_frag:
        print(f"\n  Los {n_frag} fragmentos NO son alarmas espurias: son "
              f"eventos reales")
        print(f"  mal delimitados, que ademas se contaron como perdidos. "
              f"El recall")
        print(f"  y la precision estan ambos deprimidos por la MISMA causa.")

    # --- fondo de control -------------------------------------------
    ocupado = np.zeros(n, dtype=bool)
    for ev in catalogo:
        ocupado[max(0, ev.ini - int(30 * FS)):ev.fin + int(30 * FS)] = True
    for d in det:
        ocupado[max(0, d.ini - int(30 * FS)):d.fin + int(30 * FS)] = True

    rng = np.random.default_rng(42)
    dur_tipica = int(np.median([ev.dur for ev in catalogo]))
    fondos, intentos = [], 0
    while len(fondos) < 200 and intentos < 20000:
        intentos += 1
        s0 = int(rng.integers(int(60 * FS), n - dur_tipica - 1))
        if ocupado[s0:s0 + dur_tipica].any() or not valida[s0]:
            continue
        fondos.append((s0, s0 + dur_tipica))

    # --- SNR de los tres grupos -------------------------------------
    print("\n" + "=" * 70)
    print(f"  RELACION SENAL/RUIDO POR GRUPO (canal {args.canal})")
    print("=" * 70)
    v_tp = describir("Eventos catalogados (TP)",
                     [snr_intervalo(a, args.canal, g_.ini, g_.fin)
                      for g_, _, _ in pares])
    v_fp = describir("Falsos positivos", 
                     [snr_intervalo(a, args.canal, d.ini, d.fin) for d in fp])
    v_bg = describir("Fondo aleatorio (control)",
                     [snr_intervalo(a, args.canal, s0, s1)
                      for s0, s1 in fondos])

    if v_tp.size and v_fp.size:
        umbral = float(np.percentile(v_tp, 10))
        n_como_evento = int((v_fp >= umbral).sum())
        print("\n" + "=" * 70)
        print("  VEREDICTO")
        print("=" * 70)
        print(f"  Umbral = percentil 10 de la SNR de eventos reales: "
              f"{umbral:.1f} dB")
        print(f"  Falsos positivos que lo superan: {n_como_evento} de "
              f"{v_fp.size} ({100 * n_como_evento / v_fp.size:.1f} %)")
        if v_bg.size:
            n_bg = int((v_bg >= umbral).sum())
            print(f"  Fondo aleatorio que lo supera:  {n_bg} de {v_bg.size} "
                  f"({100 * n_bg / v_bg.size:.1f} %)  <- tasa esperable "
                  f"si fueran ruido")
        print(f"""
  LECTURA
    Si el porcentaje de FP que superan el umbral es MUCHO mayor que el
    del fondo aleatorio, entonces esos FP contienen senal sismica real y
    el catalogo no es exhaustivo. En ese caso la precision reportada de
    los tres detectores es una COTA INFERIOR, y hay que decirlo asi en
    la tesis en vez de presentarla como la precision verdadera.

    Si ambos porcentajes se parecen, los FP son ruido y las metricas
    valen tal como estan.
""")

    # --- distancia al evento catalogado mas cercano ------------------
    if fp:
        inicios = np.array([ev.ini for ev in catalogo])
        d_min = [float(np.min(np.abs(inicios - d.ini))) / FS for d in fp]
        d_min = np.array(d_min)
        print("  Distancia de cada FP al evento catalogado mas cercano:")
        print(f"    mediana {np.median(d_min):.0f} s | "
              f"menos de 30 s: {int((d_min < 30).sum())} | "
              f"menos de 120 s: {int((d_min < 120).sum())} de {len(fp)}")
        print("    (muchos FP muy cerca de eventos reales sugiere "
              "fragmentacion,\n     no deteccion espuria)")

    os.makedirs(SALIDA, exist_ok=True)
    salida = pd.DataFrame([
        dict(tipo="fp", ini=d.ini, fin=d.fin, dur_s=d.dur / FS,
             snr_db=snr_intervalo(a, args.canal, d.ini, d.fin)) for d in fp
    ] + [
        dict(tipo="tp", ini=g_.ini, fin=g_.fin, dur_s=g_.dur / FS,
             snr_db=snr_intervalo(a, args.canal, g_.ini, g_.fin))
        for g_, _, _ in pares
    ])
    ruta = os.path.join(SALIDA, "revision_falsos_positivos.csv")
    salida.to_csv(ruta, index=False)
    print(f"\nDetalle en {ruta}")

    if args.figuras > 0 and fp:
        snrs_fp = [snr_intervalo(a, args.canal, d.ini, d.fin) for d in fp]
        figuras_fp(a, args.canal, fp, catalogo, snrs_fp, args.figuras,
                   os.path.join(SALIDA, "figuras_fp"))


if __name__ == "__main__":
    main()
