#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnostico_stalta.py
=====================

No calcula metricas finales: DIAGNOSTICA por que el recall salio bajo.

Responde cuatro preguntas concretas:
  1. Cuantos disparos hay y de que duracion? (hipotesis: demasiado cortos)
  2. Cuantos disparos caen sobre huecos de datos? (hipotesis: artefactos)
  3. A nivel de MUESTRA, ve el detector la energia de los eventos?
     (desacopla "detecto algo" de "delimito bien")
  4. Cual es el mejor IoU alcanzable por cada evento del catalogo?
     (si el mejor IoU es alto pero el recall bajo, el problema es el
      emparejamiento; si el mejor IoU es bajo, es el recorte)

Uso:
    python diagnostico_stalta.py
    python diagnostico_stalta.py --canales 1 2 3
"""

import argparse
import numpy as np
import pandas as pd

BASE = "/home/patto/tesis/datos/zenodo"
NPY = f"{BASE}/NVCh_10h_continuous_trace.npy"
CSV = f"{BASE}/NVCh_10h_continuous_trace_reference.csv"

FS = 100.0
CORTES = [2_160_000, 3_240_000]
GUARDA_S = 30.0
CLASES = ["VT", "LP", "TR", "AV", "IC"]

STA_S, LTA_S = 1.0, 10.0
THR_ON, THR_OFF = 3.0, 1.5
BANDA = (1.0, 20.0)


def segmentos(n):
    b = [0] + CORTES + [n]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def mascara_huecos(x, dilatar_s=2.0):
    """Marca los ceros y dilata el borde: el filtro hace timbrar el escalon."""
    m = (x == 0)
    k = int(dilatar_s * FS)
    if k > 0 and m.any():
        m = np.convolve(m.astype(np.float32),
                        np.ones(2 * k + 1), mode="same") > 0
    return m


def disparos_canal(x, ini):
    """STA/LTA de un canal. Devuelve (triggers_globales, mascara_huecos)."""
    from obspy.signal.trigger import classic_sta_lta, trigger_onset
    from obspy.signal.filter import bandpass

    huecos = mascara_huecos(x)
    if np.allclose(x, 0):
        return [], huecos

    xf = bandpass(x, BANDA[0], BANDA[1], df=FS, corners=4, zerophase=True)
    cft = classic_sta_lta(xf, int(STA_S * FS), int(LTA_S * FS))
    cft = np.nan_to_num(cft, nan=0.0, posinf=0.0, neginf=0.0)
    trg = trigger_onset(cft, THR_ON, THR_OFF)
    return [(ini + int(a), ini + int(b)) for a, b in trg], huecos


def intervalos(mask):
    mask = np.asarray(mask).astype(np.int8)
    d = np.diff(np.concatenate(([0], mask, [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canales", nargs="+", type=int, default=[1, 2, 3, 5, 6])
    args = ap.parse_args()

    a = np.load(NPY, mmap_mode="r")
    df = pd.read_csv(CSV)
    n = a.shape[-1]

    valida = np.ones(n, dtype=bool)
    g = int(GUARDA_S * FS)
    for c in CORTES:
        valida[max(0, c - g):min(n, c + g)] = False

    cat = [(int(r.idx_start), int(r.idx_end), r.event_type)
           for r in df.itertuples()
           if valida[int(r.idx_start)] and valida[int(r.idx_end) - 1]]

    votos = np.zeros(n, dtype=np.int16)
    hueco_total = np.zeros(n, dtype=bool)

    print("=" * 70)
    print("  1. DISPAROS POR CANAL")
    print("=" * 70)
    print(f"  {'ch':<4}{'n_disp':>8}{'dur_med':>10}{'dur_p90':>10}"
          f"{'%en_hueco':>11}{'%tiempo_on':>12}")

    for ch in args.canales:
        trg_ch, huecos_ch = [], np.zeros(n, dtype=bool)
        for ini, fin in segmentos(n):
            x = np.asarray(a[ch, ini:fin], dtype=np.float64)
            t, h = disparos_canal(x, ini)
            trg_ch += t
            huecos_ch[ini:fin] = h
        hueco_total |= huecos_ch

        if not trg_ch:
            print(f"  {ch:<4}{0:>8}")
            continue

        dur = np.array([(b - aa) / FS for aa, b in trg_ch])
        en_hueco = sum(1 for aa, b in trg_ch if huecos_ch[aa:b].mean() > 0.2)
        for aa, b in trg_ch:
            votos[aa:b] += 1
        print(f"  {ch:<4}{len(trg_ch):>8}{np.median(dur):>10.1f}"
              f"{np.percentile(dur, 90):>10.1f}"
              f"{100 * en_hueco / len(trg_ch):>11.1f}"
              f"{100 * dur.sum() * FS / n:>12.1f}")

    print(f"\n  Huecos de datos cubren {100 * hueco_total.mean():.1f}% "
          f"de la traza (con dilatacion de 2 s)")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  2. COBERTURA A NIVEL DE MUESTRA (ve la energia?)")
    print("=" * 70)
    print("  Fraccion de muestras de cada evento marcadas por >=k canales\n")
    print(f"  {'clase':<7}{'n':>5}" + "".join(f"{'k=' + str(k):>9}"
                                              for k in [1, 2, 3]))
    for c in CLASES:
        evs = [(s, e) for s, e, cc in cat if cc == c]
        if not evs:
            continue
        fila = f"  {c:<7}{len(evs):>5}"
        for k in [1, 2, 3]:
            cov = [np.mean(votos[s:e] >= k) for s, e in evs]
            fila += f"{np.mean(cov):>9.3f}"
        print(fila)

    fondo = ~valida.copy()
    for s, e, _ in cat:
        fondo[max(0, s - 500):e + 500] = True
    fondo = ~fondo
    print(f"\n  Fondo (sin eventos): fraccion marcada por >=1 canal = "
          f"{np.mean(votos[fondo] >= 1):.3f}, >=2 = "
          f"{np.mean(votos[fondo] >= 2):.3f}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  3. MEJOR IoU ALCANZABLE POR EVENTO")
    print("=" * 70)

    for k in [1, 2]:
        for gap_s in [0.0, 5.0, 15.0]:
            mask = (votos >= k) & valida
            ivs = intervalos(mask)
            # fusionar por gap
            fus = []
            for s, e in ivs:
                if fus and s <= fus[-1][1] + int(gap_s * FS):
                    fus[-1] = (fus[-1][0], max(fus[-1][1], e))
                else:
                    fus.append((s, e))
            fus = [(s, e) for s, e in fus if e - s >= int(2.0 * FS)]

            mejores = []
            for s, e, _ in cat:
                best = 0.0
                for ds, de in fus:
                    if de < s:
                        continue
                    if ds > e:
                        break
                    inter = min(e, de) - max(s, ds)
                    if inter > 0:
                        best = max(best, inter / (max(e, de) - min(s, ds)))
                mejores.append(best)
            mejores = np.array(mejores)
            print(f"  k>={k} gap={gap_s:>4.0f}s | n_det={len(fus):>5} | "
                  f"IoU>0: {np.mean(mejores > 0):.3f} | "
                  f">=0.1: {np.mean(mejores >= 0.1):.3f} | "
                  f">=0.3: {np.mean(mejores >= 0.3):.3f} | "
                  f">=0.5: {np.mean(mejores >= 0.5):.3f}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  4. LECTURA")
    print("=" * 70)
    print("""
  - Si '%en_hueco' es alto -> hay que enmascarar los huecos.
  - Si la cobertura por muestra es alta pero el IoU bajo -> el detector
    SI ve los eventos, el problema es que fragmenta el recorte:
    corresponde fusionar disparos y bajar thr_off.
  - Si 'IoU>0' es alto pero '>=0.3' bajo -> mismo diagnostico.
  - Si la cobertura por muestra tambien es baja -> ahi si el detector
    no ve los eventos y hay que revisar filtro y umbrales.
""")


if __name__ == "__main__":
    main()
