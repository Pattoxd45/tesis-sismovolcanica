#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Espectrogramas de los recortes de las ventanas completas de Jose: recorta cada ventana con la
configuracion elegida (PhaseNet P y S, energia con corte, ruido solo dentro de la ventana), calcula
el espectrograma de Z, N y E de la ventana con los parametros de Jose y guarda por recorte una imagen
con sus columnas (Z arriba, N al medio, E abajo) a su tamano original. Anota para cada recorte la caja
de Jose que le toca, su IoU y una etiqueta propuesta. Compara el espectrograma de Z con el de los
codigos de Jose en las primeras ventanas.

Uso: python espectrogramas_recortes.py [--limite 50]
"""

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import butter, sosfiltfilt, spectrogram

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import recorte_ventanas_jose as rv
import extraer_previo_jose as ep

lib, ed, iu, ej = rv.lib, rv.ed, rv.iu, rv.ej
FS = lib.FS
# parametros del espectrograma de Jose (configuracion_espectrograma_YOLO_SOLO_Z_300s.txt)
ORDEN_FILTRO = 15
BANDA_FILTRO = (1.0, 15.0)
NPERSEG, NOVERLAP, NFFT = 256, 192, 512
BANDA_IMAGEN = (0.0, 15.0)
RANGO_DB = 60.0
CMAP = "jet"
# IoU desde el que un recorte toma la clase de su caja: el mismo de la evaluacion
IOU_ETIQUETA = ed.IOU_MIN
DIR_IMAGENES = os.path.join(lib.DIR_TESIS, "datos", "espectrogramas_recortes")
CODIGO_JOSE = os.path.join(ej.DIR_JOSE, "Codigos", "Verificacion_descargar_preprocesamiento",
                           "17_generar_dataset_yolo_completo_300s_CORREGIDO_OPTIMIZADO.py")
# ventanas en que se compara el espectrograma de Z con el que calculan los codigos de Jose
N_VERIFICAR = 20


def espectrograma(x, sos):
    """Como crear_espectrograma de Jose: Butterworth 1-15 Hz de ida y vuelta, PSD en dB y 0-15 Hz; vmin y vmax de la imagen."""
    f, t, sxx = spectrogram(sosfiltfilt(sos, np.asarray(x, dtype=np.float64)), fs=FS, window="hann", nperseg=NPERSEG,
                            noverlap=NOVERLAP, nfft=NFFT, detrend="constant", scaling="density", mode="psd")
    m = (f >= BANDA_IMAGEN[0]) & (f <= BANDA_IMAGEN[1])
    db = 10.0 * np.log10(np.maximum(sxx[m, :], np.finfo(float).tiny))
    vmax = float(np.nanmax(db))
    return db, vmax - RANGO_DB, vmax


def columnas(ini, fin, n_col):
    """Columnas del espectrograma cuyo centro cae dentro del recorte (al menos la mas cercana a su centro)."""
    centro = NPERSEG // 2 + (NPERSEG - NOVERLAP) * np.arange(n_col)
    k = np.flatnonzero((centro >= ini) & (centro < fin))
    return k if k.size else np.array([int(np.argmin(np.abs(centro - (ini + fin) / 2)))])


def imagen(dbs, lims, k):
    """Z, N y E apiladas, cada una con la escala de su espectrograma completo, como plt.imsave con origin lower."""
    import matplotlib
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    partes = []
    for c in ("Z", "N", "E"):
        sm = ScalarMappable(norm=Normalize(*lims[c]), cmap=matplotlib.colormaps[CMAP])
        partes.append(sm.to_rgba(dbs[c][:, k], bytes=True)[::-1, :, :3])
    return np.vstack(partes)


def etiquetar(cajas, ivs):
    """Por recorte: la caja que le toca (uno a uno por IoU), su IoU y la etiqueta propuesta."""
    cat = [ed.Intervalo(int(a), int(b), c) for a, b, c in zip(cajas.ini, cajas.fin, cajas.clase)]
    det = [ed.Intervalo(int(a), int(b)) for a, b in ivs]
    pares, _, _ = ed.emparejar(cat, det, iu.TOCA)
    par = {id(d): (g, v) for g, d, v in pares}
    tipo = {id(g): t for g, t in zip(cat, cajas.tipo_marca)}
    out = []
    for d in det:
        g, v = par.get(id(d), (None, 0.0))
        toca = any(ed.iou(g2, d) > 0 for g2 in cat)
        if g is not None and v >= IOU_ETIQUETA:
            etiqueta = g.clase
        elif toca:
            etiqueta = "fuera"
        else:
            etiqueta = "sin caja"
        out.append(dict(clase_caja=g.clase if g is not None else "", tipo_caja=tipo[id(g)] if g is not None else "",
                        iou=v, etiqueta=etiqueta))
    return out


def modulo_jose():
    if not os.path.exists(CODIGO_JOSE):
        return None
    spec = importlib.util.spec_from_file_location("generador_jose", CODIGO_JOSE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def muestra(idx, dir_img, ruta, por_clase=4):
    """Hoja con algunos recortes de test por etiqueta, para revisarlos a ojo."""
    import matplotlib.pyplot as plt

    etiquetas = ["VT", "LP", "TR", "AV", "IC", "sin caja"]
    fig, ax = plt.subplots(len(etiquetas), por_clase, figsize=(3 * por_clase, 2.6 * len(etiquetas)))
    t = idx[idx.split == "test"]
    for i, et in enumerate(etiquetas):
        x = t[t.etiqueta == et]
        x = x.sample(min(por_clase, len(x)), random_state=0) if len(x) else x
        for j in range(por_clase):
            a = ax[i, j]
            a.axis("off")
            if j < len(x):
                r = x.iloc[j]
                a.imshow(plt.imread(os.path.join(dir_img, r.archivo)), aspect="auto")
                a.set_title(f"{et} {r.dur_s:.0f} s IoU {r.iou:.2f}", fontsize=8)
    fig.suptitle("Recortes de test (Z arriba, N al medio, E abajo; 0-15 Hz)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None, help="solo las primeras N ventanas, para probar")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(lib.DIR_SALIDA, "configuracion_elegida.json")) as f:
        cfg = json.load(f)[rv.FLUJO]
    up, us = cfg["P_threshold"], cfg["S_threshold"]
    kw = dict(overlap=int(cfg["overlap"]), blinding=(int(cfg["blinding"]), int(cfg["blinding"])),
              stacking=cfg["stacking"])
    reg = pd.read_csv(ej.REGISTRO).set_index("evento")
    ev = ep.completas(ej.REGISTRO)
    if args.limite:
        ev = ev[:args.limite]
    cajas = rv.cargar_cajas()
    print(f"{len(ev)} ventanas completas; recorte con PhaseNet P_threshold {up}, S_threshold {us} y ruido solo dentro "
          f"de la ventana; espectrograma de Jose (Butterworth orden {ORDEN_FILTRO}, {NPERSEG}/{NOVERLAP}/{NFFT}, "
          f"{RANGO_DB:.0f} dB, {CMAP})")

    m = rv.an.cargar_modelo("phasenet")
    sos_det = lib.sos_filtro(lib.FILTROS[rv.FILTRO])
    sos_esp = butter(ORDEN_FILTRO, list(BANDA_FILTRO), btype="bandpass", fs=FS, output="sos")
    jose = modulo_jose()
    difs = []
    os.makedirs(DIR_IMAGENES, exist_ok=True)
    filas, t_ini = [], time.time()
    for n, e in enumerate(ev, 1):
        w = np.load(os.path.join(ej.DIR_SALIDA, f"{e}_zne.npz"))
        x = {c: w[c] for c in ("Z", "N", "E")}
        _, _, _, ivs = rv.procesar(x, 0, m, kw, up, us, sos_det)
        esp = {c: espectrograma(x[c], sos_esp) for c in x}
        dbs = {c: v[0] for c, v in esp.items()}
        lims = {c: (v[1], v[2]) for c, v in esp.items()}
        if jose is not None and n <= N_VERIFICAR:
            _, _, db_j, vmin_j, vmax_j = jose.crear_espectrograma(jose.filtrar_butterworth(x["Z"]))
            difs.append(max(float(np.max(np.abs(db_j - dbs["Z"]))), abs(vmax_j - lims["Z"][1])))
        n_col = dbs["Z"].shape[1]
        for k, ((a, b), et) in enumerate(zip(ivs, etiquetar(cajas[e], ivs))):
            col = columnas(a, b, n_col)
            archivo = f"{e}_{k:02d}.png"
            Image.fromarray(imagen(dbs, lims, col)).save(os.path.join(DIR_IMAGENES, archivo))
            filas.append(dict(recorte=archivo[:-4], ventana=e, split=reg.loc[e, "split"], k=k, ini_s=a / FS,
                              fin_s=b / FS, dur_s=(b - a) / FS, columnas=col.size, archivo=archivo, **et))
        if n % 500 == 0:
            print(f"  {n}/{len(ev)} ventanas en {(time.time() - t_ini) / 60:.1f} min", flush=True)

    idx = pd.DataFrame(filas)
    os.makedirs(rv.DIR_SALIDA, exist_ok=True)
    ruta_idx = os.path.join(rv.DIR_SALIDA, "espectrogramas_recortes.csv")
    idx.to_csv(ruta_idx, index=False)
    print(f"\nTiempo: {(time.time() - t_ini) / 60:.1f} min; {len(idx)} recortes de {len(ev)} ventanas")
    if difs:
        print(f"Espectrograma de Z contra el de los codigos de Jose en {len(difs)} ventanas: diferencia maxima "
              f"{max(difs):.2e} dB")
    else:
        print("No se encontro el codigo de Jose; no se comparo el espectrograma")
    print(f"\nEtiqueta propuesta (caja con IoU >= {IOU_ETIQUETA}; 'fuera': toca una caja con menos; "
          f"'sin caja': no toca ninguna)")
    print(pd.crosstab(idx.etiqueta, idx.split, margins=True, margins_name="total").to_string())
    print("\nDuracion de los recortes por etiqueta (s): mediana, percentil 90 y maximo")
    d = idx.groupby("etiqueta").dur_s.agg(["median", lambda s: s.quantile(0.9), "max"]).round(1)
    d.columns = ["mediana", "p90", "max"]
    print(d.to_string())
    ruta_m = os.path.join(rv.DIR_SALIDA, "muestra_espectrogramas.png")
    muestra(idx, DIR_IMAGENES, ruta_m)
    print(f"\nImagenes en {DIR_IMAGENES}; indice en {ruta_idx}; hoja de muestra en {ruta_m}")


if __name__ == "__main__":
    main()
