#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema completo sobre la traza continua de FRE (OE4): deteccion y recorte con PhaseNet en ventanas de
5 minutos (como ventanas_fre3c.py), espectrograma de cada recorte en una ventana de 300 s centrada en el
(como las de Jose, con su filtro sobre la senal sin filtrar y el rango de dB del maximo de esa ventana) y
clasificacion con los 5 modelos de cada red guardados por folds_cnn.py, con el recorte y con la ventana fija.
Compara con el catalogo: clasificacion de los recortes con IoU >= 0,3; sensibilidad completa por clase
(detectado, recortado con IoU >= 0,3 y bien clasificado); falsas alarmas clasificadas como volcanicas por
hora; promedio, minimo, maximo y desviacion entre los 5 modelos; y el tiempo de cada etapa.

Uso: python sistema_traza.py [--folds ~/tesis/resultados/clasificacion/folds] [--modelos efficientnet_b0]
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "clasificacion"))
import entrenar_cnn as ec
import espectrogramas_fijos as ef
import espectrogramas_recortes as er

lib, ed, iu = er.rv.lib, er.rv.ed, er.rv.iu
FS = lib.FS
N_VENTANA = int(round(ef.VENTANA_S * FS))
IOU_MIN = er.IOU_ETIQUETA
FOLDS = os.path.join(lib.DIR_TESIS, "resultados", "clasificacion", "folds")
SALIDA = os.path.join(lib.DIR_TESIS, "resultados", "sistema")


def ventana_centrada(a, b, lim_a, lim_b, n=N_VENTANA):
    """Inicio de la ventana de n muestras con el recorte al medio, sin salir de su tramo; si el recorte es mas
    largo que la ventana, la ventana parte en el inicio del recorte."""
    w0 = a if b - a >= n else (a + b) // 2 - n // 2
    return int(min(max(w0, lim_a), lim_b - n))


def imagenes(crudo, ivs, limites, sos, fija):
    """Imagen del recorte y del tramo fijo de cada recorte, coloreadas con el rango de su ventana centrada."""
    rec, fij, info = [], [], []
    for a, b in ivs:
        lim_a, lim_b = next((x, y) for x, y in limites if x <= a < y)
        w0 = ventana_centrada(a, b, lim_a, lim_b)
        esp = {c: er.espectrograma(crudo[c][w0:w0 + N_VENTANA], sos) for c in ec.COMPONENTES}
        dbs, lims = {c: v[0] for c, v in esp.items()}, {c: (v[1], v[2]) for c, v in esp.items()}
        n_col = dbs["Z"].shape[1]
        rec.append(er.imagen(dbs, lims, er.columnas(a - w0, min(b, w0 + N_VENTANA) - w0, n_col)))
        fa, fb = ef.tramo_fijo((a - w0) / FS, fija)
        fij.append(er.imagen(dbs, lims, er.columnas(fa * FS, fb * FS, n_col)))
        info.append(dict(ventana_ini=w0, cortado=b > w0 + N_VENTANA))
    return rec, fij, info


def modelos_guardados(carpeta, redes=None):
    """Pesos y configuracion de cada modelo de folds_cnn.py, por red y conjunto (recorte o ventana fija)."""
    out = {}
    for ruta in sorted(glob.glob(os.path.join(carpeta, "*.json"))):
        r = json.load(open(ruta))
        if "test" not in r or not os.path.exists(ruta[:-5] + ".pt"):
            continue
        cfg = r["configuracion"]
        if redes and cfg["modelo"] not in redes:
            continue
        out.setdefault((cfg["modelo"], cfg["conjunto"]), []).append((cfg, ruta[:-5] + ".pt"))
    return out


def clasificar(cfg, pesos, imgs, disp, lote=64):
    """Probabilidad de cada clase para cada imagen, con el modelo y la normalizacion de su entrenamiento."""
    red = ec.construir(cfg["modelo"], len(ec.CLASES), False, cfg.get("dropout"))[0]
    red.load_state_dict(torch.load(pesos, map_location="cpu"))
    red.to(disp).eval()
    mean, std = torch.tensor(cfg["mean"]).view(3, 1, 1), torch.tensor(cfg["std"]).view(3, 1, 1)
    prob = torch.sigmoid if cfg.get("probabilidad") == "sigmoide" else (lambda z: torch.softmax(z, 1))
    salida = []
    with torch.no_grad():
        for i in range(0, len(imgs), lote):
            x = torch.stack([ec.a_entrada(Image.fromarray(im), cfg["tamano"], mean, std) for im in imgs[i:i + lote]])
            salida.append(prob(red(x.to(disp))).cpu().numpy())
    return np.concatenate(salida) if salida else np.zeros((0, len(ec.CLASES)))


def etiquetas(eventos, ivs):
    """Etiqueta de cada recorte como en el entrenamiento: clase del evento con IoU >= 0,3, 'fuera' o 'sin caja'."""
    cajas = eventos.assign(tipo_marca="TRAZA")[["ini", "fin", "clase", "tipo_marca"]]
    return pd.DataFrame(er.etiquetar(cajas, ivs))


def metricas(pred, etq, inc, ivs, horas):
    """Para un modelo: matriz y metricas de los recortes con etiqueta de clase; sensibilidad completa por clase
    de los eventos incluidos; falsas alarmas (sin caja) y las que se llaman volcanicas, por hora."""
    clase = etq.etiqueta.map(ec.MAPA)
    con_clase = etq.etiqueta.isin([k for k in ec.MAPA if k != "sin caja"]).to_numpy()
    y = clase[con_clase].map({k: i for i, k in enumerate(ec.CLASES)}).to_numpy()
    c = ec.matriz(y, pred[con_clase])
    m = ec.metricas(c)
    cat = [ed.Intervalo(int(r.ini), int(r.fin), r.clase) for r in inc.itertuples()]
    det = [ed.Intervalo(int(a), int(b)) for a, b in ivs]
    pos = {id(d): k for k, d in enumerate(det)}
    pares, _, _ = ed.emparejar(cat, det, iu.TOCA)
    bien = {id(g): v >= IOU_MIN and ec.CLASES[pred[pos[id(d)]]] == ec.MAPA[g.clase] for g, d, v in pares}
    out = dict(matriz=c.tolist(), f1_macro_volcanicas=m["f1_macro_volcanicas"])
    for k in ec.CLASES:
        out[f"{k}_f1"], out[f"{k}_sensibilidad"] = m[k]["f1"], m[k]["sensibilidad"]
    for k in lib.CLASES + ["volcanico"]:
        g = [x for x in cat if (x.clase in lib.VOLCANICAS if k == "volcanico" else x.clase == k)]
        out[f"{k}_completo"] = float(np.mean([bien.get(id(x), False) for x in g])) if g else float("nan")
    sin = (etq.etiqueta == "sin caja").to_numpy()
    vol = np.isin(pred, [ec.CLASES.index(k) for k in ec.VOLCANICAS])
    out.update(falsas_por_hora=float(sin.sum() / horas), falsas_volcanicas_por_hora=float((sin & vol).sum() / horas))
    return out


def resumir(filas):
    """Promedio, minimo, maximo y desviacion estandar (muestral) entre los modelos de cada red y conjunto."""
    d = pd.DataFrame(filas).drop(columns=["matriz"])
    met = [c for c in d.columns if c not in ("modelo", "conjunto", "fold", "segundos")]
    g = d.groupby(["modelo", "conjunto"])[met]
    return pd.concat({"promedio": g.mean(), "minimo": g.min(), "maximo": g.max(), "desviacion": g.std(ddof=1)},
                     names=["estadistico"]).reset_index()


def evaluar(crudo, limites, ivs, eventos, inc, horas, modelos, sos, fija, disp):
    """Imagenes, etiquetas y, por modelo, predicciones y metricas; tambien los tiempos de cada etapa."""
    t = time.perf_counter()
    rec, fij, info = imagenes(crudo, ivs, limites, sos, fija)
    t_img = time.perf_counter() - t
    etq = pd.concat([etiquetas(eventos, ivs), pd.DataFrame(info)], axis=1)
    etq.insert(0, "fin", [b for _, b in ivs])
    etq.insert(0, "ini", [a for a, _ in ivs])
    filas = []
    for (modelo, conjunto), lista in sorted(modelos.items()):
        for cfg, pesos in sorted(lista, key=lambda x: x[0]["fold"]):
            t = time.perf_counter()
            p = clasificar(cfg, pesos, fij if conjunto.startswith("fijo") else rec, disp)
            seg = time.perf_counter() - t
            pred = p.argmax(1)
            etq[f"{modelo}_{conjunto}_fold{cfg['fold']}"] = [ec.CLASES[i] for i in pred]
            filas.append(dict(modelo=modelo, conjunto=conjunto, fold=cfg["fold"], segundos=seg,
                              **metricas(pred, etq, inc, ivs, horas)))
    return etq, filas, t_img


def deteccion(cfg):
    """Recortes de PhaseNet en ventanas de 5 minutos, como ventanas_fre3c.py, con el tiempo de la deteccion."""
    import ventanas_fre3c as vf

    up, us = cfg["P_threshold"], cfg["S_threshold"]
    kw = dict(overlap=int(cfg["overlap"]), blinding=(int(cfg["blinding"]), int(cfg["blinding"])),
              stacking=cfg["stacking"])
    z = np.load(lib.ruta_datos(vf.FILTRO))
    valida, datos = z["valida"], {c: z[c] for c in ("Z", "N", "E")}
    m = vf.an.cargar_modelo("phasenet")
    t = time.perf_counter()
    filas = []
    for a, b in vf.br.generar_ventanas(lib.N_TOTAL, lib.CORTES):
        filas += vf.avisos_de_ventana(m.annotate(vf.an.stream_tramo(datos, a, b), **kw), up, us)
    _, _, o = vf.juntar(pd.DataFrame(filas))
    t_det = time.perf_counter() - t
    t = time.perf_counter()
    env = vf.rc.envolvente(datos["Z"], datos["N"], datos["E"])
    ivs = vf.rc.recortes(o[valida[o]], env, valida, vf.rc.K_RUIDO, corte=True)
    return [(int(a), int(b)) for a, b in ivs], valida, t_det, time.perf_counter() - t


def crudo_alineado():
    """Z, N y E de FRE sin filtrar sobre la grilla de la traza, alineadas como en preparar_fre3c.py."""
    sys.path.insert(0, os.path.join(AQUI, "..", "fre3c"))
    import preparar_fre3c as pr

    zen = np.load(lib.NPY_ZENODO, mmap_mode="r")
    sos15 = lib.sos_filtro(lib.FILTROS["f1_15"])
    crudo = {c: np.zeros(lib.N_TOTAL) for c in ec.COMPONENTES}
    for nombre, n_tramo, ini, fin in [(t[0], t[1], t[2], t[3]) for t in pr.vc.TRAMOS]:
        c, _, _ = pr.preparar_tramo(nombre, n_tramo, ini, fin, zen, sos15)
        for k in ec.COMPONENTES:
            crudo[k][ini:fin] = c[k]
    return crudo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=FOLDS, help="carpeta con los JSON y pesos de folds_cnn.py")
    ap.add_argument("--modelos", nargs="*", default=None, help="solo estas redes")
    ap.add_argument("--ventana-fija", type=float, default=60.0)
    ap.add_argument("--salida", default=SALIDA)
    args = ap.parse_args()
    import ventanas_fre3c as vf
    from scipy.signal import butter

    modelos = modelos_guardados(args.folds, args.modelos)
    incompletos = {k: len(v) for k, v in modelos.items() if len(v) != 5}
    if not modelos or incompletos:
        raise SystemExit(f"DETENIDO: faltan modelos en {args.folds}: {incompletos or 'ninguno'}")
    with open(os.path.join(lib.DIR_SALIDA, "configuracion_elegida.json")) as f:
        cfg = json.load(f)[vf.FLUJO]
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    inc = eventos[eventos.incluido].reset_index(drop=True)
    print(f"Modelos: {', '.join(f'{m} {c} ({len(v)})' for (m, c), v in sorted(modelos.items()))}; {disp}")

    ivs, valida, t_det, t_rec = deteccion(cfg)
    horas = valida.sum() / FS / 3600
    ruta_prev = os.path.join(SALIDA, "ventanas_recortes.csv")
    if os.path.exists(ruta_prev):
        prev = pd.read_csv(ruta_prev)
        igual = list(zip(prev.ini, prev.fin)) == ivs
        print(f"Recortes iguales a los de ventanas_fre3c.py: {'si' if igual else 'NO'} ({len(ivs)} y {len(prev)})")
    print(f"{len(ivs)} recortes en {horas:.2f} h validas; deteccion {t_det:.1f} s, recorte {t_rec:.1f} s")

    crudo = crudo_alineado()
    sos = butter(er.ORDEN_FILTRO, list(er.BANDA_FILTRO), btype="bandpass", fs=FS, output="sos")
    etq, filas, t_img = evaluar(crudo, lib.LIMITES_TRAMO, ivs, con_senal, inc, horas, modelos, sos,
                                args.ventana_fija, disp)
    print(f"Espectrogramas e imagenes: {t_img:.1f} s ({t_img / max(len(ivs), 1):.3f} s por recorte); "
          f"{int(etq.cortado.sum())} recortes mas largos que la ventana")
    print(etq.etiqueta.value_counts().to_string())

    r = resumir(filas)
    cols = (["f1_macro_volcanicas"] + [f"{k}_{x}" for k in ec.CLASES for x in ("f1", "sensibilidad")]
            + [f"{k}_completo" for k in lib.CLASES + ["volcanico"]] + ["falsas_por_hora", "falsas_volcanicas_por_hora"])
    for (modelo, conjunto), g in r.groupby(["modelo", "conjunto"]):
        print(f"\n{modelo}, {conjunto} (5 modelos)")
        print(g.set_index("estadistico").loc[["promedio", "minimo", "maximo", "desviacion"], cols].T.round(3).to_string())
    for (modelo, conjunto), g in pd.DataFrame(filas).groupby(["modelo", "conjunto"]):
        c = np.sum([np.array(x) for x in g.matriz], axis=0)
        p = c / np.where(c.sum(axis=1, keepdims=True) > 0, c.sum(axis=1, keepdims=True), np.nan)
        print(f"\nMatriz normalizada por fila, {modelo} {conjunto} (suma de los 5 modelos; filas real, columnas prediccion):")
        print(pd.DataFrame(p, index=ec.CLASES, columns=ec.CLASES).round(3).to_string())
        print(f"   clasificacion: {g.segundos.mean():.1f} s por modelo ({g.segundos.mean() / max(len(ivs), 1) * 1000:.1f} ms "
              f"por recorte)")

    os.makedirs(args.salida, exist_ok=True)
    etq.to_csv(os.path.join(args.salida, "sistema_traza_recortes.csv"), index=False)
    pd.DataFrame(filas).to_csv(os.path.join(args.salida, "sistema_traza_modelos.csv"), index=False)
    r.to_csv(os.path.join(args.salida, "sistema_traza_resumen.csv"), index=False)
    json.dump(dict(recortes=len(ivs), horas=float(horas), deteccion_s=t_det, recorte_s=t_rec, imagenes_s=t_img,
                   dispositivo=str(disp)), open(os.path.join(args.salida, "sistema_traza_tiempos.json"), "w"), indent=1)
    print(f"\nTablas en {args.salida}")


if __name__ == "__main__":
    main()
