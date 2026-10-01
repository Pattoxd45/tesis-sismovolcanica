#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluar_por_tramo.py
====================

Evaluacion por tramo de la traza continua. Solo MIDE: no ajusta ningun
parametro ni modifica ningun script.

    Nivel 1 (instante) : STA/LTA, PhaseNet y EQTransformer
    Nivel 2 (recorte)  : STA/LTA y EQTransformer

Reutiliza las cachas de inferencia ya calculadas, asi que PhaseNet y
EQTransformer no se vuelven a correr:

    resultados/cache_picks/picks_<modelo>_<modo>.csv   -> arribos con confianza
    resultados/cache_detection/detection_eqt_<modo>.npz -> curva Detection

Las funciones que leen esas cachas y reconstruyen las salidas son las de
significancia.py (picks_por_canal, eqt_intervalos, stalta_por_canal); la
coincidencia de red y las metricas son las de evaluar_detectores.py.

DOS CONFIGURACIONES
-------------------
Los valores globales que cita el informe no salen de los valores por
omision de evaluar_detectores.py, sino de los puntos de operacion
"mejor F1" que fija significancia.py en OP_NIVEL1 y OP_NIVEL2. Para que
quede a la vista, este script calcula los globales con las DOS
configuraciones, las compara contra los valores citados y recien entonces
reporta por tramo, usando la que reproduce. Si ninguna reproduce, se
detiene y muestra la diferencia.

Uso:
    python evaluar_por_tramo.py
    python evaluar_por_tramo.py --sin-cache      # recalcula STA/LTA

STA/LTA no tiene cacha de inferencia (es barato comparado con los modelos,
pero son varios minutos sobre 10 h). Este script cachea sus intervalos por
canal en resultados/cache_stalta/, con la configuracion en el nombre.

Autor: Patricio Valdes Troncoso - Universidad Catolica de Temuco
"""

import argparse
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "ventanas"))
import barrido as br                   # noqa: E402  (solo VENTANA_FIJA_S)
import evaluar_detectores as ed        # noqa: E402
import significancia as sig            # noqa: E402

FS = ed.FS
CACHE_STALTA = os.path.join(ed.SALIDA, "cache_stalta")

# Valores citados en el informe que hay que reproducir antes de
# reportar cualquier cosa por tramo.
OBJETIVO_N1 = {"stalta": 0.542, "phasenet": 0.527, "eqtransformer": 0.581}
OBJETIVO_N2_EQT = {"recall": 0.330, "iou_medio": 0.479}

# Los valores citados estan redondeados a tres decimales, asi que el
# margen de comparacion es medio paso de ese redondeo.
TOL_REPRODUCE = 5e-4


# =====================================================================
# LAS DOS CONFIGURACIONES
# =====================================================================

def configuraciones():
    """
    Devuelve las dos configuraciones a comparar.

    'ed' : valores por omision de evaluar_detectores.py (STALTA, UMBRAL_PICK,
           UMBRAL_PROB, FUSION_S, MIN_ESTACIONES, COMPONENTES_POR_MODELO y la
           mayor de las tolerancias de TOL_ONSET_S).
    'op' : puntos de operacion "mejor F1" de significancia.py.
    """
    stalta_ed = dict(ed.STALTA, fusion_s=ed.FUSION_S,
                     min_est=ed.MIN_ESTACIONES)
    cfg_ed = {
        "tol_onset_s": max(ed.TOL_ONSET_S),
        "nivel1": {
            "stalta": dict(stalta_ed),
            "phasenet": dict(modo=ed.COMPONENTES_POR_MODELO["phasenet"],
                             umbral=ed.UMBRAL_PICK,
                             min_est=ed.MIN_ESTACIONES),
            "eqtransformer": dict(modo=ed.COMPONENTES_POR_MODELO["eqtransformer"],
                                  umbral=ed.UMBRAL_PICK,
                                  min_est=ed.MIN_ESTACIONES),
        },
        "nivel2": {
            "stalta": dict(stalta_ed),
            "eqtransformer": dict(modo=ed.COMPONENTES_POR_MODELO["eqtransformer"],
                                  umbral=ed.UMBRAL_PROB,
                                  fusion_s=ed.FUSION_S,
                                  min_est=ed.MIN_ESTACIONES),
        },
    }
    cfg_op = {
        "tol_onset_s": sig.TOL_ONSET_S,
        "nivel1": {k: dict(v) for k, v in sig.OP_NIVEL1.items()},
        "nivel2": {k: dict(v) for k, v in sig.OP_NIVEL2.items()},
    }
    return {"ed": cfg_ed, "op": cfg_op}


def etiqueta(cfg, nombre, nivel):
    """Texto compacto de la configuracion de un detector."""
    c = cfg[nivel][nombre]
    if nombre == "stalta":
        return (f"STA {c['sta_s']:g} s / LTA {c['lta_s']:g} s, "
                f"thr {c['thr_on']:g}/{c['thr_off']:g}, "
                f"fusion {c['fusion_s']:g} s, min_est {c['min_est']}")
    base = f"componentes {c['modo']}, umbral {c['umbral']:g}"
    if "fusion_s" in c:
        base += f", fusion {c['fusion_s']:g} s"
    return base + f", min_est {c['min_est']}"


# =====================================================================
# RECONSTRUCCION DE LAS SALIDAS
# =====================================================================

def stalta_por_canal_cacheado(a, canales, cfg, usar_cache=True):
    """sig.stalta_por_canal con cacha en disco, indexada por configuracion."""
    clave = (f"sta{cfg['sta_s']:g}_lta{cfg['lta_s']:g}_on{cfg['thr_on']:g}"
             f"_off{cfg['thr_off']:g}_fus{cfg['fusion_s']:g}"
             f"_ch{'-'.join(map(str, canales))}")
    ruta = os.path.join(CACHE_STALTA, f"stalta_{clave}.json")
    if usar_cache and os.path.exists(ruta):
        print(f"      usando cache: {os.path.basename(ruta)}")
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        return {int(ch): {"intervalos": [tuple(x) for x in v["intervalos"]],
                          "onsets": v["onsets"]}
                for ch, v in d.items()}
    print(f"      corriendo STA/LTA ({clave}) ...", flush=True)
    pc = sig.stalta_por_canal(a, canales, cfg)
    os.makedirs(CACHE_STALTA, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump({str(ch): {"intervalos": [list(x) for x in v["intervalos"]],
                             "onsets": v["onsets"]}
                   for ch, v in pc.items()}, f)
    return pc


def onsets_de(nombre, cfg, a, valida, canales, usar_cache=True):
    """Onsets de red de un detector, en indices de muestra."""
    c = cfg["nivel1"][nombre]
    if nombre == "stalta":
        pc = stalta_por_canal_cacheado(a, canales, c, usar_cache)
    else:
        pc = sig.picks_por_canal(nombre, c["modo"], c["umbral"])
        if pc is None:
            return None
    return ed.coincidencia_onsets(pc, valida, c["min_est"])


def detecciones_de(nombre, cfg, a, valida, canales, usar_cache=True):
    """Detecciones de red (intervalos) de un detector del nivel 2."""
    c = cfg["nivel2"][nombre]
    if nombre == "stalta":
        pc = stalta_por_canal_cacheado(a, canales, c, usar_cache)
    else:
        pc = sig.eqt_intervalos(c["modo"], c["umbral"], c["fusion_s"], canales)
        if pc is None:
            return None
    return ed.coincidencia(pc, valida, c["min_est"])


# =====================================================================
# VERIFICACION CONTRA LOS VALORES CITADOS
# =====================================================================

def globales(cfg, a, catalogo, valida, horas, canales, usar_cache):
    """Nivel 1 y nivel 2 globales de una configuracion."""
    tol = cfg["tol_onset_s"]
    salida = {"nivel1": {}, "nivel2": {}, "onsets": {}, "detecciones": {}}
    for nombre in ("stalta", "phasenet", "eqtransformer"):
        ons = onsets_de(nombre, cfg, a, valida, canales, usar_cache)
        if ons is None:
            print(f"      {nombre}: falta la cacha de arribos, se omite")
            continue
        salida["onsets"][nombre] = ons
        r = ed.metricas_onset(catalogo, ons, horas, tolerancias=[tol])
        salida["nivel1"][nombre] = r[f"tol_{tol:g}s"]
        salida["nivel1"][nombre]["n_onsets"] = r["n_onsets"]
    for nombre in ("stalta", "eqtransformer"):
        det = detecciones_de(nombre, cfg, a, valida, canales, usar_cache)
        if det is None:
            print(f"      {nombre}: falta la cacha de Detection, se omite")
            continue
        salida["detecciones"][nombre] = det
        r, *_ = ed.metricas(catalogo, det, horas, ed.IOU_MIN)
        salida["nivel2"][nombre] = r
    return salida


def verificar(todo, cfgs):
    """Compara los globales de cada configuracion con los valores citados."""
    print("=" * 78)
    print("  VERIFICACION CONTRA LOS VALORES CITADOS EN EL INFORME")
    print("=" * 78)

    for clave in ("ed", "op"):
        print(f"\n  Configuracion '{clave}'  (tolerancia de onset "
              f"+-{cfgs[clave]['tol_onset_s']:g} s)")
        for nivel in ("nivel1", "nivel2"):
            for nombre in sorted(cfgs[clave][nivel]):
                print(f"    {nivel} {nombre:<14} {etiqueta(cfgs[clave], nombre, nivel)}")

    print("\n  " + "-" * 74)
    print(f"  Nivel 1: recall de onset")
    print(f"  {'detector':<16}{'citado':>9}{'config ed':>12}{'dif':>10}"
          f"{'config op':>12}{'dif':>10}")
    difs = {"ed": [], "op": []}
    for nombre, obj in OBJETIVO_N1.items():
        fila = f"  {nombre:<16}{obj:>9.3f}"
        for clave in ("ed", "op"):
            r = todo[clave]["nivel1"].get(nombre)
            if r is None:
                fila += f"{'-':>12}{'-':>10}"
                continue
            v = r["recall"]
            d = v - obj
            difs[clave].append(abs(d))
            fila += f"{v:>12.4f}{d:>+10.4f}"
        print(fila)

    print(f"\n  Nivel 2: EQTransformer con IoU >= {ed.IOU_MIN:g}")
    print(f"  {'metrica':<16}{'citado':>9}{'config ed':>12}{'dif':>10}"
          f"{'config op':>12}{'dif':>10}")
    for met, obj in OBJETIVO_N2_EQT.items():
        fila = f"  {met:<16}{obj:>9.3f}"
        for clave in ("ed", "op"):
            r = todo[clave]["nivel2"].get("eqtransformer")
            if r is None:
                fila += f"{'-':>12}{'-':>10}"
                continue
            v = r[met]
            d = v - obj
            difs[clave].append(abs(d))
            fila += f"{v:>12.4f}{d:>+10.4f}"
        print(fila)

    print()
    elegida = None
    for clave in ("op", "ed"):
        if not difs[clave]:
            continue
        peor = max(difs[clave])
        ok = peor <= TOL_REPRODUCE
        print(f"  Configuracion '{clave}': diferencia maxima {peor:.2e}  "
              f"-> {'REPRODUCE' if ok else 'NO reproduce'} "
              f"(tolerancia {TOL_REPRODUCE:.0e})")
        if ok and elegida is None:
            elegida = clave
    return elegida


# =====================================================================
# REPORTE POR TRAMO
# =====================================================================

def ambitos(a, valida):
    """Lista de (nombre, ini, fin, horas validas) global y por tramo."""
    n = a.shape[-1]
    out = [("global", 0, n, float(valida.sum()) / FS / 3600.0)]
    for k, (t0, t1) in enumerate(ed.segmentos(n), start=1):
        out.append((f"tramo {k}", t0, t1,
                    float(valida[t0:t1].sum()) / FS / 3600.0))
    return out


def reportar_nivel1(todo, cfg, catalogo, amb):
    tol = cfg["tol_onset_s"]
    print("\n" + "=" * 78)
    print(f"  NIVEL 1 - DETECCION DEL INSTANTE (tolerancia +-{tol:g} s)")
    print("=" * 78)
    print(f"\n  {'ambito':<10}{'detector':<15}{'cat':>5}{'onsets':>8}"
          f"{'recall':>9}{'prec':>8}{'falsos/h':>10}{'err.MAE_s':>11}")
    for nombre_amb, t0, t1, horas in amb:
        cat = [e for e in catalogo if t0 <= e.ini < t1]
        for det in ("stalta", "phasenet", "eqtransformer"):
            ons = todo["onsets"].get(det)
            if ons is None:
                continue
            o = [t for t in ons if t0 <= t < t1]
            r = ed.metricas_onset(cat, o, horas, tolerancias=[tol])
            d = r[f"tol_{tol:g}s"]
            print(f"  {nombre_amb:<10}{det:<15}{len(cat):>5}{len(o):>8}"
                  f"{d['recall']:>9.3f}{d['precision']:>8.3f}"
                  f"{d['onsets_falsos_por_hora']:>10.2f}"
                  f"{d['error_mae_s']:>11.2f}")
        print()

    print("  Recall de nivel 1 por tramo y clase")
    for det in ("stalta", "phasenet", "eqtransformer"):
        ons = todo["onsets"].get(det)
        if ons is None:
            continue
        print(f"\n    {det}")
        print("      ambito  " + "".join(f"{c:>8}" for c in ed.CLASES))
        for nombre_amb, t0, t1, horas in amb:
            cat = [e for e in catalogo if t0 <= e.ini < t1]
            o = [t for t in ons if t0 <= t < t1]
            r = ed.metricas_onset(cat, o, horas, tolerancias=[tol])
            pc = r[f"tol_{tol:g}s"]["por_clase"]
            print(f"      {nombre_amb:<8}" + "".join(
                f"{pc[c]['recall']:>8.3f}" if c in pc else f"{'-':>8}"
                for c in ed.CLASES))


def reportar_nivel2(todo, catalogo, amb):
    print("\n" + "=" * 78)
    print(f"  NIVEL 2 - CALIDAD DEL RECORTE (IoU >= {ed.IOU_MIN:g})")
    print("=" * 78)
    print(f"\n  {'ambito':<10}{'detector':<15}{'cat':>5}{'det':>6}"
          f"{'recall':>9}{'prec':>8}{'F1':>8}{'FP/h':>8}{'IoU med':>9}")
    for nombre_amb, t0, t1, horas in amb:
        cat = [e for e in catalogo if t0 <= e.ini < t1]
        for det in ("stalta", "eqtransformer"):
            dd = todo["detecciones"].get(det)
            if dd is None:
                continue
            d_amb = [x for x in dd if t0 <= x.ini < t1]
            r, *_ = ed.metricas(cat, d_amb, horas, ed.IOU_MIN)
            print(f"  {nombre_amb:<10}{det:<15}{len(cat):>5}{len(d_amb):>6}"
                  f"{r['recall']:>9.3f}{r['precision']:>8.3f}{r['f1']:>8.3f}"
                  f"{r['fp_por_hora']:>8.2f}{r['iou_medio']:>9.3f}")
        print()

    print("  Recall e IoU de nivel 2 por tramo y clase")
    for det in ("stalta", "eqtransformer"):
        dd = todo["detecciones"].get(det)
        if dd is None:
            continue
        print(f"\n    {det}   (recall / IoU medio; '-' = sin eventos)")
        print("      ambito  " + "".join(f"{c:>15}" for c in ed.CLASES))
        for nombre_amb, t0, t1, horas in amb:
            cat = [e for e in catalogo if t0 <= e.ini < t1]
            d_amb = [x for x in dd if t0 <= x.ini < t1]
            r, *_ = ed.metricas(cat, d_amb, horas, ed.IOU_MIN)
            pc = r["por_clase"]
            celdas = []
            for c in ed.CLASES:
                if c not in pc:
                    celdas.append(f"{'-':>15}")
                else:
                    iou = pc[c]["iou_medio"]
                    txt = (f"{pc[c]['recall']:.3f} / "
                           + ("  nan" if np.isnan(iou) else f"{iou:.3f}"))
                    celdas.append(f"{txt:>15}")
            print(f"      {nombre_amb:<8}" + "".join(celdas))


def reportar_eqt_largas(todo, catalogo, amb, n):
    """Detecciones de EQTransformer mas largas que la ventana fija."""
    dd = todo["detecciones"].get("eqtransformer")
    print("\n" + "=" * 78)
    print(f"  EQTRANSFORMER: DETECCIONES MAS LARGAS QUE LA VENTANA FIJA DE "
          f"{br.VENTANA_FIJA_S:g} s")
    print("=" * 78)
    if dd is None:
        print("  sin cacha de Detection")
        return
    print(f"\n  {'ambito':<10}{'det':>6}{'>81,92 s':>10}{'frac':>8}"
          f"{'dur.max_s':>11}{'TR cat':>8}{'TR recall':>11}{'TR IoU':>9}")
    for nombre_amb, t0, t1, horas in amb:
        cat = [e for e in catalogo if t0 <= e.ini < t1]
        d_amb = [x for x in dd if t0 <= x.ini < t1]
        largas = [x for x in d_amb if x.dur / FS > br.VENTANA_FIJA_S]
        r, *_ = ed.metricas(cat, d_amb, horas, ed.IOU_MIN)
        tr = r["por_clase"].get("TR")
        dur_max = max((x.dur / FS for x in d_amb), default=float("nan"))
        frac = len(largas) / len(d_amb) if d_amb else float("nan")
        if tr is None:
            tr_txt = f"{0:>8}{'-':>11}{'-':>9}"
        else:
            iou = tr["iou_medio"]
            tr_txt = (f"{tr['n']:>8}{tr['recall']:>11.3f}"
                      + (f"{'nan':>9}" if np.isnan(iou) else f"{iou:>9.3f}"))
        print(f"  {nombre_amb:<10}{len(d_amb):>6}{len(largas):>10}"
              f"{frac:>8.3f}{dur_max:>11.1f}{tr_txt}")

    largas = sorted((x for x in dd if x.dur / FS > br.VENTANA_FIJA_S),
                    key=lambda x: -x.dur)
    print(f"\n  Detalle de las {len(largas)} detecciones de mas de "
          f"{br.VENTANA_FIJA_S:g} s")
    if not largas:
        print("    ninguna")
        return
    print(f"    {'tr':>3}{'ini_s':>11}{'fin_s':>11}{'dur_s':>9}  catalogo solapado")
    segs = ed.segmentos(n)
    for x in largas:
        k, t0 = next(((i, s) for i, (s, e) in enumerate(segs, start=1)
                      if s <= x.ini < e), (0, 0))
        sol = [(ed.iou(e, x), e) for e in catalogo
               if e.ini < x.fin and x.ini < e.fin]
        if not sol:
            txt = "sin solape"
        else:
            sol.sort(key=lambda p: -p[0])
            txt = ", ".join(f"{e.clase} {e.dur / FS:.0f} s IoU {v:.3f}"
                            for v, e in sol[:4])
            if len(sol) > 4:
                txt += f", +{len(sol) - 4} mas"
        print(f"    {k:>3}{(x.ini - t0) / FS:>11.2f}{(x.fin - t0) / FS:>11.2f}"
              f"{x.dur / FS:>9.2f}  {txt}")


# =====================================================================
# MAIN
# =====================================================================

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--canales", nargs="+", type=int, default=ed.CANALES)
    ap.add_argument("--sin-cache", action="store_true",
                    help="recalcula STA/LTA en vez de leer su cacha")
    args = ap.parse_args(argv)

    print("Cargando traza y catalogo ...")
    a, catalogo, valida, horas = ed.cargar()
    print(f"  {len(catalogo)} eventos, {horas:.2f} h validas, "
          f"canales {args.canales}\n")

    cfgs = configuraciones()
    todo = {}
    for clave in ("ed", "op"):
        print(f"  Reconstruyendo salidas con la configuracion '{clave}' ...")
        todo[clave] = globales(cfgs[clave], a, catalogo, valida, horas,
                               args.canales, not args.sin_cache)
    print()

    elegida = verificar(todo, cfgs)
    if elegida is None:
        print("\n  Ninguna configuracion reproduce los valores citados.")
        print("  Me detengo aqui: no reporto por tramo sobre una base que no")
        print("  se puede empatar con lo que dice el documento.")
        return 1

    print(f"\n  Reporto por tramo con la configuracion '{elegida}'.")
    amb = ambitos(a, valida)
    print("\n  Ambitos")
    for nombre_amb, t0, t1, h in amb:
        print(f"    {nombre_amb:<10} muestras [{t0}, {t1})  {h:.2f} h validas")

    reportar_nivel1(todo[elegida], cfgs[elegida], catalogo, amb)
    reportar_nivel2(todo[elegida], catalogo, amb)
    reportar_eqt_largas(todo[elegida], catalogo, amb, a.shape[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
