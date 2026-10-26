#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elige con una misma regla la configuracion de cada red a partir de la grilla y mide con ella
todo lo que sigue: deteccion por clase, onda que detecta, recortes, IoU y PhaseNet + EQTransformer.
STA/LTA queda con la configuracion de la Etapa 2.

Regla: se comparan dos opciones, solo umbrales (ventanas por defecto) o umbrales y ventanas;
se queda la que da mas sensibilidad en la rotacion de tramos (empate: menos FP/h; luego, solo
umbrales). Dentro de ella, con la traza completa, la combinacion con menos FP/h que llega a
0,95; si ninguna llega, la de mayor sensibilidad.

Uso: python configuracion_fre3c.py [--figura]   (antes: grilla_fre3c.py)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "evaluacion"))
import fre3c_lib as lib
import evaluar_detectores as ed
import recorte_fre3c as rc
import iou_fre3c as iu
import recorte_detalle_fre3c as rd
import interseccion_fre3c as ix

FS = lib.FS
FILTRO = "f1_15"
FLUJOS = {"PhaseNet P y S": "phasenet", "EQTransformer deteccion": "eqtransformer",
          "EQTransformer P y S": "eqtransformer"}
CLAVES = ["overlap", "blinding", "stacking", "P_threshold", "S_threshold", "detection_threshold"]
CLASES_INF = lib.VOLCANICAS + ["volcanico", "AV", "IC"]
UMB_IOU = iu.UMBRALES_IOU


def todos(df):
    return df[df.tramo.astype(str) == "todos"]


def elegir_configuraciones(df):
    """Aplica la regla a cada flujo de la grilla."""
    out, filas = {}, []
    for flujo in FLUJOS:
        g = df[df.detector == flujo]
        opciones = {"solo umbrales": g[g.por_defecto], "umbrales y ventanas": g}
        rot = {}
        for nombre, h in opciones.items():
            dr = ix.rotar(h, CLAVES)
            s = dr[(dr.criterio == "meta") & (dr.tramo_medido.astype(str) == "suma")].iloc[0]
            rot[nombre] = (s.k_medido / s.n_medido, s.fp_h_medido, int(s.k_medido))
        orden = sorted(rot, key=lambda n: (-rot[n][0], rot[n][1], n != "solo umbrales"))
        opcion = orden[0]
        e = ix.elegir(todos(opciones[opcion]), "meta")
        cfg = {k: (None if isinstance(e[k], float) and np.isnan(e[k]) else
                   (e[k] if k == "stacking" else float(e[k]))) for k in CLAVES}
        for k in ("overlap", "blinding"):
            cfg[k] = int(cfg[k])
        cfg["por_defecto"] = bool(e.por_defecto)
        out[flujo] = cfg
        filas.append(dict(flujo=flujo, opcion=opcion,
                          rot_solo_umbrales=rot["solo umbrales"], rot_ventanas=rot["umbrales y ventanas"],
                          k=int(e.k_volc), sens=e.sens_volc, fp_h=e.fp_h, **cfg))
    return out, filas


def curvas_de(modelo, cfg, cache):
    """Curvas de la configuracion: las guardadas si son las ventanas por defecto; si no, se anota."""
    clave = (modelo, cfg["overlap"], cfg["blinding"], cfg["stacking"])
    if clave in cache:
        return cache[clave]
    if cfg["por_defecto"]:
        c = dict(np.load(lib.ruta_anot(modelo, FILTRO)))
    else:
        import anotar_fre3c as an
        import grilla_fre3c as gr
        z = np.load(lib.ruta_datos(FILTRO))
        c = gr.anotar(an.cargar_modelo(modelo), {k: z[k] for k in ("Z", "N", "E")},
                      cfg["overlap"], cfg["blinding"], cfg["stacking"])
    cache[clave] = c
    return c


def intervalos_flujo(flujo, curvas, cfg):
    if flujo.endswith("deteccion"):
        return ix.intervalos_fase(curvas, "Detection", cfg["detection_threshold"], {})
    return ix.intervalos_ps(curvas, cfg["P_threshold"], cfg["S_threshold"], {})


def fila_clases(nombre, res, inc, horas):
    fc, ft = lib.resumir(res, inc, horas["todos"])
    c = {f["clase"]: f for f in fc}
    return dict(detector=nombre, **{f"k_{k}": c[k]["k"] for k in CLASES_INF},
                **{f"n_{k}": c[k]["n"] for k in CLASES_INF},
                **{f"lo_{k}": c[k]["ic_inf"] for k in CLASES_INF},
                **{f"hi_{k}": c[k]["ic_sup"] for k in CLASES_INF},
                fp_h=ft["fp_h"], fdr=ft["fdr"], intervalos=int(res["onsets"].size))


def onda_por_evento(curvas, up, us, inc, zona, valida):
    """Volcanicos que detecta la P sola, la S sola, las dos o ninguna (cada una con su umbral)."""
    pi, pf = ix.intervalos_fase(curvas, "P", up, {})
    si, sf = ix.intervalos_fase(curvas, "S", us, {})
    dp = ix.evaluar_interseccion(pi, pf, inc, zona, valida)["asignado"] >= 0
    ds = ix.evaluar_interseccion(si, sf, inc, zona, valida)["asignado"] >= 0
    out = []
    for cl in lib.VOLCANICAS:
        m = (inc.clase == cl).to_numpy()
        out.append(dict(clase=cl, ambas=int((dp & ds & m).sum()), solo_p=int((dp & ~ds & m).sum()),
                        solo_s=int((~dp & ds & m).sum()), ninguna=int((~dp & ~ds & m).sum()),
                        k_p=int((dp & m).sum()), k_s=int((ds & m).sum())))
    return out


def resumen_iou(pe, nombre, fp_h):
    v = pe[(pe.recorte == nombre) & pe.clase.isin(lib.VOLCANICAS)]
    ok = v[v.iou >= 0.3]
    return dict(recorte=nombre, **{f"k30_{c}": int((ok.clase == c).sum()) for c in lib.VOLCANICAS},
                k30=len(ok), iou_medio=ok.iou.mean() if len(ok) else np.nan,
                dur_mediana_s=v[v.iou > 0].dur_recorte_s.median(), fp_h=fp_h,
                toca=int((v.iou > 0).sum()), **{f"u{u:.1f}": int((v.iou >= u).sum()) for u in UMB_IOU})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figura", action="store_true")
    args = ap.parse_args()

    ruta_grilla = os.path.join(lib.DIR_SALIDA, "grilla_barrido.csv")
    if not os.path.exists(ruta_grilla):
        raise SystemExit("Falta grilla_barrido.csv: correr antes grilla_fre3c.py")
    grilla = pd.read_csv(ruta_grilla)
    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    inc = eventos[eventos.incluido].reset_index(drop=True)
    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    zona = ix.unir(con_senal.ini.to_numpy(), con_senal.fin.to_numpy())
    z = np.load(lib.ruta_datos(FILTRO))
    valida = z["valida"]
    horas = ix.horas_por_tramo(valida)
    n_volc = int(inc.clase.isin(lib.VOLCANICAS).sum())

    # 0. regla de eleccion
    cfgs, filas_e = elegir_configuraciones(grilla)
    print("0. CONFIGURACION DE CADA RED (misma regla para todas; STA/LTA: configuracion de la Etapa 2)")
    print("   rotacion = suma de los tres tramos medidos con parametros elegidos en los otros dos")
    for f in filas_e:
        a, b = f["rot_solo_umbrales"], f["rot_ventanas"]
        print(f"   {f['flujo']}: rotacion solo umbrales {a[2]}/{n_volc} = {a[0]:.3f} ({a[1]:.1f} FP/h); "
              f"umbrales y ventanas {b[2]}/{n_volc} = {b[0]:.3f} ({b[1]:.1f} FP/h) -> {f['opcion']}")
        serie = pd.Series({k: (np.nan if v is None else v) for k, v in f.items()})
        print(f"     elegida: {ix.fmt_umbral(serie)} -> {f['k']}/{n_volc} = {f['sens']:.3f}, "
              f"{f['fp_h']:.1f} FP/h")
    with open(os.path.join(lib.DIR_SALIDA, "configuracion_elegida.json"), "w") as fh:
        json.dump(cfgs, fh, indent=2)

    cache = {}
    curvas = {f: curvas_de(m, cfgs[f], cache) for f, m in FLUJOS.items()}
    ivs = {f: intervalos_flujo(f, curvas[f], cfgs[f]) for f in FLUJOS}
    ivs["STA/LTA"] = ix.intervalos_stalta(FILTRO)

    # 1. deteccion por clase
    print("\n1. DETECCION POR CLASE CON INTERSECCION (configuracion elegida; IC 95 % de Wilson)")
    res, filas_c = {}, []
    for det, (pi, pf) in ivs.items():
        res[det] = ix.evaluar_interseccion(pi, pf, inc, zona, valida)
        filas_c.append(fila_clases(det, res[det], inc, horas))
    for f in filas_c:
        if f["detector"] in cfgs:
            g = todos(grilla)
            cfg = cfgs[f["detector"]]
            m = (g.detector == f["detector"])
            for k in CLAVES:
                m &= (g[k].astype(str) == str(cfg[k])) if k == "stacking" else (
                    g[k].isna() if cfg[k] is None else np.isclose(g[k], cfg[k]))
            igual = m.any() and int(g[m].iloc[0].k_volc) == f["k_volcanico"] and np.isclose(g[m].iloc[0].fp_h, f["fp_h"])
            print(f"   comprobacion {f['detector']}: igual a la grilla: {'SI' if igual else 'NO'}")
    print(f"   {'detector':<25} " + " ".join(f"{c:>21}" for c in CLASES_INF) + f" {'FP/h':>6} {'FDR':>5}")
    for f in filas_c:
        celdas = [f"{f['k_' + c]}/{f['n_' + c]} [{f['lo_' + c]:.3f};{f['hi_' + c]:.3f}]" for c in CLASES_INF]
        print(f"   {f['detector']:<25} " + " ".join(f"{x:>21}" for x in celdas) + f" {f['fp_h']:>6.1f} {f['fdr']:>5.2f}")

    # 2. onda que detecta
    print("\n2. ONDA QUE DETECTA (P sola, S sola y juntas, cada una con su umbral)")
    filas_o = []
    for flujo in ("PhaseNet P y S", "EQTransformer P y S"):
        cfg = cfgs[flujo]
        up, us = cfg["P_threshold"], cfg["S_threshold"]
        for nombre, (pi, pf) in (("solo P", ix.intervalos_fase(curvas[flujo], "P", up, {})),
                                 ("solo S", ix.intervalos_fase(curvas[flujo], "S", us, {}))):
            r = ix.evaluar_interseccion(pi, pf, inc, zona, valida)
            f = fila_clases(f"{flujo.split()[0]}, {nombre}", r, inc, horas)
            filas_c.append(f)
            print(f"   {f['detector']:<25} VT {f['k_VT']}/{f['n_VT']} LP {f['k_LP']}/{f['n_LP']} TR {f['k_TR']}/{f['n_TR']} "
                  f"volc {f['k_volcanico']}/{n_volc} = {f['k_volcanico'] / n_volc:.3f} {f['fp_h']:.1f} FP/h")
        for x in onda_por_evento(curvas[flujo], up, us, inc, zona, valida):
            filas_o.append(dict(flujo=flujo, **x))
            print(f"     {x['clase']}: ambas {x['ambas']}, solo P {x['solo_p']}, solo S {x['solo_s']}, "
                  f"ninguna {x['ninguna']}")
    pd.DataFrame(filas_c).to_csv(os.path.join(lib.DIR_SALIDA, "config_clases.csv"), index=False)
    pd.DataFrame(filas_o).to_csv(os.path.join(lib.DIR_SALIDA, "config_ondas.csv"), index=False)

    # 3. recortes
    cpn = cfgs["PhaseNet P y S"]
    ceq = cfgs["EQTransformer deteccion"]
    env = rc.envolvente(z["Z"], z["N"], z["E"])
    o = rd.onsets_ps(curvas["PhaseNet P y S"], cpn["P_threshold"], cpn["S_threshold"], valida)
    di, dfi = ivs["EQTransformer deteccion"]
    si, sfi = ivs["STA/LTA"]
    m_eq, m_st = valida[di], valida[si]
    fp = {r["detector"]: r["fp_h"] for r in filas_c}
    recs = {
        "PhaseNet + energia con corte": (rc.recortes(o, env, valida, rc.K_RUIDO, corte=True), fp["PhaseNet P y S"]),
        "PhaseNet + energia": (rc.recortes(o, env, valida, rc.K_RUIDO), fp["PhaseNet P y S"]),
        "PhaseNet + ventana fija": (rc.recortes(o, env, valida, fija=True), fp["PhaseNet P y S"]),
        "EQTransformer, propio": (list(zip(di[m_eq].tolist(), dfi[m_eq].tolist())), fp["EQTransformer deteccion"]),
        "STA/LTA, propio": (list(zip(si[m_st].tolist(), sfi[m_st].tolist())), fp["STA/LTA"]),
        "EQTransformer, propio unido a 5 s": (rc.fusionar_por_tramo(list(zip(di.tolist(), dfi.tolist()))),
                                              fp["EQTransformer deteccion"]),
        "STA/LTA, propio unido a 5 s": (rc.recortes_stalta(rc.cft_stalta(z["Z"]), valida, ed.STALTA["thr_on"]),
                                        fp["STA/LTA"]),
    }
    recs["EQTransformer, propio unido a 5 s"] = ([iv for iv in recs["EQTransformer, propio unido a 5 s"][0]
                                                  if valida[iv[0]]], fp["EQTransformer deteccion"])
    pe = pd.DataFrame([f for n, (v, _) in recs.items() for f in iu.por_evento(n, v, inc)])
    volc = pe[pe.clase.isin(lib.VOLCANICAS)]
    res_iou = [resumen_iou(pe, n, f) for n, (_, f) in recs.items()]

    print(f"\n3. RECORTES CON LA CONFIGURACION ELEGIDA (volcanicos de {n_volc}; un recorte por evento)")
    print("   EQTransformer y STA/LTA: intervalo tal como lo entrega la libreria; 'unido a 5 s': como en la Etapa 2")
    print(f"   {'recorte':<34} {'VT':>4} {'LP':>4} {'TR':>4} {'IoU>=0.3':>8} {'IoU medio':>9} {'dur med':>7} {'FP/h':>6}")
    for r in res_iou:
        print(f"   {r['recorte']:<34} {r['k30_VT']:>4} {r['k30_LP']:>4} {r['k30_TR']:>4} {r['k30']:>8} "
              f"{r['iou_medio']:>9.2f} {r['dur_mediana_s']:>7.1f} {r['fp_h']:>6.1f}")
    print(f"\n   {'recorte':<34} {'toca':>5} " + " ".join(f"{u:>5.1f}" for u in UMB_IOU))
    for r in res_iou:
        print(f"   {r['recorte']:<34} {r['toca']:>5} " + " ".join(f"{r[f'u{u:.1f}']:>5}" for u in UMB_IOU))
    print("\n   por clase: toca, IoU >= 0,3 / 0,5 / 0,7 e IoU mediano de los que se tocan")
    for n in recs:
        for cl in lib.VOLCANICAS:
            v = pe[(pe.recorte == n) & (pe.clase == cl)].iou
            med = v[v > 0].median() if (v > 0).any() else np.nan
            print(f"   {n:<34} {cl:>3} ({len(v)}) toca {int((v > 0).sum()):>3} | "
                  + " ".join(f"{int((v >= u).sum()):>3}" for u in (0.3, 0.5, 0.7)) + f" | {med:.2f}")
    print("\n   cuatro casos de cruce: cantidad (IoU mediano)")
    for n in recs:
        g = volc[volc.recorte == n]
        celdas = []
        for c in iu.CASOS:
            x = g[g.caso == c].iou
            celdas.append(f"{c}: {len(x)}" + (f" ({x.median():.2f})" if len(x) else ""))
        celdas.append(f"sin cruce: {int((g.caso == 'sin cruce').sum())}")
        print(f"   {n:<34} " + "; ".join(celdas))
    pe.to_csv(os.path.join(lib.DIR_SALIDA, "config_iou_por_evento.csv"), index=False)
    pd.DataFrame(res_iou).to_csv(os.path.join(lib.DIR_SALIDA, "config_recortes.csv"), index=False)

    # 4. recorte propio de EQTransformer segun detection_threshold, con sus ventanas elegidas
    print(f"\n4. EQTRANSFORMER: RECORTE PROPIO SEGUN detection_threshold (overlap {ceq['overlap']}, "
          f"blinding {ceq['blinding']}, stacking {ceq['stacking']}; intervalo sin unir)")
    print(f"   {'umbral':>6} {'toca':>5} {'0.3':>5} {'0.5':>5} {'IoU mediano':>12} {'dur med (s)':>11} {'FP/h':>6}")
    filas_u = []
    for u in lib.UMBRALES:
        a, b = ix.intervalos_fase(curvas["EQTransformer deteccion"], "Detection", u, {})
        mm = valida[a]
        v = pd.DataFrame(iu.por_evento("eqt", list(zip(a[mm].tolist(), b[mm].tolist())), inc))
        v = v[v.clase.isin(lib.VOLCANICAS)]
        r = ix.evaluar_interseccion(a, b, inc, zona, valida)
        _, ft = lib.resumir(r, inc, horas["todos"])
        fila = dict(detection_threshold=u, toca=int((v.iou > 0).sum()), k30=int((v.iou >= 0.3).sum()),
                    k50=int((v.iou >= 0.5).sum()),
                    iou_mediano=v[v.iou > 0].iou.median() if (v.iou > 0).any() else np.nan,
                    dur_mediana_s=v[v.iou > 0].dur_recorte_s.median(), fp_h=ft["fp_h"])
        filas_u.append(fila)
        print(f"   {u:>6.2f} {fila['toca']:>5} {fila['k30']:>5} {fila['k50']:>5} {fila['iou_mediano']:>12.2f} "
              f"{fila['dur_mediana_s']:>11.1f} {fila['fp_h']:>6.1f}")
    pd.DataFrame(filas_u).to_csv(os.path.join(lib.DIR_SALIDA, "config_iou_eqt_umbral.csv"), index=False)

    # 5. recorte por energia: periodo de ruido y percentil, con los avisos de la configuracion elegida
    print("\n5. RECORTE POR ENERGIA CON CORTE: PERIODO DE RUIDO Y PERCENTIL")
    print(f"   {'ruido (s antes)':>16} {'percentil':>9} {'toca':>5} {'0.3':>5} {'0.5':>5} {'IoU med':>8} "
          f"{'TR 0.3':>7} {'TR IoU med':>11}")
    original = (rc.RUIDO_S, rc.PERCENTIL_RUIDO)
    filas_r = []
    for desde, hasta in rd.RUIDOS_S:
        for p in rd.PERCENTILES:
            rc.RUIDO_S, rc.PERCENTIL_RUIDO = (float(desde), float(hasta)), p
            mm = rd.medir(rc.recortes(o, env, valida, rc.K_RUIDO, corte=True), inc)
            filas_r.append(dict(ruido_desde_s=desde, ruido_hasta_s=hasta, percentil=p, **mm))
            marca = "  <- en uso" if (float(desde), float(hasta)) == original[0] and p == original[1] else ""
            print(f"   {f'{desde:g} a {hasta:g}':>16} {p:>9} {mm['toca']:>5} {mm['k30']:>5} {mm['k50']:>5} "
                  f"{mm['iou_mediano']:>8.2f} {mm['tr_k30']:>7} {mm['tr_iou_mediano']:>11.2f}{marca}")
    rc.RUIDO_S, rc.PERCENTIL_RUIDO = original
    pd.DataFrame(filas_r).to_csv(os.path.join(lib.DIR_SALIDA, "config_recorte_ruido.csv"), index=False)

    # 6. inicio del recorte segun la onda que dio el aviso
    _, dio = rd.inicio_por_onda(curvas["PhaseNet P y S"], cpn["P_threshold"], cpn["S_threshold"], env, valida, inc)
    print("\n6. INICIO DEL RECORTE POR ENERGIA SEGUN LA ONDA QUE DIO EL AVISO")
    print(f"   {'onda':>4} {'clase':>5} {'eventos':>7} {'empieza antes':>13} {'empieza despues':>15} "
          f"{'aviso tras inicio, mediana (s)':>31} {'IoU med':>8}")
    for onda in ("P", "S", "?"):
        for cl in lib.VOLCANICAS + ["todas"]:
            g = dio[dio.onda == onda] if cl == "todas" else dio[(dio.onda == onda) & (dio.clase == cl)]
            if g.empty:
                continue
            print(f"   {onda:>4} {cl:>5} {len(g):>7} {int((g.inicio_rel_s <= 0).sum()):>13} "
                  f"{int((g.inicio_rel_s > 0).sum()):>15} {g.aviso_rel_s.median():>31.1f} {g.iou.median():>8.2f}")
    dio.to_csv(os.path.join(lib.DIR_SALIDA, "config_inicio_onda.csv"), index=False)

    # 7. PhaseNet y EQTransformer juntos: misma agrupacion de 5 s que P y S
    print("\n7. PHASENET Y EQTRANSFORMER JUNTOS (intervalos agrupados con la regla de 5 s)")
    a = np.concatenate([ivs["PhaseNet P y S"][0], ivs["EQTransformer deteccion"][0]])
    b = np.concatenate([ivs["PhaseNet P y S"][1], ivs["EQTransformer deteccion"][1]])
    ji, jf = ix.agrupar_intervalos(a, b)
    rj = ix.evaluar_interseccion(ji, jf, inc, zona, valida)
    fj = fila_clases("PhaseNet + EQTransformer", rj, inc, horas)
    nuevos = (rj["asignado"] >= 0) & (res["PhaseNet P y S"]["asignado"] < 0) & inc.clase.isin(lib.VOLCANICAS).to_numpy()
    perdidos = (rj["asignado"] < 0) & (res["PhaseNet P y S"]["asignado"] >= 0) & inc.clase.isin(lib.VOLCANICAS).to_numpy()
    print(f"   VT {fj['k_VT']}/{fj['n_VT']} LP {fj['k_LP']}/{fj['n_LP']} TR {fj['k_TR']}/{fj['n_TR']} volc {fj['k_volcanico']}/{n_volc} = "
          f"{fj['k_volcanico'] / n_volc:.3f}, {fj['fp_h']:.1f} FP/h; nuevos frente a PhaseNet solo: "
          f"{ {k: int(v) for k, v in inc[nuevos].clase.value_counts().items()} }; perdidos: {int(perdidos.sum())}")
    pd.DataFrame([fj]).to_csv(os.path.join(lib.DIR_SALIDA, "config_juntos.csv"), index=False)

    if args.figura:
        iu.figura(volc[volc.recorte.isin(iu.RECORTES)], n_volc)
    print(f"\nTablas config_*.csv y configuracion_elegida.json en {lib.DIR_SALIDA}")


if __name__ == "__main__":
    main()
