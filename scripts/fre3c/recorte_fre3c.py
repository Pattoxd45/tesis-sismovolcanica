#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recorte de PhaseNet por energia: parte 2,5 s antes del arribo y termina cuando la
envolvente de Z, N y E vuelve al nivel del ruido previo. Evalua el recorte
(IoU >= 0,30) y lo compara con la ventana fija de 81,92 s y con los recortes
propios de EQTransformer y STA/LTA.

Uso: python recorte_fre3c.py [--umbral 0.10] [--figura]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "evaluacion"))
import fre3c_lib as lib
import evaluar_detectores as ed

FS = lib.FS
TOL = int(round(lib.TOL_S * FS))
VENTANA_ENV_S = 1.0
RUIDO_S = (120.0, 5.0)  # ruido: de 120 s a 5 s antes del arribo
PERCENTIL_RUIDO = 20  # robusto a la coda de un evento anterior
K_RUIDO = 2.0
CALMA_S = 5.0
DUR_MIN_S = 2.0
DUR_MAX_S = 300.0
MARGEN_PREVIO_S = 2.5  # cubre el error de inicio medido (1,7 a 2,4 s), como en ventanas/barrido.py
VENTANA_FIJA_S = 81.92
UMBRALES = [0.05, 0.10, 0.30]
KS = [1.5, 2.0, 3.0]
IOU_ALTO = 0.5
UMBRAL_EQT = 0.3  # umbral por defecto de SeisBench
UMBRALES_EQT = [0.1, 0.3, 0.5]
THR_ON_STALTA = [3.0, 4.0, 5.0]
DESPUES_S = (5.0, 65.0)  # nivel despues del evento, solo para el diagnostico


def media_movil(x, w):
    c = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    h = w // 2
    i = np.arange(x.size)
    a = np.clip(i - h, 0, x.size)
    b = np.clip(i + h + 1, 0, x.size)
    return (c[b] - c[a]) / (b - a)


def envolvente(z, n, e, w=int(VENTANA_ENV_S * FS)):
    z, n, e = (np.asarray(v, dtype=np.float64) for v in (z, n, e))
    return np.sqrt(media_movil(z * z + n * n + e * e, w))


def tramo_de(t, limites=lib.LIMITES_TRAMO):
    for a, b in limites:
        if a <= t < b:
            return a, b
    return None


def umbral_ruido(env, valida, t0, k, lo):
    a = max(lo, t0 - int(RUIDO_S[0] * FS))
    b = max(lo, t0 - int(RUIDO_S[1] * FS))
    ruido = env[a:b][valida[a:b]]
    if ruido.size < FS:
        return None
    return k * np.percentile(ruido, PERCENTIL_RUIDO)


def fin_por_energia(env, valida, t0, k=K_RUIDO, limites=lib.LIMITES_TRAMO):
    """Inicio del primer tramo de CALMA_S s con la envolvente bajo k veces el nivel del ruido previo."""
    tr = tramo_de(t0, limites)
    if tr is None:
        return None
    lo, hi = tr
    umbral = umbral_ruido(env, valida, t0, k, lo)
    if umbral is None:
        return None
    fin_max = min(hi, t0 + int(DUR_MAX_S * FS))
    nv = np.flatnonzero(~valida[t0:fin_max])
    if nv.size:
        fin_max = t0 + int(nv[0])
    ini = t0 + int(DUR_MIN_S * FS)
    h = int(CALMA_S * FS)
    if fin_max - ini < h:
        return fin_max
    calma = env[ini:fin_max] < umbral
    c = np.concatenate(([0], np.cumsum(calma)))
    idx = np.flatnonzero(c[h:] - c[:-h] == h)
    return ini + int(idx[0]) if idx.size else fin_max


def recortes(onsets, env, valida, k=K_RUIDO, fija=False, corte=False, limites=lib.LIMITES_TRAMO):
    """corte: el recorte por energia termina antes del siguiente arribo, si llega primero."""
    o = np.sort(np.asarray(onsets, dtype=np.int64))
    m = int(MARGEN_PREVIO_S * FS)
    ivs = []
    for i, t0 in enumerate(o):
        tr = tramo_de(int(t0), limites)
        if tr is None:
            continue
        ini = max(tr[0], int(t0) - m)
        if fija:
            fin = min(tr[1], ini + int(round(VENTANA_FIJA_S * FS)))
        else:
            fin = fin_por_energia(env, valida, int(t0), k, limites)
            if fin is None:
                continue
            if corte and i + 1 < o.size and o[i + 1] < tr[1]:
                fin = min(fin, int(o[i + 1]) - m)
        ivs.append((ini, fin))
    return ivs


def fusionar_por_tramo(ivs, limites=lib.LIMITES_TRAMO):
    gap = int(ed.FUSION_S * FS)
    out = []
    for a, b in limites:
        out += ed.fusionar_canal([iv for iv in ivs if a <= iv[0] < b], gap)
    return out


def recortes_eqt(det, umbral, valida, limites=lib.LIMITES_TRAMO):
    """Intervalos de la curva Detection como los entrega SeisBench, unidos a 5 s como en la Etapa 2."""
    ini, fin, _, _ = lib.disparos_por_tramo(det, umbral, limites)
    ivs = fusionar_por_tramo(list(zip(ini.tolist(), fin.tolist())), limites)
    return [(a, b) for a, b in ivs if valida[a]]


def cft_stalta(z, cfg=ed.STALTA, limites=lib.LIMITES_TRAMO):
    from obspy.signal.filter import bandpass
    from obspy.signal.trigger import classic_sta_lta

    out = []
    for a, b in limites:
        x = bandpass(np.asarray(z[a:b], dtype=np.float64), cfg["banda"][0], cfg["banda"][1],
                     df=FS, corners=4, zerophase=True)
        c = classic_sta_lta(x, int(cfg["sta_s"] * FS), int(cfg["lta_s"] * FS))
        out.append(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0))
    return out


def recortes_stalta(cfts, valida, thr_on, thr_off=ed.STALTA["thr_off"], limites=lib.LIMITES_TRAMO):
    """Encendido y apagado del disparo, unidos a 5 s, igual que detector_stalta de la Etapa 2."""
    from obspy.signal.trigger import trigger_onset

    trig = []
    for (a, _), c in zip(limites, cfts):
        trig += [(a + int(i), a + int(j)) for i, j in trigger_onset(c, thr_on, thr_off)]
    ivs = fusionar_por_tramo(trig, limites)
    return [(a, b) for a, b in ivs if valida[a]]


def onsets_pn_ps(curvas, umbral, valida):
    partes = [lib.disparos_por_tramo(curvas[f], umbral)[2] for f in ("P", "S")]
    o = lib.agrupar_anclado(np.concatenate(partes), TOL)
    return o[valida[o]]


def evaluar_nivel2(ivs, inc, zona_ini, zona_fin, horas, margen_s=MARGEN_PREVIO_S):
    """margen_s: distancia del arribo al inicio del recorte (0 en los recortes propios)."""
    cat = [ed.Intervalo(int(r.ini), int(r.fin), r.clase) for r in inc.itertuples()]
    det = [ed.Intervalo(a, b) for a, b in ivs]
    pares, _, fp = ed.emparejar(cat, det, ed.IOU_MIN)
    altos = {id(g) for g, _, _ in ed.emparejar(cat, det, IOU_ALTO)[0]}
    por_ev = {id(g): (v, d) for g, d, v in pares}
    filas = []
    for clase in lib.CLASES + ["volcanico"]:
        sel = [g for g in cat if (g.clase in lib.VOLCANICAS if clase == "volcanico" else g.clase == clase)]
        hit = [g for g in sel if id(g) in por_ev]
        k, n = len(hit), len(sel)
        k50 = sum(id(g) in altos for g in sel)
        lo, hi = lib.wilson(k, n)
        ious = [por_ev[id(g)][0] for g in hit]
        err_fin = [(por_ev[id(g)][1].fin - g.fin) / FS for g in hit]
        filas.append(dict(clase=clase, k=k, n=n, sens=k / n if n else np.nan, ic_inf=lo, ic_sup=hi,
                          k_iou50=k50, sens_iou50=k50 / n if n else np.nan,
                          iou_medio=float(np.mean(ious)) if ious else np.nan,
                          err_fin_mediano_s=float(np.median(err_fin)) if err_fin else np.nan))
    ext = [d for d in fp if not lib.dentro_de_intervalos([d.ini + int(margen_s * FS)],
                                                         zona_ini, zona_fin)[0]]
    vol = [g for g in cat if g.clase in lib.VOLCANICAS]
    dv = [d.dur / FS for d in det if any(d.ini < g.fin and d.fin > g.ini for g in vol)]
    return filas, dict(recortes=len(det), sin_pareja=len(fp), externos=len(ext),
                       fp_h=len(ext) / horas if horas else np.nan,
                       dur_mediana_volc_s=float(np.median(dv)) if dv else np.nan)


def diagnostico(inc, ev_senal, ivs, onsets, env, valida, limites=lib.LIMITES_TRAMO):
    """Volcanicos sin recorte con IoU >= 0,30: causa y niveles medidos respecto del ruido previo."""
    en_evento = np.zeros(env.size, dtype=bool)
    for r in ev_senal.itertuples():
        en_evento[r.ini:r.fin] = True
    ivs = np.array(ivs, dtype=np.int64).reshape(-1, 2)
    onsets = np.asarray(onsets, dtype=np.int64)
    filas = []
    for r in inc[inc.clase.isin(lib.VOLCANICAS)].itertuples():
        base = dict(id=r.id, clase=r.clase, tramo=r.tramo, dur_s=round(float(r.dur_s), 1))
        sol = (ivs[:, 0] < r.fin) & (ivs[:, 1] > r.ini)
        if not sol.any():
            hay = bool(np.any((onsets >= r.ini - TOL) & (onsets < r.fin)))
            filas.append(dict(base, iou=0.0, causa="sin ruido previo" if hay else "sin arribo"))
            continue
        c = ivs[sol]
        inter = np.minimum(c[:, 1], r.fin) - np.maximum(c[:, 0], r.ini)
        union = np.maximum(c[:, 1], r.fin) - np.minimum(c[:, 0], r.ini)
        j = int(np.argmax(inter / union))
        v = inter[j] / union[j]
        if v >= ed.IOU_MIN:
            continue
        a, b = int(c[j, 0]), int(c[j, 1])
        dur_r = (b - a) / FS
        causa = ("recorte corto" if dur_r < 0.5 * r.dur_s else
                 "recorte largo" if dur_r > 2 * r.dur_s else "recorte desplazado")
        t0 = a + int(MARGEN_PREVIO_S * FS)
        lo = tramo_de(t0, limites)[0]
        ruido = umbral_ruido(env, valida, t0, 1.0, lo) or np.nan
        ra, rb = max(lo, t0 - int(RUIDO_S[0] * FS)), max(lo, t0 - int(RUIDO_S[1] * FS))
        m = valida[ra:rb]
        da, db = r.fin + int(DESPUES_S[0] * FS), r.fin + int(DESPUES_S[1] * FS)
        despues = env[da:db][valida[da:db]]
        otros = int(((ev_senal.ini < b) & (ev_senal.fin > a) & (ev_senal.id != r.id)).sum())
        hi = tramo_de(t0, limites)[1]
        fin_por = ("fin de tramo" if b >= hi else "hueco" if not valida[b] else
                   "tope" if b - t0 >= int(DUR_MAX_S * FS) else "calma")
        filas.append(dict(base, iou=round(float(v), 2), dur_recorte_s=round(dur_r, 1),
                          ini_rel_s=round((a - r.ini) / FS, 1), fin_rel_s=round((b - r.fin) / FS, 1),
                          causa=causa, fin_por=fin_por, rec_ini=a, rec_fin=b,
                          ocup_ruido=round(float(en_evento[ra:rb][m].mean()), 2) if m.any() else np.nan,
                          snr_evento=round(float(np.median(env[r.ini:r.fin]) / ruido), 1),
                          snr_despues=(round(float(np.percentile(despues, PERCENTIL_RUIDO) / ruido), 1)
                                       if despues.size else np.nan),
                          otros_en_recorte=otros))
    return pd.DataFrame(filas)


def estilo(ax, tinta2):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=tinta2, labelsize=8)


def figura(env, datos, inc, ivs_e, ivs_f, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tinta, tinta2, gris = "#0b0b0b", "#52514e", "#e4e3df"
    c_cat, c_ene, c_fija = "#8a8984", "#6250d6", "#eda100"
    cat = [ed.Intervalo(int(r.ini), int(r.fin), r.clase) for r in inc.itertuples()]
    det = [ed.Intervalo(a, b) for a, b in ivs_e]
    pares, _, _ = ed.emparejar(cat, det, ed.IOU_MIN)
    casos = []
    for clase in lib.VOLCANICAS:
        p = sorted([x for x in pares if x[0].clase == clase], key=lambda x: x[2])
        if p:
            casos.append((clase, p[len(p) // 2]))
    if not casos:
        print("Sin aciertos para la figura")
        return
    fig, axs = plt.subplots(len(casos), 1, figsize=(10, 2.6 * len(casos) + 0.8), squeeze=False)
    for ax, (clase, (g, d, v)) in zip(axs[:, 0], casos):
        fija = next(((a, b) for a, b in ivs_f if a == d.ini), None)
        a = max(0, min(g.ini, d.ini) - int(20 * FS))
        b = max(g.fin, d.fin, fija[1] if fija else 0) + int(20 * FS)
        t = (np.arange(a, b) - g.ini) / FS
        z = datos["Z"][a:b]
        e = env[a:b]
        ev = slice(g.ini - a, g.fin - a)
        ax.plot(t, np.clip(z / np.abs(z[ev]).max(), -1.05, 1.05), color=gris, lw=0.6)
        ax.plot(t, np.clip(e / e[ev].max(), 0, 1.05), color=tinta2, lw=1.4, label="envolvente Z, N, E")
        ax.axvspan(0, (g.fin - g.ini) / FS, color=c_cat, alpha=0.25, label="evento del catalogo")
        ax.hlines(-1.15, (d.ini - g.ini) / FS, (d.fin - g.ini) / FS, color=c_ene, lw=4,
                  label="recorte por energia")
        titulo = f"{clase}: evento de {(g.fin - g.ini) / FS:.1f} s | IoU energia {v:.2f}"
        if fija:
            vf = ed.iou(g, ed.Intervalo(*fija))
            ax.hlines(-1.35, (fija[0] - g.ini) / FS, (fija[1] - g.ini) / FS, color=c_fija, lw=4,
                      label="ventana fija 81,92 s")
            titulo += f" | IoU ventana fija {vf:.2f}"
        ax.set_title(titulo.replace(".", ","), fontsize=10, color=tinta, loc="left")
        ax.set_ylim(-1.5, 1.1)
        ax.set_yticks([])
        estilo(ax, tinta2)
    axs[-1, 0].set_xlabel("segundos desde el inicio del catalogo", color=tinta2, fontsize=9)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.suptitle("Recorte de PhaseNet por energia contra ventana fija (FRE, caso mediano por clase)",
                 fontsize=11, color=tinta, x=0.01, ha="left", y=0.995)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(ruta, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Figura: {ruta}")


def figura_fallas(env, valida, eventos, dp, ruta, k=K_RUIDO):
    """Un LP con recorte largo y un TR con recorte corto: envolvente en veces el ruido previo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullLocator

    tinta, tinta2 = "#0b0b0b", "#52514e"
    c_cat, c_ene, c_umb, c_ruido = "#8a8984", "#6250d6", "#eda100", "#c9c8c3"
    casos = []
    for clase, causa in (("LP", "recorte largo"), ("TR", "recorte corto")):
        s = dp[(dp.clase == clase) & (dp.causa == causa)].sort_values("iou")
        if len(s):
            casos.append(s.iloc[len(s) // 2])
    if not casos:
        return
    fig, axs = plt.subplots(len(casos), 1, figsize=(10, 3.1 * len(casos) + 0.8), squeeze=False)
    for ax, f in zip(axs[:, 0], casos):
        r = eventos[eventos.id == f.id].iloc[0]
        a, b = int(f.rec_ini), int(f.rec_fin)
        t0 = a + int(MARGEN_PREVIO_S * FS)
        lo, hi = tramo_de(t0)
        ruido = umbral_ruido(env, valida, t0, 1.0, lo)
        if not ruido:
            continue
        x0 = max(lo, t0 - int((RUIDO_S[0] + 10) * FS))
        x1 = min(hi, max(b, int(r.fin)) + int(30 * FS))
        t = (np.arange(x0, x1) - r.ini) / FS
        y = np.where(valida[x0:x1], env[x0:x1] / ruido, np.nan)
        ax.axvspan((t0 - RUIDO_S[0] * FS - r.ini) / FS, (t0 - RUIDO_S[1] * FS - r.ini) / FS,
                   color=c_ruido, alpha=0.35, label="ventana de ruido previo")
        ax.axvspan(0, r.dur_s, color=c_cat, alpha=0.3, label="evento del catalogo")
        ax.plot(t, y, color=tinta2, lw=1.0, label="envolvente / ruido previo")
        ax.axhline(k, color=c_umb, lw=1.5, ls="--", label=f"umbral de fin ({k:g} veces el ruido)")
        ymin = 0.3
        ax.hlines(ymin * 1.25, (a - r.ini) / FS, (b - r.ini) / FS, color=c_ene, lw=5, label="recorte por energia")
        ax.set_yscale("log")
        ytop = max(10.0, np.nanmax(y) * 1.5)
        ax.set_ylim(ymin, ytop)
        ax.set_yticks([v for v in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if v < ytop])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
        ax.yaxis.set_minor_locator(NullLocator())
        if f.fin_por in ("hueco", "fin de tramo"):
            nota = "el recorte termina en un hueco del registro" if f.fin_por == "hueco" else "el recorte termina con el tramo"
        elif f.causa == "recorte largo":
            nota = f"despues del evento la envolvente queda en {f.snr_despues:.1f} veces el ruido previo"
        else:
            nota = (f"ruido previo {f.ocup_ruido * 100:.0f} % ocupado por eventos; "
                    f"dentro del evento la envolvente es {f.snr_evento:.1f} veces el ruido previo")
        titulo = (f"{f.clase} {f.id:.0f}: evento de {r.dur_s:.1f} s, recorte de {f.dur_recorte_s:.1f} s "
                  f"(IoU {f.iou:.2f})\n{nota}")
        ax.set_title(titulo.replace(".", ","), fontsize=9.5, color=tinta, loc="left")
        ax.set_ylabel("veces el ruido", color=tinta2, fontsize=8.5)
        estilo(ax, tinta2)
        ax.spines["left"].set_visible(True)
    axs[-1, 0].set_xlabel("segundos desde el inicio del catalogo", color=tinta2, fontsize=9)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.suptitle("Por que falla el recorte por energia (FRE, caso mediano de cada falla)",
                 fontsize=11, color=tinta, x=0.01, ha="left", y=0.995)
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(ruta, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"Figura: {ruta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.10, help="umbral de PhaseNet para la tabla y la figura")
    ap.add_argument("--filtro", default="f1_15", choices=list(lib.FILTROS))
    ap.add_argument("--filtro-eqt", default="f1_45", choices=list(lib.FILTROS))
    ap.add_argument("--figura", action="store_true")
    args = ap.parse_args()

    z = np.load(lib.ruta_datos(args.filtro))
    datos = {c: z[c] for c in ("Z", "N", "E")}
    valida = z["valida"]
    env = envolvente(datos["Z"], datos["N"], datos["E"])
    a = np.load(lib.ruta_anot("phasenet", args.filtro))
    curvas = {k: a[k] for k in ("P", "S")}
    det_eqt = np.load(lib.ruta_anot("eqtransformer", args.filtro_eqt))["Detection"]
    cfts = cft_stalta(datos["Z"])
    eventos = pd.read_csv(lib.RUTA_EVENTOS)
    inc = eventos[eventos.incluido].reset_index(drop=True)
    con_senal = eventos[eventos.estado_crudo != "no_coincide"]
    zi, zf = (con_senal.ini - TOL).to_numpy(), con_senal.fin.to_numpy()
    horas = valida.sum() / FS / 3600

    filas = []

    def agregar(detector, u, tipo, k, ivs):
        fc, tot = evaluar_nivel2(ivs, inc, zi, zf, horas, 0.0 if tipo == "propio" else MARGEN_PREVIO_S)
        for f in fc:
            filas.append(dict(detector=detector, umbral=u, recorte=tipo, k_ruido=k, **f, **tot))

    for u in sorted(set(UMBRALES + [args.umbral])):
        o = onsets_pn_ps(curvas, u, valida)
        agregar("phasenet", u, "fija", None, recortes(o, env, valida, fija=True))
        for k in KS:
            agregar("phasenet", u, "energia", k, recortes(o, env, valida, k))
        agregar("phasenet", u, "energia_corte", K_RUIDO, recortes(o, env, valida, K_RUIDO, corte=True))
    for u in UMBRALES_EQT:
        agregar("eqtransformer", u, "propio", None, recortes_eqt(det_eqt, u, valida))
    for u in THR_ON_STALTA:
        agregar("stalta", u, "propio", None, recortes_stalta(cfts, valida, u))
    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(lib.DIR_SALIDA, "nivel2_recorte.csv"), index=False)

    def fila(detector, u, tipo, k, clase):
        s = df[(df.detector == detector) & np.isclose(df.umbral, u) & (df.recorte == tipo) & (df.clase == clase)]
        s = s[s.k_ruido.isna()] if k is None else s[np.isclose(s.k_ruido, k)]
        return s.iloc[0]

    conf = [("PN + energia", ("phasenet", args.umbral, "energia", K_RUIDO)),
            ("PN + E + corte", ("phasenet", args.umbral, "energia_corte", K_RUIDO)),
            ("PN + fija", ("phasenet", args.umbral, "fija", None)),
            ("EQT propio", ("eqtransformer", UMBRAL_EQT, "propio", None)),
            ("STA/LTA propio", ("stalta", ed.STALTA["thr_on"], "propio", None))]
    print("NIVEL 2 EN FRE: RECORTE DE CADA DETECTOR CONTRA EL CATALOGO (IoU >= 0,30)")
    print(f"  PN + energia:   arribo de PhaseNet (P y S, umbral {args.umbral:.2f}, {args.filtro}) "
          f"y fin por energia (k = {K_RUIDO:g})")
    print("  PN + E + corte: igual, pero el recorte termina 2,5 s antes del siguiente arribo de PhaseNet")
    print("  PN + fija:      arribo de PhaseNet y ventana de 81,92 s (la del entrenamiento)")
    print(f"  EQT propio:     intervalo de la curva Detection (umbral {UMBRAL_EQT:.2f}, {args.filtro_eqt})")
    s = ed.STALTA
    print(f"  STA/LTA propio: disparo en Z ({s['sta_s']:g}/{s['lta_s']:g} s, encendido {s['thr_on']:g}, "
          f"apagado {s['thr_off']:g}; optimo de la Etapa 2)")
    print("  celda: aciertos/n (IoU medio de los aciertos)\n")
    print(f"  {'clase':<10}" + "".join(f"{n:>17}" for n, _ in conf))
    for clase in lib.CLASES + ["volcanico"]:
        celdas = []
        for _, c in conf:
            x = fila(*c, clase)
            celdas.append(f"{x['k']:.0f}/{x.n:.0f} ({x.iou_medio:.2f})")
        print(f"  {clase:<10}" + "".join(f"{c:>17}" for c in celdas))

    print("\n  VOLCANICOS (VT, LP y TR):")
    print(f"  {'recorte':<16} {'IoU>=0,3':>9} {'IoU>=0,5':>9} {'IoU medio':>10} {'err_fin_s':>10} "
          f"{'dur_s':>7} {'recortes':>9} {'fuera de evento/h':>18}")
    for nombre, c in conf:
        x = fila(*c, "volcanico")
        print(f"  {nombre:<16} {x.sens:>9.3f} {x.sens_iou50:>9.3f} {x.iou_medio:>10.2f} "
              f"{x.err_fin_mediano_s:>10.1f} {x.dur_mediana_volc_s:>7.1f} {x.recortes:>9.0f} {x.fp_h:>18.1f}")
    dcat = inc[inc.clase.isin(lib.VOLCANICAS)].dur_s.median()
    print("  err_fin_s: fin del recorte - fin del catalogo, mediana de los aciertos")
    print(f"  dur_s: duracion mediana de los recortes que tocan un volcanico (catalogo: {dcat:.1f} s)")

    print("\nROBUSTEZ: sensibilidad volcanica en nivel 2 (IoU medio)")
    print("  PhaseNet, por umbral y recorte:")
    print(f"  {'umbral':>6} {'fija':>13} " + " ".join(f"{'k=' + format(k, 'g'):>13}" for k in KS)
          + f" {'k=2 + corte':>13}")
    for u in UMBRALES:
        celdas = []
        for tipo, k in [("fija", None)] + [("energia", k) for k in KS] + [("energia_corte", K_RUIDO)]:
            x = fila("phasenet", u, tipo, k, "volcanico")
            celdas.append(f"{x.sens:.3f} ({x.iou_medio:.2f})")
        print(f"  {u:>6.2f} " + " ".join(f"{c:>13}" for c in celdas))
    for nombre, det, us in (("EQTransformer, por umbral", "eqtransformer", UMBRALES_EQT),
                            ("STA/LTA, por umbral de encendido", "stalta", THR_ON_STALTA)):
        x = [fila(det, u, "propio", None, "volcanico") for u in us]
        print(f"  {nombre}: " + "   ".join(f"{u:g} -> {v.sens:.3f} ({v.iou_medio:.2f})" for u, v in zip(us, x)))

    o = onsets_pn_ps(curvas, args.umbral, valida)
    dp = diagnostico(inc, con_senal, recortes(o, env, valida), o, env, valida)
    if len(dp):
        print(f"\nVOLCANICOS SIN RECORTE BUENO CON PN + ENERGIA (IoU < 0,30), umbral {args.umbral:.2f}, "
              f"k = {K_RUIDO:g}: {len(dp)}")
        print("  causa por clase:")
        print(pd.crosstab(dp.clase, dp.causa).to_string())
        print("\n  ini_rel/fin_rel: inicio y fin del mejor recorte respecto del evento (s)")
        print("  ocup_ruido: fraccion de la ventana de ruido (120 a 5 s antes del arribo) cubierta por eventos")
        print(f"  snr_evento: mediana de la envolvente dentro del evento / ruido previo (el fin exige < {K_RUIDO:g})")
        print("  snr_despues: percentil 20 de la envolvente de 5 a 65 s tras el evento / ruido previo")
        print("  otros: eventos del catalogo dentro del recorte")
        print("  fin_por: calma (regla normal), tope de 300 s, hueco del registro o fin de tramo\n")
        cols = ["id", "clase", "tramo", "dur_s", "iou", "dur_recorte_s", "ini_rel_s", "fin_rel_s",
                "causa", "fin_por", "ocup_ruido", "snr_evento", "snr_despues", "otros_en_recorte"]
        print(dp.reindex(columns=cols).sort_values(["clase", "causa", "iou"])
              .rename(columns={"otros_en_recorte": "otros"}).to_string(index=False))
        dp.to_csv(os.path.join(lib.DIR_SALIDA, "nivel2_perdidos.csv"), index=False)

        print("\n  LECTURA:")
        co = dp[dp.causa == "recorte corto"]
        if len(co):
            print(f"  recorte corto: {len(co)} | con snr_evento < {K_RUIDO:g}: {(co.snr_evento < K_RUIDO).sum()} | "
                  f"con mas de la mitad del ruido ocupado por eventos: {(co.ocup_ruido > 0.5).sum()}")
        la = dp[dp.causa == "recorte largo"]
        if len(la):
            print(f"  recorte largo: {len(la)} | con snr_despues >= {K_RUIDO:g}: {(la.snr_despues >= K_RUIDO).sum()} | "
                  f"con otro evento dentro del recorte: {(la.otros_en_recorte > 0).sum()} | "
                  f"en el tope de {DUR_MAX_S:g} s: {(la.dur_recorte_s >= DUR_MAX_S + MARGEN_PREVIO_S - 0.05).sum()}")
        print("  como termina el recorte: " + ", ".join(f"{c}: {n}" for c, n in dp.fin_por.value_counts().items()))
        print("  por tramo: " + ", ".join(f"tramo {t}: {n}" for t, n in dp.tramo.value_counts().sort_index().items()))

    if args.figura:
        figura(env, datos, inc, recortes(o, env, valida), recortes(o, env, valida, fija=True),
               os.path.join(lib.DIR_SALIDA, "fig5_recortes.png"))
        if len(dp):
            figura_fallas(env, valida, eventos, dp, os.path.join(lib.DIR_SALIDA, "fig6_fallas.png"))
    print(f"\nTabla completa en {lib.DIR_SALIDA}/nivel2_recorte.csv")


if __name__ == "__main__":
    main()
