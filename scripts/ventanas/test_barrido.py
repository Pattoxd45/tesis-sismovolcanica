#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_barrido.py
===============

Casos sinteticos de solucion conocida para barrido.py (seccion 2.4 del
informe, criterio RF-12). No usa datos reales.

Para aislar el PROCEDIMIENTO (barrido + fusion + recorte) del detector, los
casos 1 a 4 usan un detector oraculo: umbral fijo sobre la envolvente RMS de
1 s. Sobre la traza completa ese detector delimita bien cada evento
sintetico, asi que el barrido es correcto si reproduce ese mismo resultado.

Luego se mide STA/LTA con la configuracion real, para comprobar que
procesarlo por ventanas da lo mismo que sobre la traza completa. El
comportamiento propio de STA/LTA en cada caso se reporta como medicion,
no como prueba: es una propiedad del detector, no del barrido.

Uso:
    python test_barrido.py
    python test_barrido.py --figura casos_sinteticos.png
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import barrido as br  # noqa: E402

FS = br.FS
RNG = np.random.default_rng(20261002)
SIGMA_RUIDO = 0.05

# Configuracion STA/LTA real (optimo del barrido de 324 combinaciones),
# copiada de evaluar_detectores.py para no depender de sus rutas.
STALTA = dict(sta_s=2.0, lta_s=60.0, thr_on=4.0, thr_off=1.05,
              banda=(1.0, 20.0))

RESULTADOS = []


def informar(nombre, ok, detalle=""):
    RESULTADOS.append(ok)
    print(f"  [{'PASA ' if ok else 'FALLA'}] {nombre}" +
          (f"\n          {detalle}" if detalle else ""))


# ---------------------------------------------------------------------
# Senal sintetica
# ---------------------------------------------------------------------

def ruido(n):
    return RNG.normal(0.0, SIGMA_RUIDO, n)


def evento(dur_s, amp, f_hz):
    """Senoide con envolvente de subida y bajada de 0,5 s."""
    n = int(round(dur_s * FS))
    t = np.arange(n) / FS
    env = np.ones(n)
    r = min(int(0.5 * FS), n // 2)
    env[:r] = np.linspace(0, 1, r)
    env[n - r:] = np.linspace(1, 0, r)
    return amp * env * np.sin(2 * np.pi * f_hz * t)


def insertar(x, ini_s, ev):
    i = int(round(ini_s * FS))
    x[i:i + len(ev)] += ev
    return i, i + len(ev)


# ---------------------------------------------------------------------
# Detector oraculo (envolvente RMS de 1 s con umbral fijo)
# ---------------------------------------------------------------------

UMBRAL_ORACULO = 0.25


def oraculo(x, umbral=UMBRAL_ORACULO):
    k = int(FS)
    rms = np.sqrt(np.convolve(x ** 2, np.ones(k) / k, mode="same"))
    m = (rms >= umbral).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def oraculo_normalizado(x):
    """Variante PROHIBIDA por el diseno: normaliza la ventana por su maximo."""
    return oraculo(x / (np.max(np.abs(x)) + 1e-12))


def barrer(x, cortes, detector, ventanas=None):
    """Barrido + fusion sobre una traza sintetica de un canal."""
    if ventanas is None:
        ventanas = br.generar_ventanas(len(x), cortes)
    crudos = br.detectar_por_ventanas(x, ventanas, detector)
    fus = br.fusionar([(d["ini"], d["fin"]) for d in crudos],
                      int(br.GAP_FUSION_S * FS))
    return fus, crudos, ventanas


def completa(x, cortes, detector):
    """Referencia: el mismo detector sobre cada tramo completo."""
    bordes = [0] + list(cortes) + [len(x)]
    out = []
    for i in range(len(bordes) - 1):
        a, b = bordes[i], bordes[i + 1]
        out += [(a + s, a + e) for s, e in detector(x[a:b])]
    return br.fusionar(out, int(br.GAP_FUSION_S * FS))


def iou(p, q):
    inter = max(0, min(p[1], q[1]) - max(p[0], q[0]))
    union = max(p[1], q[1]) - min(p[0], q[0])
    return inter / union if union > 0 else 0.0


def cubre(iv, verdad, tol_s=1.0):
    """El intervalo detectado contiene al verdadero, con tolerancia."""
    t = int(tol_s * FS)
    return iv[0] <= verdad[0] + t and iv[1] >= verdad[1] - t


# =====================================================================
# PRUEBAS
# =====================================================================

def prueba_ventanas():
    print("\n1. Generacion de ventanas y garantia de cobertura")
    n, cortes = 3_664_193, [2_160_000, 3_240_001]
    v = br.generar_ventanas(n, cortes)
    bordes = [0] + cortes + [n]
    cruza = [w for w in v for c in cortes if w[0] < c < w[1]]
    informar(f"{len(v)} ventanas para la traza real; ninguna cruza un corte",
             not cruza, f"ventanas que cruzan: {len(cruza)}")

    # Todo intervalo de hasta 240,8 s dentro de un tramo debe caber entero
    # en alguna ventana. Se prueban posiciones al azar y los extremos.
    d = int(round(br.DUR_MAX_CAT_S * FS))
    fallos = 0
    for i in range(len(bordes) - 1):
        a, b = bordes[i], bordes[i + 1]
        pos = list(RNG.integers(a, b - d, 3000)) + [a, b - d]
        for p in pos:
            if not br.ventanas_que_contienen(p, p + d, v):
                fallos += 1
    informar("todo evento de 240,8 s cabe entero en al menos una ventana",
             fallos == 0, f"intervalos sin ventana que los contenga: {fallos}")

    # Control negativo: con paso de 65 s (> 59,2 s) la garantia se rompe.
    v2 = br.generar_ventanas(n, cortes, paso_s=65.0)
    sin = sum(1 for p in RNG.integers(0, cortes[0] - d, 3000)
              if not br.ventanas_que_contienen(p, p + d, v2))
    informar("control negativo: con paso de 65 s hay eventos sin ventana",
             sin > 0, f"eventos de 240,8 s sin ventana completa: {sin} de 3000")


def prueba_fusion():
    print("\n2. Fusion de intervalos (logica pura)")
    g = int(5 * FS)
    casos = [
        ([(0, 100), (90, 200)], [(0, 200)], "solapados se unen"),
        ([(0, 100), (100 + g, 300)], [(0, 300)], "separados justo por el umbral se unen"),
        ([(0, 100), (100 + g + 1, 300)], [(0, 100), (100 + g + 1, 300)],
         "separados por mas del umbral no se unen"),
        ([(200, 300), (0, 250), (290, 400)], [(0, 400)], "orden arbitrario y cadena"),
        ([], [], "lista vacia"),
    ]
    for entrada, esperado, nombre in casos:
        r = br.fusionar(entrada, g)
        informar(nombre, r == esperado, f"obtenido {r}, esperado {esperado}")


def caso_1():
    print("\n3. Caso 1: evento de 3,4 s aislado")
    x = ruido(int(1200 * FS))
    verdad = insertar(x, 500.0, evento(3.4, 1.0, 5.0))
    ref = completa(x, [], oraculo)
    fus, _, _ = barrer(x, [], oraculo)
    informar("el barrido reproduce la deteccion sobre la traza completa",
             fus == ref, f"barrido {fus} | completa {ref}")
    ok = len(fus) == 1 and abs(fus[0][0] - ref[0][0]) == 0
    informar("mismo onset que la traza completa", ok,
             f"onset barrido {fus[0][0] / FS:.2f} s, completa {ref[0][0] / FS:.2f} s,"
             f" verdadero {verdad[0] / FS:.2f} s")
    return x, verdad, fus


def caso_2():
    print("\n4. Caso 2: evento de 240,8 s que cruza bordes de ventana")
    x = ruido(int(1200 * FS))
    verdad = insertar(x, 200.0, evento(240.8, 1.0, 2.0))
    ref = completa(x, [], oraculo)
    fus, crudos, v = barrer(x, [], oraculo)
    cortados = [c for c in crudos if c["toca_borde_der"]
                and c["ini"] < verdad[1] and verdad[0] < c["fin"]]
    informar("el caso ejercita la fusion: hay detecciones cortadas en el borde",
             len(cortados) > 0, f"detecciones cortadas en borde derecho: {len(cortados)}")
    informar("la fusion recupera el evento entero",
             len(fus) == 1 and fus == ref and cubre(fus[0], verdad),
             f"fusionado {fus[0][0] / FS:.1f}-{fus[0][1] / FS:.1f} s | verdadero "
             f"{verdad[0] / FS:.1f}-{verdad[1] / FS:.1f} s")

    # Red de seguridad: se quitan las ventanas que contienen el evento entero
    # y la fusion debe reconstruirlo solo con los trozos cortados.
    contenedoras = br.ventanas_que_contienen(*verdad, v)
    v_sin = [w for k, w in enumerate(v) if k not in contenedoras]
    fus2, _, _ = barrer(x, [], oraculo, ventanas=v_sin)
    trozo = [f for f in fus2 if f[0] < verdad[1] and verdad[0] < f[1]]
    informar("sin ventanas que lo contengan, la fusion lo reconstruye con trozos",
             len(trozo) == 1 and cubre(trozo[0], verdad),
             f"ventanas contenedoras quitadas: {len(contenedoras)}")
    return x, verdad, fus, crudos, v


def caso_3():
    print("\n5. Caso 3: evento debil a 30 s de uno diez veces mayor")
    x = ruido(int(1200 * FS))
    fuerte = insertar(x, 400.0, evento(60.0, 10.0, 2.0))
    debil = insertar(x, 400.0 + 60.0 + 30.0, evento(5.0, 1.0, 5.0))
    fus, _, v = barrer(x, [], oraculo)
    comun = [k for k in br.ventanas_que_contienen(fuerte[0], debil[1], v)]
    informar("ambos eventos caen en una misma ventana de barrido",
             len(comun) > 0, f"ventanas que contienen a los dos: {len(comun)}")
    ok = (len(fus) == 2 and cubre(fus[0], fuerte) and cubre(fus[1], debil))
    informar("sin normalizar la ventana, salen como dos intervalos separados", ok,
             f"intervalos: {[(round(a / FS, 1), round(b / FS, 1)) for a, b in fus]}")
    fus_n, _, _ = barrer(x, [], oraculo_normalizado)
    perdido = not any(f[0] < debil[1] and debil[0] < f[1] for f in fus_n)
    informar("control: normalizando por ventana, el evento debil se pierde",
             perdido, "confirma por que el diseno prohibe normalizar la ventana")
    return x, fuerte, debil, fus


def caso_4():
    print("\n6. Caso 4: evento de exactamente 81,92 s")
    x = ruido(int(1200 * FS))
    verdad = insertar(x, 300.0, evento(81.92, 1.0, 2.0))
    fus, _, _ = barrer(x, [], oraculo)
    n = len(x)
    adapt = br.recorte_adaptativo(*fus[0], n, [])
    fijo = br.recorte_fijo(fus[0][0], n, [])
    informar("el recorte adaptativo contiene el evento completo",
             cubre(adapt, verdad), f"recorte {adapt[0] / FS:.2f}-{adapt[1] / FS:.2f} s")
    corte = (verdad[1] - fijo[1]) / FS
    informar("la ventana fija de 81,92 s con margen previo lo trunca", corte > 0,
             f"la ventana fija deja fuera los ultimos {corte:.2f} s del evento")


def prueba_stalta():
    print("\n7. STA/LTA real: por ventanas contra traza completa")
    import importlib.util
    if importlib.util.find_spec("obspy") is None:
        print("  (se omite: obspy no esta instalado)")
        return None
    # Tres tramos, como la traza real, con eventos de duraciones del catalogo.
    tramos_s = [3600.0, 1800.0, 900.0]
    n = int(sum(tramos_s) * FS)
    cortes = [int(tramos_s[0] * FS), int((tramos_s[0] + tramos_s[1]) * FS)]
    x = ruido(n)
    t, verdad = 120.0, []
    fin_total = sum(tramos_s)
    while t < fin_total - 300:
        dur = float(RNG.uniform(3.4, 240.8))
        a = int(t * FS)
        if any(a < k < a + int((dur + 90) * FS) for k in cortes):
            t = min(k for k in cortes if k > a) / FS + 120.0
            continue
        verdad.append(insertar(x, t, evento(dur, float(RNG.uniform(0.3, 2.0)),
                                           float(RNG.choice([2.0, 5.0])))))
        t += dur + float(RNG.uniform(60, 400))
    det = lambda w: br.stalta_en_ventana(w, STALTA)  # noqa: E731
    ref = completa(x, cortes, det)
    fus, _, _ = barrer(x, cortes, det)

    def emparejar(p, q):
        return [max((iou(a, b) for b in q), default=0.0) for a in p]

    i_ref = emparejar(ref, fus)
    i_fus = emparejar(fus, ref)
    informar(f"{len(verdad)} eventos sinteticos: toda deteccion completa tiene "
             "par en el barrido con IoU >= 0,9", min(i_ref, default=1) >= 0.9,
             f"completa {len(ref)} intervalos, barrido {len(fus)}; "
             f"IoU minimo {min(i_ref, default=1):.3f}")
    informar("y todo intervalo del barrido tiene par en la traza completa",
             min(i_fus, default=1) >= 0.9, f"IoU minimo {min(i_fus, default=1):.3f}")

    # Medicion (no prueba): duracion que STA/LTA asigna a eventos largos.
    largos = [v for v in verdad if (v[1] - v[0]) / FS > 150]
    if largos:
        print("\n  Medicion: cobertura de STA/LTA sobre eventos de mas de 150 s")
        for v in largos[:5]:
            m = max((iou(v, f) for f in fus), default=0.0)
            print(f"    evento de {(v[1] - v[0]) / FS:6.1f} s -> IoU con su mejor"
                  f" deteccion {m:.2f}")
    return None


# =====================================================================
# FIGURA
# =====================================================================

def figura(destino, c2):
    import figuras

    x, verdad, fus, crudos, v = c2
    v0 = verdad[0]
    rel = lambda m: (m - v0) / FS  # noqa: E731
    tocan = br.ventanas_que_tocan(verdad[0], verdad[1], v)
    lo, hi = v[tocan[0]][0], v[tocan[-1]][1]
    disparos = {}
    for c in crudos:
        if c["ventana"] in tocan:
            disparos.setdefault(c["ventana"], []).append(
                (rel(c["ini"]), rel(c["fin"]), c["toca_borde_izq"] or c["toca_borde_der"]))
    ra = br.recorte_adaptativo(*fus[0], len(x), [])
    rf = br.recorte_fijo(fus[0][0], len(x), [])
    figuras.figura_barrido(
        t=(np.arange(lo, hi) - v0) / FS, x=x[lo:hi],
        ref=(0.0, rel(verdad[1])), color_ref="#d62728",
        etiqueta_ref="evento sintético de 240,8 s",
        ventanas=[(k, rel(v[k][0]), rel(v[k][1])) for k in tocan],
        disparos=disparos,
        contenedoras=set(br.ventanas_que_contienen(verdad[0], verdad[1], v)),
        fusionados=[(rel(a), rel(b)) for a, b in fus],
        recorte_adapt=(rel(ra[0]), rel(ra[1])), recorte_fijo=(rel(rf[0]), rel(rf[1])),
        titulo="Caso sintético 2: evento de 240,8 s barrido con ventanas de 300 s"
               " y paso de 55 s (detector oráculo)",
        destino=destino)
    print(f"\n  figura guardada en {destino}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figura", default=None)
    args = ap.parse_args()

    prueba_ventanas()
    prueba_fusion()
    caso_1()
    c2 = caso_2()
    caso_3()
    caso_4()
    prueba_stalta()
    if args.figura:
        figura(args.figura, c2)

    n_ok = sum(RESULTADOS)
    print(f"\nResumen: {n_ok} de {len(RESULTADOS)} pruebas pasan")
    sys.exit(0 if n_ok == len(RESULTADOS) else 1)


if __name__ == "__main__":
    main()
