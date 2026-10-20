#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Casos sinteticos para recorte_fre3c.py.

Uso: python test_recorte_fre3c.py
"""

import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import recorte_fre3c as rc

FS = rc.FS
S = int(FS)
RNG = np.random.default_rng(7)
RES = []


def informar(nombre, ok, detalle=""):
    RES.append(bool(ok))
    print(f"  [{'PASA ' if ok else 'FALLA'}] {nombre}" + (f"\n          {detalle}" if detalle else ""))


def ruido(n, sigma=1.0):
    return [RNG.normal(0, sigma, n) for _ in range(3)]


def senoide(n, f=5.0):
    return np.sin(2 * np.pi * f * np.arange(n) / FS)


def test_envolvente():
    print("\n1. Envolvente")
    x = np.full(1000, 4.0)
    informar("media movil de una constante es la constante", np.allclose(rc.media_movil(x, 100), 4.0))
    z, n, e = ruido(60 * S)
    env = rc.envolvente(z, n, e)
    informar("ruido de sigma 1 en tres componentes: envolvente ~ raiz de 3",
             abs(np.median(env) - np.sqrt(3)) < 0.1, f"mediana {np.median(env):.3f}")


def test_fin():
    print("\n2. Fin por energia")
    n = 600 * S
    lim = [(0, n)]
    t0 = 100 * S
    valida = np.ones(n, dtype=bool)

    # decaimiento exponencial: la envolvente^2 es 3 sigma^2 + A^2 exp(-2t/tau) / 2, asi que
    # cae bajo el umbral u en t = (tau/2) ln(A^2 / (2 (u^2 - 3 sigma^2)))
    z, nn, e = ruido(n)
    A, tau, k = 50.0, 10.0, 2.0
    t = np.arange(n - t0) / FS
    z[t0:] += A * np.exp(-t / tau) * senoide(n - t0)
    env = rc.envolvente(z, nn, e)
    u = rc.umbral_ruido(env, valida, t0, k, 0)
    esperado = t0 + (tau / 2) * np.log(A ** 2 / (2 * (u ** 2 - 3))) * FS
    fin = rc.fin_por_energia(env, valida, t0, k, lim)
    informar("decaimiento exponencial: fin a menos de 2 s del valor teorico",
             abs(fin - esperado) < 2 * S, f"fin {(fin - t0) / FS:.1f} s, teorico {(esperado - t0) / FS:.1f} s")

    # pausa breve de 2 s dentro del evento: no corta
    z, nn, e = ruido(n)
    z[t0:t0 + 20 * S] += 20 * senoide(20 * S)
    z[t0 + 22 * S:t0 + 40 * S] += 20 * senoide(18 * S)
    fin = rc.fin_por_energia(rc.envolvente(z, nn, e), valida, t0, 2.0, lim)
    informar("pausa de 2 s dentro del evento: el recorte sigue hasta ~40 s",
             abs((fin - t0) / FS - 40) < 1.5, f"fin {(fin - t0) / FS:.1f} s")

    # tremor que no termina: tope de 300 s
    z, nn, e = ruido(n)
    z[t0:] += 20 * senoide(n - t0)
    fin = rc.fin_por_energia(rc.envolvente(z, nn, e), valida, t0, 2.0, lim)
    informar("tremor continuo: el recorte se corta en 300 s", fin - t0 == 300 * S, f"{(fin - t0) / FS:.1f} s")

    # zona no valida a los 50 s: el recorte termina ahi
    v2 = valida.copy()
    v2[t0 + 50 * S:t0 + 60 * S] = False
    fin = rc.fin_por_energia(rc.envolvente(z, nn, e), v2, t0, 2.0, lim)
    informar("hueco a los 50 s: el recorte termina en el hueco", fin - t0 == 50 * S, f"{(fin - t0) / FS:.1f} s")

    # sin ruido previo valido: no hay recorte
    fin = rc.fin_por_energia(rc.envolvente(z, nn, e), valida, 3 * S, 2.0, lim)
    informar("arribo a 3 s del inicio del tramo, sin ruido previo: no hay recorte", fin is None)

    # ruido previo con la coda de otro evento en 40 de los 115 s: el umbral no se infla
    z, nn, e = ruido(n)
    z[t0 - 60 * S:t0 - 20 * S] += 20 * senoide(40 * S)
    env = rc.envolvente(z, nn, e)
    u = rc.umbral_ruido(env, valida, t0, 1.0, 0)
    informar("evento previo dentro de la ventana de ruido: el nivel de ruido sigue ~ raiz de 3",
             abs(u - np.sqrt(3)) < 0.25, f"nivel {u:.2f}")


def test_recortes_y_nivel2():
    print("\n3. Recortes y nivel 2")
    n = 600 * S
    lim = [(0, n)]
    valida = np.ones(n, dtype=bool)
    z, nn, e = ruido(n)
    env = rc.envolvente(z, nn, e)
    iv = rc.recortes([100 * S], env, valida, fija=True, limites=lim)
    informar("ventana fija: 2,5 s antes del arribo y 81,92 s de largo",
             iv == [(100 * S - 250, 100 * S - 250 + 8192)], f"{iv}")
    iv = rc.recortes([560 * S], env, valida, fija=True, limites=lim)
    informar("ventana fija recortada al final del tramo", iv[0][1] == n)

    # tremor continuo desde 100 s y un segundo arribo a 130 s
    z2, n2, e2 = ruido(n)
    z2[100 * S:] += 20 * senoide(n - 100 * S)
    env2 = rc.envolvente(z2, n2, e2)
    sin = rc.recortes([100 * S, 130 * S], env2, valida, limites=lim)
    con = rc.recortes([100 * S, 130 * S], env2, valida, corte=True, limites=lim)
    informar("con corte: el primer recorte termina 2,5 s antes del siguiente arribo; el ultimo no cambia",
             sin[0][1] == 400 * S and con[0][1] == 130 * S - 250 and con[1] == sin[1], f"sin {sin}, con {con}")

    inc = pd.DataFrame([dict(ini=1000 * S, fin=1060 * S, clase="TR"),
                        dict(ini=2000 * S, fin=2010 * S, clase="VT")])
    ivs = [(1010 * S, 1070 * S),          # IoU 50/70 = 0,71: acierto TR
           (2005 * S, 2200 * S),          # IoU 5/200: no acierta el VT
           (3000 * S, 3020 * S)]          # fuera de todo evento
    filas, tot = rc.evaluar_nivel2(ivs, inc, (inc.ini - 500).to_numpy(), inc.fin.to_numpy(), 1.0)
    f = {r["clase"]: r for r in filas}
    informar("TR 1/1 con IoU 0,71; VT 0/1",
             f["TR"]["k"] == 1 and abs(f["TR"]["iou_medio"] - 50 / 70) < 1e-9 and f["VT"]["k"] == 0)
    informar("2 recortes sin pareja, 1 fuera de todo evento",
             tot["sin_pareja"] == 2 and tot["externos"] == 1, f"{tot}")


def test_propios():
    print("\n4. Recortes propios de EQTransformer y STA/LTA")
    n = 600 * S
    lim = [(0, n)]
    valida = np.ones(n, dtype=bool)
    valida[500 * S:520 * S] = False
    det = np.zeros(n)
    det[100 * S:110 * S] = 0.8
    det[113 * S:120 * S] = 0.8      # a 3 s del anterior: se une
    det[200 * S:230 * S] = 0.8
    det[505 * S:510 * S] = 0.8      # empieza en zona no valida: se descarta
    det[300 * S:305 * S] = 0.2      # bajo el umbral
    iv = rc.recortes_eqt(det, 0.3, valida, lim)
    esperado = [(100 * S, 120 * S), (200 * S, 230 * S)]
    # trigger_onset marca el fin en la ultima muestra sobre el umbral de apagado
    ok = len(iv) == 2 and all(a == x and 0 <= y - b <= 1 for (a, b), (x, y) in zip(iv, esperado))
    informar("EQT: une a 5 s, descarta zona no valida y lo que no supera el umbral", ok, f"{iv}")

    z = RNG.normal(0, 1, n)
    z[300 * S:330 * S] += 20 * senoide(30 * S)
    iv = rc.recortes_stalta(rc.cft_stalta(z, limites=lim), np.ones(n, dtype=bool), 4.0, limites=lim)
    ok = len(iv) == 1 and abs(iv[0][0] - 300 * S) < 1 * S and abs(iv[0][1] - 330 * S) < 3 * S
    informar("STA/LTA: rafaga de 300 a 330 s da un recorte de ~300 a ~330 s", ok,
             f"{[(a / FS, b / FS) for a, b in iv]}")


def test_diagnostico():
    print("\n5. Diagnostico de los volcanicos sin buen recorte")
    n = 1000 * S
    valida = np.ones(n, dtype=bool)
    valida[800 * S:805 * S] = False
    env = np.ones(n)
    env[500 * S:530 * S] = 1.5
    env[530 * S:800 * S] = 3.0
    ev = pd.DataFrame([dict(id=0, clase="IC", tramo=1, ini=400 * S, fin=460 * S),
                       dict(id=1, clase="LP", tramo=1, ini=500 * S, fin=530 * S),
                       dict(id=2, clase="VT", tramo=1, ini=600 * S, fin=610 * S),
                       dict(id=3, clase="TR", tramo=1, ini=900 * S, fin=950 * S),
                       dict(id=4, clase="VT", tramo=1, ini=960 * S, fin=970 * S)])
    ev["dur_s"] = (ev.fin - ev.ini) / FS
    ivs = [(500 * S - 250, 800 * S), (600 * S - 250, 611 * S)]
    onsets = np.array([500 * S, 600 * S, 961 * S])
    dp = rc.diagnostico(ev, ev, ivs, onsets, env, valida).set_index("id")
    f = dp.loc[1]
    informar("LP de 30 s con recorte de 302,5 s: recorte largo", f.causa == "recorte largo", f.causa)
    informar("ruido ocupado 60 de 115 s por otro evento",
             abs(f.ocup_ruido - round(60 / 115, 2)) < 1e-9, f"{f.ocup_ruido}")
    informar("snr del evento 1,5 y despues del evento 3,0; un evento dentro del recorte",
             f.snr_evento == 1.5 and f.snr_despues == 3.0 and f.otros_en_recorte == 1,
             f"{f.snr_evento}, {f.snr_despues}, {f.otros_en_recorte}")
    informar("el recorte termina en un hueco del registro", f.fin_por == "hueco", f.fin_por)
    informar("VT con buen recorte no aparece", 2 not in dp.index)
    informar("TR sin arribo y VT con arribo pero sin recorte",
             dp.loc[3].causa == "sin arribo" and dp.loc[4].causa == "sin ruido previo",
             f"{dp.loc[3].causa}, {dp.loc[4].causa}")


def main():
    print("TEST recorte_fre3c")
    test_envolvente()
    test_fin()
    test_recortes_y_nivel2()
    test_propios()
    test_diagnostico()
    print(f"\n{sum(RES)} de {len(RES)} casos pasan")
    if sum(RES) != len(RES):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
