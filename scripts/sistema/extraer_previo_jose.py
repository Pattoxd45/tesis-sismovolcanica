#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrae Z, N y E de FRE de los 120 s anteriores a cada ventana completa de Jose, para medir si el
recorte por energia cambia cuando su periodo de ruido puede usar senal de antes de la ventana. Lee
las tablas de la MariaDB local y trae del NAS (solo lectura) las que falten. Si se corta, se retoma.

Uso: [TESIS_NAS=usuario@ip] python extraer_previo_jose.py [--limite 20]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "winston"))
sys.path.insert(0, os.path.join(AQUI, "..", "fre3c"))
sys.path.insert(0, os.path.join(AQUI, "..", "evaluacion"))
import extraer_winston as ew
import extraer_ventanas_jose as ej
import recorte_fre3c as rc
import winston_lib as wl

FS = wl.FS
# el periodo de ruido del recorte por energia empieza 120 s antes del aviso
PREVIO_S = rc.RUIDO_S[0]
N_PREVIO = int(PREVIO_S * FS)
DIR_SALIDA = os.path.expanduser("~/tesis/datos/jose_zne_previo")
REGISTRO = os.path.expanduser("~/tesis/resultados/sistema/extraccion_previo_jose.csv")


def completas(registro):
    """Ventanas con Z, N y E al 100 % y sin tablas cortas."""
    d = pd.read_csv(registro)
    sin_cortas = d.tablas_sin_leer.fillna("").eq("") if "tablas_sin_leer" in d else True
    ok = (d.con_dato_Z == 1) & (d.con_dato_N == 1) & (d.con_dato_E == 1) & sin_cortas
    return d[ok].evento.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None, help="solo las primeras N ventanas, para probar")
    ap.add_argument("--tanda", type=int, default=ej.TANDA_DIAS)
    args = ap.parse_args()
    nas = os.environ.get("TESIS_NAS")

    ev = completas(ej.REGISTRO)
    if args.limite:
        ev = ev[:args.limite]
    cob = pd.read_csv(ej.COBERTURA).set_index("evento_principal")
    os.makedirs(DIR_SALIDA, exist_ok=True)
    registro = pd.read_csv(REGISTRO).to_dict("records") if os.path.exists(REGISTRO) else []
    hechas = ej.ya_hechas(os.listdir(DIR_SALIDA), [r["evento"] for r in registro])
    registro = [r for r in registro if r["evento"] in hechas]
    pend = [e for e in ev if e not in hechas]
    print(f"{len(ev)} ventanas completas; ya extraidos sus {PREVIO_S:.0f} s previos {len(ev) - len(pend)}; "
          f"por extraer {len(pend)}")
    if not pend:
        return

    ini = {e: float(cob.loc[e].inicio_ventana_w) for e in pend}
    dias_v = {e: ew.dias(ej.utc(ini[e] - PREVIO_S), ej.utc(ini[e])) for e in pend}
    grupos, por_tanda = ej.tandas(dias_v, args.tanda)
    cortas = ej.tablas_cortas()
    for i, (g, eventos) in enumerate(zip(grupos, por_tanda), 1):
        cargadas = ew.tablas_cargadas()
        faltan = [d for d in g if any(ew.tabla_local(ej.ESTACION, c, d) not in cargadas for c in ("Z", "N", "E"))]
        if faltan:
            if not nas:
                raise SystemExit(f"DETENIDO: faltan tablas de {', '.join(map(str, faltan))}; correr con TESIS_NAS=usuario@ip")
            ew.traer_tablas(nas, ej.ESTACION, faltan)
            cargadas = ew.tablas_cargadas()
        cargadas -= cortas
        for e in eventos:
            t0 = ini[e] + wl.J2K_A_UNIX - PREVIO_S
            senal, infos = ew.senal_entre(ej.ESTACION, dias_v[e], t0, N_PREVIO, cargadas)
            sin_leer = sorted(t for c in senal for d in dias_v[e] if (t := ew.tabla_local(ej.ESTACION, c, d)) in cortas)
            np.savez_compressed(os.path.join(DIR_SALIDA, f"{e}_zne.npz"), t0=t0, fs=FS, evento=e, **senal)
            registro.append(dict(evento=e, tablas_sin_leer=";".join(sin_leer),
                                 **{f"con_dato_{c}": infos[c]["validas"] / N_PREVIO for c in senal}))
        pd.DataFrame(registro).to_csv(REGISTRO, index=False)
        print(f"  tanda {i}/{len(grupos)}: {len(eventos)} ventanas; tablas traidas del NAS para {len(faltan)} dias", flush=True)

    d = pd.DataFrame(registro)
    completo = (d.con_dato_Z == 1) & (d.con_dato_N == 1) & (d.con_dato_E == 1)
    print(f"\n{len(d)} ventanas con sus {PREVIO_S:.0f} s previos; con Z, N y E completos en ese tramo: {int(completo.sum())}")
    print(f"Senal en {DIR_SALIDA}; detalle en {REGISTRO}")


if __name__ == "__main__":
    main()
