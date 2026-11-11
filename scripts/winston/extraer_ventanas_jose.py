#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extrae Z, N y E de FRE para las ventanas de 300 s de Jose que tienen las tres componentes: trae del
NAS las tablas de dia por tandas (en el NAS solo se lee; las tablas quedan guardadas en la MariaDB
local), guarda un npz por ventana y compara la Z con la que extrajo Jose. Si se corta, se retoma.

Uso: TESIS_NAS=usuario@ip python extraer_ventanas_jose.py [--limite 20] [--tanda 10]
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import extraer_winston as ew
import winston_lib as wl

FS = wl.FS
ESTACION = "FRE"
DIR_JOSE = os.path.expanduser("~/tesis/datos/jose/Pato")
COBERTURA = os.path.expanduser("~/tesis/resultados/sistema/cobertura_jose.csv")
DIR_SALIDA = os.path.expanduser("~/tesis/datos/jose_zne")
REGISTRO = os.path.expanduser("~/tesis/resultados/sistema/extraccion_jose.csv")
# 300 s a 100 Hz: el largo de las ventanas de Jose
N_VENTANA = 30000
# dias por tanda: cerca de 0,8 GB por tanda; asi la clave de sudo (15 min) no alcanza a vencer
TANDA_DIAS = 10
# desfases (en muestras) en que se busca la Z de Jose si no coincide en el mismo instante
DESFASES = range(-2, 3)


def utc(segundos_j2k):
    return datetime.fromtimestamp(segundos_j2k + wl.J2K_A_UNIX, tz=timezone.utc)


def tandas(dias_por_ventana, k):
    """Agrupa los dias en tandas de k en orden; cada ventana queda en la tanda donde se completan sus dias."""
    todos = sorted(set(d for ds in dias_por_ventana.values() for d in ds))
    grupos = [todos[i:i + k] for i in range(0, len(todos), k)]
    ultimo = {d: i for i, g in enumerate(grupos) for d in g}
    por_tanda = [[] for _ in grupos]
    for v, ds in dias_por_ventana.items():
        por_tanda[max(ultimo[d] for d in ds)].append(v)
    return grupos, por_tanda


def comparar_z(z, j):
    """Parte de las muestras con dato de la Z propia iguales a la de Jose, y el desfase en que mejor calzan."""
    if j is None or j.size != z.size:
        return np.nan, None
    mejor = (np.nan, None)
    for k in DESFASES:
        a = z[max(k, 0):z.size + min(k, 0)]
        b = j[max(-k, 0):j.size + min(-k, 0)]
        m = np.isfinite(a)
        if m.any():
            f = float(np.mean(a[m] == b[m]))
            if np.isnan(mejor[0]) or f > mejor[0]:
                mejor = (f, k)
    return mejor


def z_de_jose(evento, clase):
    ruta = os.path.join(DIR_JOSE, "dataset_SOLO_Z_300s", clase, f"{evento}_300s_raw.npy")
    return np.load(ruta).astype(np.float64) if os.path.exists(ruta) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None, help="solo las primeras N ventanas, para probar")
    ap.add_argument("--tanda", type=int, default=TANDA_DIAS)
    args = ap.parse_args()
    nas = os.environ.get("TESIS_NAS")
    if not nas:
        raise SystemExit("DETENIDO: falta TESIS_NAS=usuario@ip del NAS")

    c = pd.read_csv(COBERTURA)
    v = c[c.Z & c.N & c.E].sort_values("inicio_ventana_w").reset_index(drop=True)
    if args.limite:
        v = v.head(args.limite)
    os.makedirs(DIR_SALIDA, exist_ok=True)
    hechas = {f[:-len("_zne.npz")] for f in os.listdir(DIR_SALIDA) if f.endswith("_zne.npz")}
    pend = v[~v.evento_principal.isin(hechas)].set_index("evento_principal")
    print(f"{len(v)} ventanas con Z, N y E; ya extraidas {len(v) - len(pend)}; por extraer {len(pend)}")
    if pend.empty:
        return

    dias_v = {e: ew.dias(utc(r.inicio_ventana_w), utc(r.fin_ventana_w)) for e, r in pend.iterrows()}
    grupos, por_tanda = tandas(dias_v, args.tanda)
    registro = pd.read_csv(REGISTRO).to_dict("records") if os.path.exists(REGISTRO) else []
    t_ini, copiados_total = time.time(), 0
    print(f"{len(grupos)} tandas de hasta {args.tanda} dias (puede pedir la contrasena del NAS y la clave de sudo)")
    for i, (g, eventos) in enumerate(zip(grupos, por_tanda), 1):
        sin_dato, copiados = ew.traer_tablas(nas, ESTACION, g)
        copiados_total += copiados // 3
        cargadas = ew.tablas_cargadas()
        for e in eventos:
            r = pend.loc[e]
            t0 = r.inicio_ventana_w + wl.J2K_A_UNIX
            senal, infos = ew.senal_entre(ESTACION, dias_v[e], t0, N_VENTANA, cargadas)
            f, k = comparar_z(senal["Z"], z_de_jose(e, r.clase_principal))
            np.savez_compressed(os.path.join(DIR_SALIDA, f"{e}_zne.npz"), t0=t0, fs=FS, evento=e,
                                clase=r.clase_principal, split=r.split, **senal)
            registro.append(dict(evento=e, clase=r.clase_principal, split=r.split,
                                 **{f"con_dato_{c}": infos[c]["validas"] / N_VENTANA for c in senal},
                                 **{f"pisadas_{c}": infos[c]["conflictos"] for c in senal},
                                 z_igual_jose=f, desfase_jose=k))
        pd.DataFrame(registro).to_csv(REGISTRO, index=False)
        h = (time.time() - t_ini) / 3600
        print(f"  tanda {i}/{len(grupos)}: {len(g)} dias ({g[0]} a {g[-1]}), {len(eventos)} ventanas; "
              f"tablas traidas {copiados // 3}, sin tabla {len(sin_dato)}; llevamos {len(registro)} ventanas y "
              f"{copiados_total} tablas en {h:.2f} h", flush=True)

    d = pd.DataFrame(registro)
    print(f"\nResumen de {len(d)} ventanas:")
    for c in ("Z", "N", "E"):
        print(f"  {c}: con dato en mediana {d[f'con_dato_{c}'].median():.1%}; ventanas con menos de 95 %: "
              f"{int((d[f'con_dato_{c}'] < 0.95).sum())}")
    z = d.z_igual_jose.dropna()
    print(f"  Z igual a la de Jose: {int((z == 1).sum())} de {len(z)} ventanas comparadas; "
          f"desfases encontrados: {d.desfase_jose.value_counts(dropna=False).to_dict()}")
    print(f"Ventanas en {DIR_SALIDA}; detalle en {REGISTRO}")


if __name__ == "__main__":
    main()
