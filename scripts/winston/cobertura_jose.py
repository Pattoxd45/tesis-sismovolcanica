#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cuantas ventanas de 300 s del conjunto de Jose tienen Z, N y E de FRE en el respaldo Winston del
NAS: lista por SSH los dias que existen de cada componente (en el NAS solo se lee) y los cruza con
los dias que necesita cada ventana. No mira huecos dentro del dia; eso se ve al extraer.

Uso: TESIS_NAS=usuario@ip python cobertura_jose.py [--jose ~/tesis/datos/jose/Pato]
"""

import argparse
import os
import re
import sys
from datetime import date, datetime, timezone

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import extraer_winston as ew
import winston_lib as wl

DIR_SALIDA = os.path.expanduser("~/tesis/resultados/sistema")
# tabla de un dia con senal: termina en $$AAAA_MM_DD.MYD; las de helicorder llevan una h antes de la fecha
DIA_TABLA = re.compile(r"\$\$(\d{4})_(\d{2})_(\d{2})\.myd$", re.IGNORECASE)


def nombre(ruta):
    return os.path.basename(ruta.strip()).replace("_300s.png", "")


def cargar_jose(dir_jose):
    meta = os.path.join(dir_jose, "metadata_dataset", "metadata", "resumen_ventanas_para_entrenamiento.csv")
    v = pd.read_csv(meta, encoding="utf-8-sig")
    splits = {}
    for s in ("desarrollo", "test"):
        with open(os.path.join(dir_jose, "splits", f"{s}.txt")) as f:
            splits.update({nombre(l): s for l in f if l.strip()})
    v["split"] = v.evento_principal.map(splits)
    if v.split.isna().any():
        raise SystemExit(f"DETENIDO: {int(v.split.isna().sum())} ventanas sin particion")
    return v


def dias_en_nas(nas):
    """Dias con tabla de senal de cada componente de FRE en el respaldo."""
    carpetas = ew.CARPETAS["FRE"]
    cmd = f'cd "{ew.DIR_NAS}" && ' + "; ".join(f"echo '##{c}'; ls '{d}'" for c, d in carpetas.items())
    r = ew.ssh(nas, cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"DETENIDO: no se pudo listar el NAS: {r.stderr.strip()}")
    dias, comp = {c: set() for c in carpetas}, None
    for linea in r.stdout.splitlines():
        if linea.startswith("##"):
            comp = linea[2:].strip()
            continue
        m = DIA_TABLA.search(linea.strip())
        if comp and m:
            dias[comp].add(date(*map(int, m.groups())))
    return dias


def cobertura(v, dias):
    """Por ventana, si cada componente tiene la tabla de todos los dias que la ventana necesita."""
    filas = []
    for r in v.itertuples():
        ini = datetime.fromtimestamp(r.inicio_ventana_w + wl.J2K_A_UNIX, tz=timezone.utc)
        fin = datetime.fromtimestamp(r.fin_ventana_w + wl.J2K_A_UNIX, tz=timezone.utc)
        necesita = ew.dias(ini, fin)
        filas.append({c: all(d in dias[c] for d in necesita) for c in dias})
    return pd.concat([v.reset_index(drop=True), pd.DataFrame(filas)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jose", default=os.path.expanduser("~/tesis/datos/jose/Pato"))
    args = ap.parse_args()
    nas = os.environ.get("TESIS_NAS")
    if not nas:
        raise SystemExit("DETENIDO: falta TESIS_NAS=usuario@ip del NAS")

    v = cargar_jose(args.jose)
    dias = dias_en_nas(nas)
    for c, d in dias.items():
        print(f"{c}: {len(d)} dias en el respaldo, de {min(d)} a {max(d)}")
    c = cobertura(v, dias)
    c["estado"] = c.apply(lambda r: "Z, N y E" if r.Z and r.N and r.E else
                          "falta " + " y ".join(k for k in ("Z", "N", "E") if not r[k]), axis=1)
    print(f"\n{len(c)} ventanas de Jose (desarrollo {int((c.split == 'desarrollo').sum())}, "
          f"test {int((c.split == 'test').sum())})")
    print(pd.crosstab([c.split, c.estado], c.clase_principal, margins=True).to_string())
    os.makedirs(DIR_SALIDA, exist_ok=True)
    salida = os.path.join(DIR_SALIDA, "cobertura_jose.csv")
    c.to_csv(salida, index=False)
    print(f"\nTabla por ventana en {salida}")


if __name__ == "__main__":
    main()
