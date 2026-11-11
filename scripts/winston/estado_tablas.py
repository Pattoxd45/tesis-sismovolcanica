#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estado de las tablas de dia de FRE en el respaldo del NAS: lee por SSH la cabecera del indice (.MYI)
de cada tabla y el tamano de sus archivos (en el NAS solo se lee) y cuenta las que vienen marcadas
como danadas, sin cerrar o con el archivo de datos (.MYD) mas corto de lo que dice el indice; lo
cruza con las ventanas de Jose, con las tablas ya cargadas en la MariaDB local y con el registro de
la extraccion.

Uso: TESIS_NAS=usuario@ip python estado_tablas.py
"""

import os
import re
import sys
from datetime import date

import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import extraer_winston as ew
import extraer_ventanas_jose as ej

ESTACION = "FRE"
SALIDA = os.path.expanduser("~/tesis/resultados/sistema/estado_tablas.csv")
# archivo de una tabla de dia con senal; las de helicorder llevan una h antes de la fecha
DIA_ARCHIVO = re.compile(r"\$\$(\d{4})_(\d{2})_(\d{2})\.(myi|myd)$", re.IGNORECASE)
# cabecera del .MYI (myisamdef.h y mi_open.c de MariaDB): 24 bytes fijos y luego el estado de la tabla
POS_ESTADO = 24
# del estado se leen open_count, changed, sortkey, records, del, split, dellink, key_file_length y data_file_length
LARGO = 52
# bits de changed en MyISAM: 2 = marcada como danada, 4 = danada durante una reparacion
DANADA = 2 | 4


def comando_nas():
    """Por cada indice de tabla de dia: una linea con la componente y la ruta, los bytes de estado y un salto."""
    # el NAS no trae od; se usa dd y, si falta, head y tail
    leer = (f"if command -v dd >/dev/null 2>&1; then leer() {{ dd if=\"$1\" bs=1 skip={POS_ESTADO} count={LARGO} "
            f"2>/dev/null; }}; else leer() {{ head -c {POS_ESTADO + LARGO} \"$1\" | tail -c {LARGO}; }}; fi")
    partes = [f"for f in '{d}'/*; do case \"$f\" in *'$$'[0-9]*.[Mm][Yy][Ii]) echo \"## {c} $f\"; leer \"$f\"; echo;; "
              f"esac; done" for c, d in ew.CARPETAS[ESTACION].items()]
    return f'cd "{ew.DIR_NAS}" && {leer}; ' + "; ".join(partes)


def comando_listado():
    partes = [f"echo '## {c}'; ls -ln '{d}'" for c, d in ew.CARPETAS[ESTACION].items()]
    return f'cd "{ew.DIR_NAS}" && ' + "; ".join(partes)


def parsear_estado(salida):
    """salida en bytes: por tabla, '## componente ruta', salto, LARGO bytes de estado y salto."""
    filas, sin_leer, i = [], 0, 0
    while True:
        j = salida.find(b"## ", i)
        k = salida.find(b"\n", j) if j >= 0 else -1
        if k < 0:
            break
        comp, ruta = salida[j + 3:k].decode(errors="replace").split(" ", 1)
        b = salida[k + 1:k + 1 + LARGO]
        if len(b) < LARGO or salida[k + 1 + LARGO:k + 2 + LARGO] != b"\n":
            sin_leer += 1
            i = k + 1
            continue
        i = k + 2 + LARGO
        m = DIA_ARCHIVO.search(ruta.strip())
        if m:
            filas.append(dict(comp=comp, dia=date(*map(int, m.groups()[:3])),
                              abierta=int.from_bytes(b[0:2], "big"), changed=b[2],
                              registros=int.from_bytes(b[4:12], "big"),
                              largo_indice=int.from_bytes(b[36:44], "big"),
                              largo_datos=int.from_bytes(b[44:52], "big")))
    e = pd.DataFrame(filas, columns=["comp", "dia", "abierta", "changed", "registros", "largo_indice", "largo_datos"])
    e["danada"] = (e.changed & DANADA) != 0
    e["sin_cerrar"] = (e.abierta > 0) & ~e.danada
    return e, sin_leer


def parsear_listado(texto):
    """Tamano en bytes del .MYI y del .MYD de cada (componente, dia), desde ls -ln."""
    tam, comp = {}, None
    for linea in texto.splitlines():
        if linea.startswith("## "):
            comp = linea[3:].strip()
            continue
        t = linea.split()
        m = DIA_ARCHIVO.search(t[-1]) if comp and len(t) >= 9 else None
        if m:
            clave = (comp, date(*map(int, m.groups()[:3])))
            tam.setdefault(clave, {})[m.group(4).upper()] = int(t[4])
    return tam


def ventanas_afectadas(v, claves):
    """Ventanas que necesitan alguna de las tablas (componente, dia) dadas."""
    out = []
    for r in v.itertuples():
        ds = ew.dias(ej.utc(r.inicio_ventana_w), ej.utc(r.fin_ventana_w))
        if any((c, d) in claves for c in ("Z", "N", "E") for d in ds):
            out.append(r.evento_principal)
    return out


def main():
    nas = os.environ.get("TESIS_NAS")
    if not nas:
        raise SystemExit("DETENIDO: falta TESIS_NAS=usuario@ip del NAS")

    print(f"1. TABLAS DE DIA DE {ESTACION} EN EL RESPALDO (se leen {LARGO} bytes de la cabecera de cada indice y el "
          f"tamano de los archivos)")
    r = ew.ssh(nas, comando_nas(), capture_output=True)
    e, sin_leer = parsear_estado(r.stdout)
    errores = sorted(set(r.stderr.decode(errors="replace").splitlines()))[:5]
    if e.empty:
        raise SystemExit("DETENIDO: no se pudo leer el estado en el NAS: " + " | ".join(errores))
    r2 = ew.ssh(nas, comando_listado(), capture_output=True, text=True)
    tam = parsear_listado(r2.stdout)
    e["tam_indice"] = [tam.get((c, d), {}).get("MYI") for c, d in zip(e.comp, e.dia)]
    e["tam_datos"] = [tam.get((c, d), {}).get("MYD") for c, d in zip(e.comp, e.dia)]
    e["corta"] = e.tam_datos < e.largo_datos
    for c in ("Z", "N", "E"):
        x = e[e.comp == c]
        print(f"   {c}: {len(x)} tablas; marcadas como danadas {int(x.danada.sum())}; sin cerrar {int(x.sin_cerrar.sum())}; "
              f"datos mas cortos que lo que dice el indice {int(x.corta.sum())}; "
              f"indice y datos del largo que dice la cabecera {int(((x.tam_indice == x.largo_indice) & (x.tam_datos == x.largo_datos)).sum())}")
    if sin_leer:
        print(f"   indices que no se pudieron leer: {sin_leer}; errores: " + " | ".join(errores))
    cortas = e[e.corta]
    for f in cortas.sort_values("dia").head(30).itertuples():
        print(f"   corta: {f.comp} {f.dia}: datos {f.tam_datos} bytes de {f.largo_datos} ({f.tam_datos / f.largo_datos:.0%})")
    malas = set(zip(cortas.comp, cortas.dia)) | set(zip(e[e.danada].comp, e[e.danada].dia))
    sin_cerrar = set(zip(e[e.sin_cerrar].comp, e[e.sin_cerrar].dia))

    print("\n2. VENTANAS DE JOSE CON Z, N Y E QUE USAN ESAS TABLAS")
    cob = pd.read_csv(ej.COBERTURA)
    v = cob[cob.Z & cob.N & cob.E]
    a_m, a_s = ventanas_afectadas(v, malas), ventanas_afectadas(v, sin_cerrar)
    vm = v[v.evento_principal.isin(a_m)]
    print(f"   con alguna tabla danada o corta: {len(a_m)} de {len(v)} (desarrollo {int((vm.split == 'desarrollo').sum())}, "
          f"test {int((vm.split == 'test').sum())}); con alguna sin cerrar: {len(a_s)}")
    if len(vm):
        print("   por clase: " + ", ".join(f"{k} {n}" for k, n in vm.clase_principal.value_counts().items()))

    print("\n3. MARIADB LOCAL")
    _, out, _ = ew.mysql_local("SELECT @@myisam_recover_options, @@log_error")
    opc, _, log = out.rstrip("\n").partition("\t")
    print(f"   myisam_recover_options: {opc or '(vacio)'}; registro de errores: {log or '(salida estandar)'}")
    cargadas = ew.tablas_cargadas()
    ya = {(c, d) for c, d in malas | sin_cerrar if ew.tabla_local(ESTACION, c, d) in cargadas}
    print(f"   tablas ya cargadas: {len(cargadas)}; de ellas danadas o cortas en el respaldo {len(ya & malas)}, "
          f"sin cerrar {len(ya & sin_cerrar)}")

    print("\n4. REGISTRO DE LA EXTRACCION")
    if os.path.exists(ej.REGISTRO):
        d = pd.read_csv(ej.REGISTRO)
        npz = [f for f in os.listdir(ej.DIR_SALIDA) if f.endswith("_zne.npz")]
        hechas = ej.ya_hechas(npz, d.evento)
        print(f"   filas {len(d)}; npz {len(npz)}; npz sin fila (tanda cortada) {len(npz) - len(hechas)}")
        z = d.z_igual_jose.dropna()
        print(f"   Z igual a la de Jose: {int((z == 1).sum())} de {len(z)}; menor que 1: {int((z < 1).sum())}")
        for c in ("Z", "N", "E"):
            print(f"   {c} con menos de 95 % de dato: {int((d[f'con_dato_{c}'] < 0.95).sum())}")
        baja = d[(d.con_dato_Z < 0.95) | (d.con_dato_N < 0.95) | (d.con_dato_E < 0.95)]
        print(f"   ventanas ya extraidas que usan tablas danadas o cortas: {int(d.evento.isin(a_m).sum())}; de las "
              f"{len(baja)} con menos de 95 % en alguna componente, usan tablas cortas {int(baja.evento.isin(a_m).sum())}")

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    e.to_csv(SALIDA, index=False)
    print(f"\nEstado por tabla en {SALIDA}")


if __name__ == "__main__":
    main()
