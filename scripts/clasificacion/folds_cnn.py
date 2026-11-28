#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validacion cruzada con la mejor combinacion de la busqueda de una red: entrena los 5 folds de Jose con el
recorte y con la ventana fija de 60 s, y evalua en test el mejor modelo de cada fold (guarda sus pesos).
La combinacion sale del archivo de Optuna de la busqueda; el protocolo y el aumento son los mismos. Una
corrida que ya tiene su JSON no se repite, asi que si el trabajo se corta se vuelve a lanzar igual.
Los argumentos que no son de este script (aumento, --trabajadores, etc.) pasan tal cual a entrenar_cnn.py.

Uso: python folds_cnn.py --datos <carpeta> --modelo efficientnet_b0 --busqueda <carpeta> --salida <carpeta>
     [--ventanas-fijas 60] [--jitter-maximo 5 --jitter-ancho 50 70 --elastica 50 0] [--trabajadores 3]
"""

import argparse
import json
import os
import sys

import optuna

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import buscar_cnn as bc
import entrenar_cnn as ec

FOLDS = range(1, 6)


def mejor_combinacion(busqueda, modelo, fold=1):
    """Mejor combinacion de la busqueda de la red (F1 macro de VT, LP y TR en validacion)."""
    ruta = os.path.join(busqueda, f"busqueda_{modelo}_fold{fold}.db")
    if not os.path.exists(ruta):
        raise SystemExit(f"DETENIDO: no esta la busqueda {ruta}")
    e = optuna.load_study(study_name=f"{modelo}_fold{fold}", storage=f"sqlite:///{ruta}")
    return e.best_params, e.best_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--modelo", required=True, choices=sorted(ec.CABEZAS))
    ap.add_argument("--busqueda", required=True, help="carpeta con busqueda_<modelo>_fold1.db")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--ventanas-fijas", type=float, nargs="*", default=[60.0], help="ademas del recorte")
    ap.add_argument("--fold-busqueda", type=int, default=1)
    ap.add_argument("--semilla", type=int, default=0)
    args, resto = ap.parse_known_args()

    p, f1 = mejor_combinacion(args.busqueda, args.modelo, args.fold_busqueda)
    print(f"{args.modelo}: mejor combinacion de la busqueda (F1 macro volcanicas {f1:.3f} en el fold {args.fold_busqueda}): {p}", flush=True)
    os.makedirs(args.salida, exist_ok=True)
    for fija in [None] + list(args.ventanas_fijas):
        for fold in FOLDS:
            base = (["--datos", args.datos, "--modelo", args.modelo, "--fold", str(fold), "--salida", args.salida,
                     "--semilla", str(args.semilla)] + bc.PROTOCOLO + resto
                    + ["--evaluar-test", "--guardar-pesos"] + (["--ventana-fija", f"{fija:g}"] if fija else []))
            a = ec.argumentos(bc.argumentos_prueba(p, base))
            ruta = os.path.join(args.salida, ec.nombre_de(a) + ".json")
            if os.path.exists(ruta):
                print(f"  ya esta: {os.path.basename(ruta)}", flush=True)
                continue
            ec.entrenar(a)
    with open(os.path.join(args.salida, f"combinacion_{args.modelo}.json"), "w") as f:
        json.dump(dict(modelo=args.modelo, combinacion=p, f1_busqueda=f1, protocolo=bc.PROTOCOLO, extra=resto), f, indent=1)


if __name__ == "__main__":
    main()
