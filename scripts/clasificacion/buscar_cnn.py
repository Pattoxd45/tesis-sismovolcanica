#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Busqueda bayesiana de hiperparametros (Optuna, muestreador TPE) para una red, en un fold de validacion.
Prueba optimizador (AdamW o SGD con momentum 0,9), tasa de aprendizaje, dropout, lote, weight decay y
perdida (entropia cruzada con pesos por clase o focal), con el protocolo fijo: ajuste completo, minimo
5 s, sin los sin caja en entrenamiento, maximo 50 epocas y paciencia 5. Cada prueba queda en un archivo
SQLite: si el trabajo se corta, al volver a lanzarlo sigue donde quedo. El objetivo es el F1 macro de VT,
LP y TR en validacion; la sensibilidad y el F1 de cada clase quedan en la tabla final.
Los argumentos que no son de este script (aumento, --trabajadores, etc.) pasan tal cual a entrenar_cnn.py.

Uso: python buscar_cnn.py --datos <carpeta> --modelo efficientnet_b0 --salida <carpeta> [--pruebas 20]
     [--fold 1] [--jitter-maximo 5 --jitter-ancho 50 70 --elastica 50 0] [--trabajadores 3]
"""

import argparse
import json
import os
import sys

import optuna

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import entrenar_cnn as ec

ESPACIO = dict(optimizador=["adamw", "sgd"], lr=[1e-2, 1e-3, 1e-4, 1e-5], dropout=[0.2, 0.3], lote=[16, 32],
               weight_decay=[0.0, 1e-4], perdida=["entropia", "focal"])
PROTOCOLO = ["--ajuste-completo", "--duracion-minima", "5", "--sin-falsas-alarmas", "--epocas", "50", "--paciencia", "5"]
# momentum habitual de SGD; el de PyTorch por defecto es 0, que es SGD sin momentum
MOMENTUM_SGD = "0.9"


def argumentos_prueba(p, base):
    """Argumentos de entrenar_cnn.py para una combinacion; los pesos por clase van solo con la entropia cruzada."""
    a = base + ["--optimizador", p["optimizador"], "--lr", f"{p['lr']:g}", "--dropout", f"{p['dropout']:g}",
                "--lote", str(p["lote"]), "--weight-decay", f"{p['weight_decay']:g}", "--perdida", p["perdida"]]
    if p["optimizador"] == "sgd":
        a += ["--momentum", MOMENTUM_SGD]
    if p["perdida"] == "entropia":
        a += ["--pesos-clase"]
    return a


def resultado(res):
    """F1 macro volcanicas y metricas por clase de la mejor epoca."""
    m = res["historia"][res["mejor_epoca"] - 1]
    return m["f1_macro_volcanicas"], {f"{k}_{x}": m[k][x] for k in ec.CLASES for x in ("sensibilidad", "f1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True)
    ap.add_argument("--modelo", required=True, choices=sorted(ec.CABEZAS))
    ap.add_argument("--salida", required=True)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--pruebas", type=int, default=20, help="pruebas terminadas que se quieren en total")
    ap.add_argument("--semilla", type=int, default=0)
    args, resto = ap.parse_known_args()

    os.makedirs(args.salida, exist_ok=True)
    base = ["--datos", args.datos, "--modelo", args.modelo, "--fold", str(args.fold), "--salida", args.salida,
            "--semilla", str(args.semilla)] + PROTOCOLO + resto
    estudio = optuna.create_study(study_name=f"{args.modelo}_fold{args.fold}", direction="maximize",
                                  storage=f"sqlite:///{os.path.join(args.salida, f'busqueda_{args.modelo}_fold{args.fold}.db')}",
                                  sampler=optuna.samplers.TPESampler(seed=args.semilla), load_if_exists=True)

    def objetivo(trial):
        p = {k: trial.suggest_categorical(k, v) for k, v in ESPACIO.items()}
        a = ec.argumentos(argumentos_prueba(p, base))
        ruta = os.path.join(args.salida, ec.nombre_de(a) + ".json")
        # una combinacion ya entrenada (el muestreador puede repetirla) no se vuelve a entrenar
        res = json.load(open(ruta)) if os.path.exists(ruta) else ec.entrenar(a)
        f1, por_clase = resultado(res)
        for k, v in dict(json=os.path.basename(ruta), mejor_epoca=res["mejor_epoca"], epocas=len(res["historia"]),
                         **por_clase).items():
            trial.set_user_attr(k, v)
        return f1

    hechas = len([t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"{args.modelo}, fold {args.fold}: {hechas} pruebas terminadas de {args.pruebas}", flush=True)
    if hechas < args.pruebas:
        # una prueba que falla (por ejemplo, sin memoria) queda registrada y la busqueda sigue
        estudio.optimize(objetivo, n_trials=args.pruebas - hechas, catch=(RuntimeError,))

    t = estudio.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    t.columns = [c.replace("params_", "").replace("user_attrs_", "") for c in t.columns]
    t = t.sort_values("value", ascending=False)
    t.to_csv(os.path.join(args.salida, f"busqueda_{args.modelo}_fold{args.fold}.csv"), index=False)
    cols = [c for c in ["number", "value", *ESPACIO, "mejor_epoca", "VT_sensibilidad", "LP_sensibilidad", "TR_sensibilidad"]
            if c in t.columns]
    print(t[cols].round(4).to_string(index=False))
    if any(x.state == optuna.trial.TrialState.COMPLETE for x in estudio.trials):
        print(f"\nMejor combinacion (F1 macro volcanicas {estudio.best_value:.3f}): {estudio.best_params}")


if __name__ == "__main__":
    main()
