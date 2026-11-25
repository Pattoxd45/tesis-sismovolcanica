#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Casos sinteticos para paquete_colab.py y entrenar_cnn.py: folds de Jose por ventana, metricas desde la
matriz de confusion, imagen de entrada, dropout, perdida focal, optimizadores, parada temprana y
corridas cortas con pesos al azar (sin GPU ni descargas), incluida la evaluacion en test del mejor modelo.

Uso: python test_clasificacion.py
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import paquete_colab as pc


def caso(nombre, ok):
    print(f"  {'OK   ' if ok else 'FALLA'} {nombre}")
    return ok


def splits(d, val, desarrollo):
    for k, vs in enumerate(val, 1):
        with open(os.path.join(d, f"fold_{k:02d}_val.txt"), "w") as f:
            f.writelines(f"/ruta/{v}_300s.png\n" for v in vs)
    with open(os.path.join(d, "desarrollo.txt"), "w") as f:
        f.writelines(f"/ruta/{v}_300s.png\n" for v in desarrollo)


def detenido(f, *a):
    try:
        f(*a)
        return False
    except SystemExit:
        return True


def paquete_en_db(ec, er, rng):
    """Paquete en dB con seis ventanas sinteticas (una por fold de validacion y una de test), comparado con los PNG
    hechos como en espectrogramas_recortes.py; entrenamiento corto con jitter del rango de dB y ElasticTransform."""
    import re
    import zipfile

    import torch
    from scipy.signal import butter

    todo = True
    d = tempfile.mkdtemp()
    dv, dimg, dfijo, dsp = (os.path.join(d, x) for x in ("zne", "img", "img_fijo", "splits"))
    for x in (dv, dimg, dfijo, dsp):
        os.makedirs(x)
    sos = butter(er.ORDEN_FILTRO, list(er.BANDA_FILTRO), btype="bandpass", fs=er.FS, output="sos")
    t = np.arange(30000) / 100.0
    nombres = [f"v{k}" for k in range(1, 7)]
    filas, fijas, colores = [], [], True
    for j, v in enumerate(nombres):
        sig = {c: rng.normal(0, 1, t.size) + 40 * np.sin(2 * np.pi * (3 + j) * t) * ((t > 40) & (t < 70)) for c in "ZNE"}
        np.savez(os.path.join(dv, f"{v}_zne.npz"), **sig)
        esp = {c: er.espectrograma(sig[c], sos) for c in "ZNE"}
        dbs, lims = {c: x[0] for c, x in esp.items()}, {c: (x[1], x[2]) for c, x in esp.items()}
        for k, (et, a, b) in enumerate((("VT", 40.0, 52.0), ("LP", 45.0, 75.0), ("TR", 100.0, 250.0), ("sin caja", 260.0, 270.0))):
            archivo = f"{v}_{k:02d}.png"
            col = er.columnas(a * 100, b * 100, dbs["Z"].shape[1])
            im = er.imagen(dbs, lims, col)
            colores &= bool((ec.colorear(dbs, lims, col) == im).all())
            Image.fromarray(im).save(os.path.join(dimg, archivo))
            af = min(a, 240.0)
            Image.fromarray(er.imagen(dbs, lims, er.columnas(af * 100, (af + 60) * 100, dbs["Z"].shape[1]))).save(
                os.path.join(dfijo, archivo))
            filas.append(dict(recorte=archivo[:-4], ventana=v, split="test" if v == "v6" else "desarrollo", k=k, ini_s=a,
                              fin_s=b, dur_s=b - a, columnas=col.size, archivo=archivo, iou=0.5, etiqueta=et))
            fijas.append(dict(archivo=archivo, ini_fijo_s=af, fin_fijo_s=af + 60))
    todo &= caso("colorear de entrenar_cnn.py da la misma imagen que espectrogramas_recortes.py", colores)

    splits(dsp, [[v] for v in nombres[:5]], nombres[:5])
    pd.DataFrame(filas).to_csv(os.path.join(d, "indice.csv"), index=False)
    pd.DataFrame(fijas).to_csv(os.path.join(d, "fijo.csv"), index=False)
    ruta_zip = os.path.join(d, "clasificacion_db.zip")
    r = subprocess.run([sys.executable, os.path.join(AQUI, "paquete_db.py"), "--indice", os.path.join(d, "indice.csv"),
                        "--indice-fijo", os.path.join(d, "fijo.csv"), "--ventanas", dv, "--imagenes", dimg,
                        "--imagenes-fijo", dfijo, "--splits", dsp, "--salida", ruta_zip, "--verificar", "6"],
                        capture_output=True, text=True)
    m = re.search(r"Comparacion con los PNG: (\d+) imagenes; (\d+) con algun pixel distinto \((\d+) pixeles", r.stdout)
    ok = r.returncode == 0 and m is not None
    if ok:
        with zipfile.ZipFile(ruta_zip) as z:
            z.extractall(d)
        paq = os.path.join(d, "clasificacion_db")
        idx = pd.read_csv(os.path.join(paq, "indice.csv"))
        # 48 imagenes de 231 filas; se acepta como mucho un pixel distinto cada mil
        ok = (int(m.group(1)) == 48 and int(m.group(3)) <= 0.001 * 231 * idx.columnas.sum() * 2
              and len(os.listdir(os.path.join(paq, "ventanas"))) == 6
              and (idx.col_fin - idx.col_ini == idx.columnas).all()
              and (idx.col_fin_fijo60s - idx.col_ini_fijo60s).between(92, 95).all()
              and idx.fold.astype("object").fillna(0).tolist() == [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 4 + [0] * 4)
        print(f"    (paquete en dB: {m.group(2)} de 48 imagenes con algun pixel distinto de los PNG; {m.group(3)} pixeles)")
    else:
        print(r.stdout[-400:], r.stderr[-800:])
    todo &= caso("paquete en dB: una ventana por archivo, columnas de cada recorte y del tramo fijo, folds de Jose e "
                 "imagenes iguales a los PNG", ok)
    if not ok:
        return False

    aum = ec.Aumento(5.0, (50.0, 70.0))
    torch.manual_seed(0)
    s = np.array([aum.rango() for _ in range(500)])
    todo &= caso("jitter: corrimiento entre -5 y 5 dB y ancho entre 50 y 70 dB; sin jitter, 0 y 60 dB",
                 s[:, 0].min() >= -5 and s[:, 0].max() <= 5 and s[:, 0].min() < -4.5 and s[:, 0].max() > 4.5
                 and s[:, 1].min() >= 50 and s[:, 1].max() <= 70 and ec.Aumento().rango() == (0.0, 60.0))

    base = ["--modelo", "efficientnet_b0", "--fold", "2", "--epocas", "1", "--lote", "4", "--trabajadores", "0",
            "--sin-preentrenar"]
    for extra, archivo in ((["--jitter-maximo", "5", "--jitter-ancho", "50", "70", "--elastica", "50", "0"],
                            "efficientnet_b0_fold2_estirar_lote4_jit5_ancho50-70_ela50-0.json"),
                           (["--ventana-fija", "60", "--evaluar-test"], "efficientnet_b0_fold2_estirar_fijo60s_lote4.json")):
        salida = tempfile.mkdtemp()
        r = subprocess.run([sys.executable, os.path.join(AQUI, "entrenar_cnn.py"), "--datos", paq, "--salida", salida,
                            *base, *extra], capture_output=True, text=True)
        ruta = os.path.join(salida, archivo)
        ok = r.returncode == 0 and os.path.exists(ruta)
        if ok:
            res = json.load(open(ruta))
            cfg = res["configuracion"]
            ok = (cfg["paquete"] == "dB" and sum(cfg["n_entrenamiento"].values()) == 16
                  and sum(cfg["n_validacion"].values()) == 3 and cfg["n_sin_caja_validacion"] == 1
                  and ("test" not in res or sum(res["test"]["n"].values()) == 3))
        else:
            print(r.stderr[-800:])
        todo &= caso(f"entrenamiento con el paquete en dB {' '.join(extra)}", ok)

    r = subprocess.run([sys.executable, os.path.join(AQUI, "entrenar_cnn.py"), "--datos", dimg, "--salida", tempfile.mkdtemp(),
                        *base, "--jitter-maximo", "5"], capture_output=True, text=True)
    todo &= caso("jitter del rango de dB con el paquete de imagenes: se detiene", r.returncode != 0 and "DETENIDO" in r.stderr)

    try:
        import optuna  # noqa: F401
    except ImportError:
        print("  (sin Optuna: se omite el caso de buscar_cnn.py)")
        return todo
    salida = tempfile.mkdtemp()
    busq = [sys.executable, os.path.join(AQUI, "buscar_cnn.py"), "--datos", paq, "--modelo", "efficientnet_b0", "--fold", "2",
            "--salida", salida, "--epocas", "1", "--trabajadores", "0", "--sin-preentrenar", "--jitter-maximo", "5"]
    r1 = subprocess.run(busq + ["--pruebas", "2"], capture_output=True, text=True)
    r2 = subprocess.run(busq + ["--pruebas", "3"], capture_output=True, text=True)
    ruta = os.path.join(salida, "busqueda_efficientnet_b0_fold2.csv")
    ok = r1.returncode == 0 and r2.returncode == 0 and os.path.exists(ruta) and "2 pruebas terminadas de 3" in r2.stdout
    if ok:
        t = pd.read_csv(ruta)
        js = [json.load(open(os.path.join(salida, j))) for j in t.json]
        ok = (len(t) == 3 and (t.state == "COMPLETE").all()
              and all(j["configuracion"]["epocas"] == 1 and j["configuracion"]["paciencia"] == 5
                      and j["configuracion"]["jitter_maximo"] == 5 and j["configuracion"]["sin_falsas_alarmas"]
                      and j["configuracion"]["pesos_clase"] == (j["configuracion"]["perdida"] == "entropia")
                      and (j["configuracion"]["momentum"] == 0.9) == (j["configuracion"]["optimizador"] == "sgd") for j in js))
    else:
        print(r1.stderr[-800:], r2.stderr[-800:])
    todo &= caso("busqueda con Optuna: 2 pruebas, se relanza y sigue hasta 3; protocolo fijo y argumentos extra pasan", ok)
    return todo


def main():
    todo = True
    d = tempfile.mkdtemp()
    splits(d, [["a"], ["b"], ["c"], ["d"], ["e", "f"]], list("abcdef"))
    fold = pc.folds_jose(d)
    todo &= caso("fold de validacion de cada ventana de desarrollo", fold == dict(a=1, b=2, c=3, d=4, e=5, f=5))
    splits(d, [["a"], ["a"], ["c"], ["d"], ["e", "f"]], list("acdef"))
    todo &= caso("ventana en dos folds: se detiene", detenido(pc.folds_jose, d))
    splits(d, [["a"], ["b"], ["c"], ["d"], ["e"]], list("abcdef"))
    todo &= caso("folds que no cubren desarrollo: se detiene", detenido(pc.folds_jose, d))
    idx = pd.DataFrame(dict(ventana=["a", "a", "t"], split=["desarrollo", "desarrollo", "test"]))
    todo &= caso("recortes de una ventana con su fold; test sin fold",
                 pc.indice_con_folds(idx, dict(a=3)).fold.astype("object").fillna(0).tolist() == [3, 3, 0])
    todo &= caso("ventana de desarrollo sin fold: se detiene",
                 detenido(pc.indice_con_folds, pd.DataFrame(dict(ventana=["z"], split=["desarrollo"])), dict(a=3)))

    import duraciones_recortes as dr
    dur = pd.Series([1.0, 3.0, 6.0, 12.0, 50.0, 70.0])
    et = pd.Series(["VT", "VT", "VT", "TR", "TR", "TR"])
    fm = dr.fraccion(dur, et, [5], mayor=False)
    ff = dr.fraccion(dur, et, [60], mayor=True)
    todo &= caso("duraciones: parte mas corta que el minimo y parte que se cortaria, por etiqueta",
                 abs(fm.loc["VT", "5 s"] - 2 / 3) < 1e-3 and fm.loc["TR", "5 s"] == 0
                 and ff.loc["VT", "60 s"] == 0 and abs(ff.loc["TR", "60 s"] - 1 / 3) < 1e-3
                 and dr.percentiles(dur, et).loc["TR", "n"] == 3)

    import espectrogramas_fijos as ef
    todo &= caso("tramo fijo: desde el inicio del recorte; al final de la ventana, sus ultimos segundos",
                 ef.tramo_fijo(10.0, 60.0) == (10.0, 70.0) and ef.tramo_fijo(280.0, 60.0) == (240.0, 300.0)
                 and ef.tramo_fijo(0.0, 60.0) == (0.0, 60.0))
    todo &= caso("carpeta dentro del zip: el nombre del zip",
                 pc.raiz("/x/clasificacion_colab_fijo60s.zip") == "clasificacion_colab_fijo60s")

    import resumen_cnn as rs
    import correr_todo as ct
    d = tempfile.mkdtemp()
    for f, ep in zip(ct.FOLDS, [10, 7, 9, 7, 8]):
        json.dump(dict(mejor_epoca=ep), open(os.path.join(d, ct.nombre("resnet50", f, ct.ESTIRAR) + ".json"), "w"))
    todo &= caso("corredor: 20 corridas de folds y epocas finales = mediana de las mejores (10, 7, 9, 7, 8 -> 8)",
                 len(ct.trabajos_cv()) == 20 and ct.epocas_final(d, "resnet50") == (8, [10, 7, 9, 7, 8])
                 and ct.nombre("efficientnet_b0", 0, ct.FIJO) == "efficientnet_b0_test_estirar_fijo60s_pesos_completo_lr0.0001_min5s_sin_falsas")
    m = {"f1_macro_volcanicas": 0.7, "matriz": [[8, 1, 0, 1], [0, 9, 0, 1], [0, 0, 4, 1], [2, 1, 0, 7]]}
    json.dump(dict(configuracion=dict(modelo="vgg16", fold=0, conjunto="recorte", evaluacion="test",
                                      cortos_fuera_de_validacion={"VT": 0, "LP": 0, "TR": 0, "rechazo": 10}),
                   historia=[dict(m, segundos=5.0)], mejor_epoca=1), open(os.path.join(d, "x.json"), "w"))
    r = rs.leer(os.path.join(d, "x.json"))
    todo &= caso("resumen: los recortes cortos cuentan como rechazados (3 falsos de 20) y la sensibilidad sobre el total",
                 abs(r["falsos"] - 3 / 20) < 1e-12 and abs(r["volcanicos"] - 21 / 25) < 1e-12 and r["sens_VT"] == 0.8
                 and r["n_rechazo"] == 20)

    import hoja_ejemplos as he
    d = tempfile.mkdtemp()
    preds = dict(efficientnet_b0=["LP", "TR", "VT", "rechazo"], resnet50=["LP", "rechazo", "VT", "rechazo"],
                 vgg16=["TR", "rechazo", "VT", "rechazo"])
    for m, pr in preds.items():
        pd.DataFrame(dict(archivo=["a_00.png", "a_01.png", "b_00.png", "b_01.png"], ventana=["a", "a", "b", "b"],
                          etiqueta=["sin caja", "sin caja", "LP", "sin caja"], clase=["rechazo", "rechazo", "LP", "rechazo"],
                          prediccion=pr, p_VT=0.1, p_LP=0.6, p_TR=0.2, p_rechazo=0.1)).to_csv(
            os.path.join(d, f"{m}_fold1_estirar_pesos_completo_lr0.0001_min5s_sin_falsas_predicciones.csv"), index=False)
    g = he.grupos(he.unir(d))
    todo &= caso("hoja de ejemplos: sin caja que las tres redes llaman volcanico (1), que las tres rechazan (1) y LP que EfficientNet-B0 llama VT (1)",
                 [int(v.sum()) for v in g.values()] == [1, 1, 0, 0, 1] and g["Sin caja: las tres redes dicen volcanico"].tolist() == [True, False, False, False])

    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        print("  (sin PyTorch o torchvision: se omiten los casos de entrenar_cnn.py)")
        print("TODOS PASAN" if todo else "HAY FALLAS")
        sys.exit(0 if todo else 1)
    import entrenar_cnn as ec

    c = np.array([[8, 1, 1, 0], [0, 5, 0, 5], [0, 0, 0, 2], [2, 0, 0, 16]])
    m = ec.metricas(c)
    todo &= caso("metricas: sensibilidad, precision y F1 por clase; F1 macro solo de VT, LP y TR",
                 abs(m["VT"]["sensibilidad"] - 0.8) < 1e-12 and abs(m["VT"]["precision"] - 0.8) < 1e-12
                 and abs(m["LP"]["f1"] - 2 * (5 / 6) * 0.5 / (5 / 6 + 0.5)) < 1e-12 and m["TR"]["f1"] == 0.0
                 and abs(m["f1_macro_volcanicas"] - (0.8 + m["LP"]["f1"]) / 3) < 1e-12 and abs(m["exactitud"] - 29 / 40) < 1e-12)

    di = tempfile.mkdtemp()
    Image.fromarray(np.full((231, 40, 3), 200, dtype=np.uint8)).save(os.path.join(di, "r.png"))
    for tam in ("estirar", "rellenar"):
        x, y = ec.Recortes(di, ["r.png"], [2], tam, [0, 0, 0], [1, 1, 1])[0]
        negro = float((x.sum(0) == 0).float().mean())
        todo &= caso(f"imagen {tam}: 3 x 224 x 224; " + ("sin fondo negro" if tam == "estirar" else "el recorte ocupa 40 de 465 columnas"),
                     tuple(x.shape) == (3, 224, 224) and y == 2
                     and (negro == 0 if tam == "estirar" else abs((1 - negro) - 40 / 465) < 0.02))

    for mod in ec.CABEZAS:
        red, cabeza, *_ = ec.construir(mod, 4, preentrenado=False)
        todo &= caso(f"{mod}: la capa nueva tiene 4 salidas", red(torch.zeros(1, 3, 224, 224)).shape == (1, 4)
                     and cabeza.out_features == 4)

    red, cabeza, *_ = ec.construir("resnet50", 4, preentrenado=False)
    ec.modo_entrenamiento(red, cabeza, congelada=True)
    todo &= caso("red congelada: BatchNorm en evaluacion y la capa nueva entrenando",
                 not red.bn1.training and cabeza.training)

    datos = tempfile.mkdtemp()
    os.makedirs(os.path.join(datos, "imagenes"))
    rng = np.random.default_rng(0)
    filas = []
    for i in range(60):
        et = ["VT", "LP", "TR", "IC", "sin caja", "fuera"][i % 6]
        a = f"v{i // 3}_{i % 3:02d}.png"
        Image.fromarray(rng.integers(0, 255, (231, 10 + i, 3), dtype=np.uint8)).save(os.path.join(datos, "imagenes", a))
        filas.append(dict(archivo=a, ventana=f"v{i // 3}", etiqueta=et, split="desarrollo" if i < 48 else "test",
                          fold=(i // 3) % 5 + 1 if i < 48 else pd.NA, dur_s=10.0 + i if i < 48 else 1.0 + 2 * (i - 48)))
    pd.DataFrame(filas).to_csv(os.path.join(datos, "indice.csv"), index=False)
    # sin 'fuera' quedan 40 de desarrollo: 7 del fold 2 (validacion) y 33 de los otros folds
    for modelo, extra, archivo, congelada in (
            ("efficientnet_b0", ["--pesos-clase"], "efficientnet_b0_fold2_estirar_pesos_lote8.json", True),
            ("resnet50", ["--tamano", "rellenar", "--ajuste-completo"], "resnet50_fold2_rellenar_completo_lote8.json", False),
            ("efficientnet_b0", ["--ajuste-completo", "--lr", "0.0001"], "efficientnet_b0_fold2_estirar_completo_lr0.0001_lote8.json", False),
            ("vgg16", ["--ajuste-completo", "--acumular", "4"], "vgg16_fold2_estirar_completo_lote8.json", False)):
        salida = tempfile.mkdtemp()
        r = subprocess.run([sys.executable, os.path.join(AQUI, "entrenar_cnn.py"), "--datos", datos, "--modelo", modelo,
                            "--fold", "2", "--salida", salida, "--epocas", "2", "--lote", "8", "--trabajadores", "0",
                            "--sin-preentrenar", *extra], capture_output=True, text=True)
        ok = r.returncode == 0
        if ok:
            res = json.load(open(os.path.join(salida, archivo)))
            cfg = res["configuracion"]
            # validacion del fold 2: 5 eventos (VT, LP, TR y dos IC) y 2 sin caja, que se cuentan aparte
            ok = (len(res["historia"]) == 2 and sum(cfg["n_validacion"].values()) == 5 and cfg["n_sin_caja_validacion"] == 2
                  and sum(cfg["n_entrenamiento"].values()) == 33 and cfg["congelada"] == congelada
                  and sum(map(sum, res["historia"][0]["matriz"])) == 5
                  and sum(res["historia"][0]["sin_caja_predichos"].values()) == 2
                  and cfg["lr_usada"] == (1e-4 if "--lr" in extra else 1e-3))
        else:
            print(r.stderr[-800:])
        todo &= caso(f"corrida corta {modelo} {' '.join(extra)}: fold 2 a validacion, 'fuera' y test fuera, JSON con 2 epocas", ok)

    # el mismo conjunto con nombre de paquete de duracion fija y una duracion minima de 30 s:
    # de la validacion del fold 2 salen 4 recortes de 13, 14, 28 y 29 s (IC, sin caja, VT y LP)
    fijo = os.path.join(tempfile.mkdtemp(), "clasificacion_colab_fijo60s")
    os.symlink(datos, fijo)
    salida = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(AQUI, "entrenar_cnn.py"), "--datos", fijo, "--modelo", "efficientnet_b0",
                        "--fold", "2", "--salida", salida, "--epocas", "1", "--lote", "8", "--trabajadores", "0",
                        "--sin-preentrenar", "--duracion-minima", "30"], capture_output=True, text=True)
    ruta = os.path.join(salida, "efficientnet_b0_fold2_estirar_fijo60s_min30s_lote8.json")
    ok = r.returncode == 0 and os.path.exists(ruta)
    if ok:
        cfg = json.load(open(ruta))["configuracion"]
        ok = (cfg["conjunto"] == "fijo60s" and sum(cfg["n_validacion"].values()) == 2 and cfg["n_sin_caja_validacion"] == 1
              and cfg["cortos_fuera_de_validacion"] == {"VT": 1, "LP": 1, "TR": 0, "rechazo": 1} and cfg["cortos_sin_caja"] == 1)
    else:
        print(r.stderr[-800:])
    todo &= caso("conjunto de duracion fija en el nombre y duracion minima que saca los recortes cortos", ok)

    from torch import nn
    from torch.utils.data import DataLoader
    x = torch.randn(8, 5, generator=torch.Generator().manual_seed(0))
    y = torch.tensor([0, 1, 2, 3, 0, 0, 3, 3])
    peso = torch.tensor([2.0, 1.0, 0.5, 0.25])

    def un_paso(k):
        torch.manual_seed(1)
        red = nn.Linear(5, 4)
        ref = nn.CrossEntropyLoss(weight=peso)(red(x), y).item()
        opt = torch.optim.AdamW(red.parameters(), lr=0.1)
        t = 8 // k
        l = ec.paso(red, *ec.perdida_y_cuenta("entropia", peso, 4),
                    [(x[i:i + t], y[i:i + t]) for i in range(0, 8, t)], opt, torch.device("cpu"))
        return red.weight.detach().clone(), l, ref

    w1, l1, ref = un_paso(1)
    w2, l2, _ = un_paso(2)
    w4, l4, _ = un_paso(4)
    todo &= caso("lote en 1, 2 o 4 partes: misma perdida ponderada y mismos pesos despues del paso",
                 abs(l1 - ref) < 1e-6 and abs(l2 - ref) < 1e-6 and abs(l4 - ref) < 1e-6
                 and torch.allclose(w1, w2, atol=1e-6) and torch.allclose(w1, w4, atol=1e-6))

    # sin falsas alarmas: el entrenamiento pierde los 6 sin caja de los otros folds (33 -> 27); la validacion no cambia
    salida = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(AQUI, "entrenar_cnn.py"), "--datos", datos, "--modelo", "efficientnet_b0",
                        "--fold", "2", "--salida", salida, "--epocas", "1", "--lote", "8", "--trabajadores", "0",
                        "--sin-preentrenar", "--sin-falsas-alarmas"], capture_output=True, text=True)
    ruta = os.path.join(salida, "efficientnet_b0_fold2_estirar_sin_falsas_lote8.json")
    ok = r.returncode == 0 and os.path.exists(ruta)
    if ok:
        cfg = json.load(open(ruta))["configuracion"]
        pr = pd.read_csv(ruta[:-5] + "_predicciones.csv")
        ok = (sum(cfg["n_entrenamiento"].values()) == 27 and sum(cfg["n_validacion"].values()) == 5
              and len(pr) == 7 and (pr.etiqueta == "sin caja").sum() == 2
              and list(pr.columns) == ["archivo", "ventana", "etiqueta", "clase", "prediccion", "p_VT", "p_LP", "p_TR", "p_rechazo"]
              and abs(pr[["p_VT", "p_LP", "p_TR", "p_rechazo"]].sum(axis=1) - 1).max() < 1e-5)
    else:
        print(r.stderr[-800:])
    todo &= caso("sin falsas alarmas: entrena sin los sin caja y guarda la prediccion y las probabilidades de cada recorte", ok)

    todo &= caso("el corredor nombra los resultados igual que entrenar_cnn.py",
                 all(ct.nombre(m, f, c) == ec.nombre_resultado(m, f, "estirar", ec.conjunto_de(c), True, True, 1e-4, 5.0, True)
                     for m, f, c in ct.trabajos_cv() + [(m, 0, ct.ESTIRAR) for m, _ in ct.REDES]))

    # fold 0: entrena con todo desarrollo (40 sin 'fuera') y evalua en test (10); con minimo 5 s salen de test
    # los de 1 y 3 s (un VT y un LP); informa la ultima epoca y guarda los pesos
    salida = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(AQUI, "entrenar_cnn.py"), "--datos", datos, "--modelo", "efficientnet_b0",
                        "--fold", "0", "--salida", salida, "--epocas", "2", "--lote", "8", "--trabajadores", "0",
                        "--sin-preentrenar", "--duracion-minima", "5", "--guardar-pesos"], capture_output=True, text=True)
    ruta = os.path.join(salida, "efficientnet_b0_test_estirar_min5s_lote8.json")
    ok = r.returncode == 0 and os.path.exists(ruta) and os.path.exists(ruta[:-5] + ".pt")
    if ok:
        res = json.load(open(ruta))
        cfg = res["configuracion"]
        ok = (cfg["evaluacion"] == "test" and res["mejor_epoca"] == 2 and sum(cfg["n_validacion"].values()) == 6
              and cfg["n_sin_caja_validacion"] == 2
              and sum(cfg["n_entrenamiento"].values()) == 40
              and cfg["cortos_fuera_de_validacion"] == {"VT": 1, "LP": 1, "TR": 0, "rechazo": 0})
    else:
        print(r.stderr[-800:])
    todo &= caso("fold 0: entrena con desarrollo, evalua en test la ultima epoca, cuenta los cortos de test y guarda pesos", ok)

    esperado = {"vgg16": ([0.5, 0.5], [0.3, 0.3]), "efficientnet_b0": ([0.2], [0.3]), "resnet50": ([], [0.3])}
    for mod, (base, con) in esperado.items():
        r0, _, *_ = ec.construir(mod, 4, preentrenado=False)
        r3, cab, *_ = ec.construir(mod, 4, preentrenado=False, dropout=0.3)
        todo &= caso(f"{mod}: dropout por defecto {base}; con --dropout 0.3 queda {con} y sigue con 4 salidas",
                     ec.dropouts(r0) == base and ec.dropouts(r3) == con and r3.eval()(torch.zeros(1, 3, 224, 224)).shape == (1, 4)
                     and sum(p.numel() for p in cab.parameters()) == sum(p.numel() for p in ec.construir(mod, 4, False)[1].parameters()))

    from torchvision.ops import sigmoid_focal_loss

    def paso_focal(k):
        torch.manual_seed(1)
        red = nn.Linear(5, 4)
        ref = sigmoid_focal_loss(red(x), nn.functional.one_hot(y, 4).float(), reduction="mean").item()
        opt = torch.optim.SGD(red.parameters(), lr=0.1)
        t = 8 // k
        l = ec.paso(red, *ec.perdida_y_cuenta("focal", None, 4), [(x[i:i + t], y[i:i + t]) for i in range(0, 8, t)],
                    opt, torch.device("cpu"))
        return red.weight.detach().clone(), l, ref

    (w1, l1, ref), (w4, l4, _) = paso_focal(1), paso_focal(4)
    todo &= caso("focal en 1 o 4 partes: igual a la de torchvision con reduction='mean' y mismos pesos despues del paso",
                 abs(l1 - ref) < 1e-6 and abs(l4 - ref) < 1e-6 and torch.allclose(w1, w4, atol=1e-6))

    ps = [nn.Parameter(torch.zeros(2))]
    a, s = ec.optimizador("adamw", ps), ec.optimizador("sgd", ps, lr=0.01, momentum=0.9, weight_decay=1e-4)
    todo &= caso("optimizadores: AdamW con sus valores de PyTorch; SGD con lr, momentum y weight_decay dados",
                 isinstance(a, torch.optim.AdamW) and a.defaults["lr"] == 1e-3 and a.defaults["weight_decay"] == 1e-2
                 and isinstance(s, torch.optim.SGD) and s.defaults["momentum"] == 0.9 and s.defaults["weight_decay"] == 1e-4
                 and detenido(ec.optimizador, "adamw", ps, None, 0.9))

    h = [dict(f1_macro_volcanicas=v) for v in (0.1, 0.3, 0.3, 0.2)]
    todo &= caso("parada temprana: mejor en la epoca 2; con paciencia 2 se detiene en la 4, con 3 sigue, sin paciencia sigue",
                 ec.detener(h, 2) and not ec.detener(h, 3) and not ec.detener(h, None) and not ec.detener(h[:3], 2))

    # fold 2 con SGD, weight decay, dropout, lote 8, paciencia 1 y el mejor modelo evaluado en test; con minimo 5 s
    # test queda con 6 eventos (VT 1, LP 1, TR 2, rechazo 2) y 2 sin caja, y salen un VT y un LP cortos
    salida = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(AQUI, "entrenar_cnn.py"), "--datos", datos, "--modelo", "resnet50",
                        "--fold", "2", "--salida", salida, "--epocas", "6", "--lote", "8", "--trabajadores", "0",
                        "--sin-preentrenar", "--ajuste-completo", "--optimizador", "sgd", "--lr", "0.01", "--momentum", "0.9",
                        "--weight-decay", "0.0001", "--dropout", "0.3", "--paciencia", "1", "--evaluar-test",
                        "--duracion-minima", "5", "--guardar-pesos"], capture_output=True, text=True)
    ruta = os.path.join(salida, "resnet50_fold2_estirar_completo_lr0.01_min5s_sgd_mom0.9_wd0.0001_do0.3_lote8_pac1.json")
    ok = r.returncode == 0 and os.path.exists(ruta) and os.path.exists(ruta[:-5] + ".pt")
    if ok:
        res = json.load(open(ruta))
        cfg, t, hist = res["configuracion"], res["test"], res["historia"]
        f1 = [e["f1_macro_volcanicas"] for e in hist]
        pt = pd.read_csv(ruta[:-5] + "_test_predicciones.csv")
        ok = (cfg["momentum_usado"] == 0.9 and cfg["weight_decay_usado"] == 1e-4 and cfg["dropout_red"] == [0.3]
              and res["epocas_corridas"] == len(hist) and res["mejor_epoca"] == int(np.argmax(f1)) + 1
              and (len(hist) == 6 or len(hist) - res["mejor_epoca"] == 1)
              and t["epoca"] == res["mejor_epoca"] and t["n"] == {"VT": 1, "LP": 1, "TR": 2, "rechazo": 2}
              and t["n_sin_caja"] == 2 and t["cortos"] == {"VT": 1, "LP": 1, "TR": 0, "rechazo": 0}
              and sum(map(sum, t["matriz"])) == 6 and sum(t["sin_caja_predichos"].values()) == 2
              and len(pt) == 8 and set(pt.ventana).isdisjoint(set(pd.read_csv(ruta[:-5] + "_predicciones.csv").ventana)))
        # los pesos guardados son los de la mejor epoca: con ellos sale la misma prediccion de test
        red_pt = ec.construir("resnet50", 4, False, 0.3)[0]
        red_pt.load_state_dict(torch.load(ruta[:-5] + ".pt"))
        carga = DataLoader(ec.Recortes(os.path.join(datos, "imagenes"), pt.archivo, [0] * len(pt), "estirar",
                                       [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), batch_size=8)
        ok = ok and (np.array(ec.CLASES)[ec.predecir(red_pt, carga, torch.device("cpu"),
                                                     lambda z: torch.softmax(z, 1)).argmax(1)] == pt.prediccion).all()
    else:
        print(r.stderr[-800:])
    todo &= caso("SGD con paciencia 1: se detiene una epoca despues de la mejor, evalua en test ese modelo y guarda sus pesos", ok)

    salida = tempfile.mkdtemp()
    r = subprocess.run([sys.executable, os.path.join(AQUI, "entrenar_cnn.py"), "--datos", datos, "--modelo", "efficientnet_b0",
                        "--fold", "2", "--salida", salida, "--epocas", "1", "--lote", "8", "--trabajadores", "0",
                        "--sin-preentrenar", "--perdida", "focal"], capture_output=True, text=True)
    ruta = os.path.join(salida, "efficientnet_b0_fold2_estirar_lote8_focal.json")
    ok = r.returncode == 0 and os.path.exists(ruta)
    if ok:
        pr = pd.read_csv(ruta[:-5] + "_predicciones.csv")
        p = pr[["p_VT", "p_LP", "p_TR", "p_rechazo"]].to_numpy()
        ok = (json.load(open(ruta))["configuracion"]["probabilidad"] == "sigmoide" and ((p >= 0) & (p <= 1)).all()
              and (pr.prediccion == np.array(ec.CLASES)[p.argmax(1)]).all())
    else:
        print(r.stderr[-800:])
    todo &= caso("perdida focal: corre y guarda la sigmoide de cada clase; la prediccion es la clase mas alta", ok)

    for fold, extra, que in (("2", ["--perdida", "focal", "--pesos-clase"], "focal con pesos por clase"),
                             ("0", ["--paciencia", "5"], "fold 0 con paciencia"),
                             ("0", ["--evaluar-test"], "fold 0 con evaluar test")):
        r = subprocess.run([sys.executable, os.path.join(AQUI, "entrenar_cnn.py"), "--datos", datos, "--modelo",
                            "efficientnet_b0", "--fold", fold, "--salida", tempfile.mkdtemp(), "--epocas", "1", "--lote", "8",
                            "--trabajadores", "0", "--sin-preentrenar", *extra], capture_output=True, text=True)
        todo &= caso(f"{que}: se detiene sin entrenar", r.returncode != 0 and "DETENIDO" in r.stderr)

    import figuras_aumento as fa
    sel = pd.DataFrame(dict(archivo=[f"w_{i}.png" for i in range(7)], ventana="w",
                            split=["desarrollo"] * 6 + ["test"], etiqueta=["VT", "VT", "VT", "LP", "TR", "TR", "LP"],
                            dur_s=[5.0, 10.0, 12.0, 30.0, 150.0, 220.0, 30.0], columnas=[8, 16, 19, 47, 233, 344, 47]))
    e = fa.elegir(sel)
    todo &= caso("figuras: corto = VT de menos de 224 columnas con duracion mediana; largo = el mas cercano a 336 "
                 "columnas; evento = LP de desarrollo",
                 e["corto"].archivo == "w_1.png" and e["largo"].archivo == "w_5.png" and e["evento"].archivo == "w_3.png")
    todo &= caso("rango de dB: maximo corrido y ancho nuevo en cada componente",
                 fa.con_rango({"Z": (-60.0, 0.0), "N": (-50.0, 10.0)}, -10, 45) == {"Z": (-55.0, -10.0), "N": (-45.0, 0.0)})
    rayas = np.zeros((224, 224, 3), dtype=np.uint8)
    rayas[::8] = 255
    todo &= caso("ElasticTransform con alpha (50, 0) solo mueve en el tiempo: rayas horizontales iguales lejos del borde",
                 (fa.elastica(rayas, (50.0, 0.0), 5.0, 0)[:, 20:-20] == rayas[:, 20:-20]).all()
                 and not (fa.elastica(rayas, (50.0, 50.0), 5.0, 0)[:, 20:-20] == rayas[:, 20:-20]).all())

    # una ventana sintetica de 300 s con tres eventos: las tres figuras se generan
    dv = tempfile.mkdtemp()
    t = np.arange(30000) / 100.0
    sig = {c: rng.normal(0, 1, t.size) + 50 * np.sin(2 * np.pi * 5 * t) * ((t > 50) & (t < 62)) for c in "ZNE"}
    np.savez(os.path.join(dv, "w_zne.npz"), **sig)
    sel = pd.DataFrame(dict(archivo=["w_0.png", "w_1.png", "w_2.png"], ventana="w", split="desarrollo",
                            etiqueta=["VT", "TR", "LP"], ini_s=[50.0, 20.0, 120.0], fin_s=[62.0, 240.0, 150.0]))
    sel["dur_s"] = sel.fin_s - sel.ini_s
    sel["columnas"] = [fa.er.columnas(a * 100, b * 100, 465).size for a, b in zip(sel.ini_s, sel.fin_s)]
    sel.to_csv(os.path.join(dv, "indice.csv"), index=False)
    r = subprocess.run([sys.executable, os.path.join(AQUI, "figuras_aumento.py"), "--indice", os.path.join(dv, "indice.csv"),
                        "--ventanas", dv, "--salida", os.path.join(dv, "fig")], capture_output=True, text=True)
    todo &= caso("figuras de estiramiento, rangos de dB y ElasticTransform con una ventana sintetica",
                 r.returncode == 0 and all(os.path.exists(os.path.join(dv, "fig", f)) for f in
                                           ("figura_estiramiento.png", "figura_rangos_db.png", "figura_elastica.png")))
    if r.returncode:
        print(r.stderr[-800:])

    todo &= paquete_en_db(ec, fa.er, rng)

    print("TODOS PASAN" if todo else "HAY FALLAS")
    sys.exit(0 if todo else 1)


if __name__ == "__main__":
    main()
