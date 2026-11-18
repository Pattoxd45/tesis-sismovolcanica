#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ajuste por transferencia de una CNN preentrenada en ImageNet (VGG16, ResNet50 o EfficientNet-B0 de
torchvision) para clasificar los recortes en VT, LP, TR y rechazo, con un fold de la validacion
cruzada de Jose; con --fold 0 entrena con todo desarrollo y evalua una vez en test, en la ultima
epoca. Guarda un JSON con la configuracion, las versiones y las metricas de cada epoca.

Uso (Colab): python entrenar_cnn.py --datos /content/clasificacion_colab --modelo vgg16 --fold 1
             --salida /content/drive/MyDrive/tesis/resultados_cnn [--tamano estirar|rellenar]
             [--pesos-clase] [--ajuste-completo] [--lr 0.0001] [--duracion-minima 5] [--epocas 10]
             [--acumular 2] [--guardar-pesos]
"""

import argparse
import json
import os
import platform
import random
import time

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

CLASES = ["VT", "LP", "TR", "rechazo"]
VOLCANICAS = ["VT", "LP", "TR"]
# propuesta enviada al profesor: AV, IC y los recortes que no tocan ninguna caja van al rechazo;
# los que tocan una caja con IoU menor que 0,3 ("fuera") no se usan
MAPA = {"VT": "VT", "LP": "LP", "TR": "TR", "AV": "rechazo", "IC": "rechazo", "sin caja": "rechazo"}
# lado de entrada con que se preentrenaron las tres redes (crop_size 224 de sus transforms); no se usa el
# recorte central de esas transforms porque cortaria el inicio y el fin del evento
LADO = 224
# columnas del espectrograma de una ventana completa de 300 s (465 en las imagenes de Jose)
ANCHO_VENTANA = 465
CABEZAS = {"vgg16": ("classifier", 6), "resnet50": ("fc", None), "efficientnet_b0": ("classifier", 1)}


class Recortes(Dataset):
    """Imagen de cada recorte llevada a LADO x LADO: estirada, o puesta a la izquierda de una ventana de 300 s en negro."""
    def __init__(self, dir_img, archivos, y, tamano, mean, std):
        self.dir_img, self.archivos, self.y, self.tamano = dir_img, list(archivos), list(y), tamano
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.archivos)

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.dir_img, self.archivos[i])).convert("RGB")
        if self.tamano == "rellenar" and im.width < ANCHO_VENTANA:
            fondo = Image.new("RGB", (ANCHO_VENTANA, im.height))
            fondo.paste(im, (0, 0))
            im = fondo
        im = im.resize((LADO, LADO), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
        return (x - self.mean) / self.std, self.y[i]


def nombre_resultado(modelo, fold, tamano="estirar", conjunto="", pesos_clase=False, ajuste_completo=False, lr=None,
                     duracion_minima=None):
    """Nombre del JSON (y de los pesos) de una corrida; fold 0 es el entrenamiento final evaluado en test."""
    return (f"{modelo}_{'test' if fold == 0 else f'fold{fold}'}_{tamano}" + (f"_{conjunto}" if conjunto else "")
            + ("_pesos" if pesos_clase else "") + ("_completo" if ajuste_completo else "")
            + (f"_lr{lr:g}" if lr is not None else "") + (f"_min{duracion_minima:g}s" if duracion_minima is not None else ""))


def conjunto_de(datos):
    """El conjunto de imagenes sale del nombre de la carpeta: clasificacion_colab_fijo60s -> fijo60s."""
    base = os.path.basename(os.path.normpath(datos))
    return base[len("clasificacion_colab"):].strip("_") if base.startswith("clasificacion_colab") else ""


def construir(modelo, n_clases, preentrenado=True):
    """Red de torchvision con sus pesos por defecto y la ultima capa cambiada por una de n_clases salidas."""
    pesos = torchvision.models.get_model_weights(modelo).DEFAULT if preentrenado else None
    red = torchvision.models.get_model(modelo, weights=pesos)
    attr, i = CABEZAS[modelo]
    vieja = getattr(red, attr) if i is None else getattr(red, attr)[i]
    nueva = nn.Linear(vieja.in_features, n_clases)
    if i is None:
        setattr(red, attr, nueva)
    else:
        getattr(red, attr)[i] = nueva
    t = pesos.transforms() if pesos is not None else None
    # media y desviacion con que se normalizaron las imagenes al preentrenar la red
    mean = list(t.mean) if t is not None else [0.485, 0.456, 0.406]
    std = list(t.std) if t is not None else [0.229, 0.224, 0.225]
    return red, nueva, mean, std, (pesos.name if pesos is not None else None)


def modo_entrenamiento(red, cabeza, congelada):
    """Con la red congelada solo la capa nueva entrena; el resto queda en modo evaluacion (BatchNorm y Dropout fijos)."""
    red.train()
    if congelada:
        for m in red.modules():
            m.eval()
        cabeza.train()


def paso(red, perdida_suma, peso, grupo, opt, disp):
    """Un paso del optimizador con un lote dividido en partes: suma los gradientes de las partes y divide por el peso
    total del lote, igual que la perdida media (ponderada) de CrossEntropyLoss sobre el lote completo."""
    opt.zero_grad()
    total = sum(float(peso[y].sum()) if peso is not None else len(y) for _, y in grupo)
    suma = 0.0
    for x, y in grupo:
        l = perdida_suma(red(x.to(disp)), y.to(disp)) / total
        l.backward()
        suma += l.item()
    opt.step()
    return suma


def metricas(c):
    """Desde la matriz de confusion (filas: clase real; columnas: prediccion)."""
    out = {}
    for i, k in enumerate(CLASES):
        tp, real, pred = c[i, i], c[i].sum(), c[:, i].sum()
        rec = tp / real if real else float("nan")
        pre = tp / pred if pred else float("nan")
        f1 = 2 * pre * rec / (pre + rec) if pred and real and (pre + rec) else 0.0
        out[k] = dict(n=int(real), sensibilidad=float(rec), precision=float(pre), f1=float(f1))
    out["f1_macro_volcanicas"] = float(np.mean([out[k]["f1"] for k in VOLCANICAS]))
    out["exactitud"] = float(np.trace(c) / c.sum())
    return out


def evaluar(red, carga, disp):
    red.eval()
    c = np.zeros((len(CLASES), len(CLASES)), dtype=np.int64)
    with torch.no_grad():
        for x, y in carga:
            p = red(x.to(disp)).argmax(1).cpu().numpy()
            np.add.at(c, (y.numpy(), p), 1)
    return c


def semilla(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True, help="carpeta con indice.csv e imagenes/")
    ap.add_argument("--modelo", required=True, choices=sorted(CABEZAS))
    ap.add_argument("--fold", type=int, required=True, help="1 a 5: fold de validacion; 0: todo desarrollo y test")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--tamano", default="estirar", choices=["estirar", "rellenar"])
    ap.add_argument("--pesos-clase", action="store_true", help="pesa la perdida por el inverso de la frecuencia de cada clase")
    ap.add_argument("--ajuste-completo", action="store_true", help="entrena toda la red, no solo la capa nueva")
    ap.add_argument("--lr", type=float, default=None, help="tasa de aprendizaje de AdamW; sin el argumento, su valor por defecto")
    ap.add_argument("--duracion-minima", type=float, default=None,
                    help="deja fuera los recortes mas cortos que esto (s), en entrenamiento y validacion")
    ap.add_argument("--epocas", type=int, default=10)
    ap.add_argument("--lote", type=int, default=64)
    ap.add_argument("--acumular", type=int, default=1,
                    help="pasa cada lote en K partes y suma sus gradientes: mismo lote efectivo con menos memoria")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--trabajadores", type=int, default=2)
    ap.add_argument("--limite", type=int, default=None, help="solo N recortes por particion, para probar")
    ap.add_argument("--sin-preentrenar", action="store_true", help="pesos al azar, solo para probar sin descargar")
    ap.add_argument("--guardar-pesos", action="store_true")
    args = ap.parse_args()
    semilla(args.semilla)
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = pd.read_csv(os.path.join(args.datos, "indice.csv"))
    idx = idx[idx.etiqueta.isin(MAPA)].copy()
    idx["y"] = idx.etiqueta.map(MAPA).map({k: i for i, k in enumerate(CLASES)})
    es_test = args.fold == 0
    cortos = {}
    if args.duracion_minima is not None:
        evaluados = idx[idx.split == "test"] if es_test else idx[idx.fold == args.fold]
        c = evaluados[evaluados.dur_s < args.duracion_minima]
        cortos = {k: int((c.y == i).sum()) for i, k in enumerate(CLASES)}
        idx = idx[idx.dur_s >= args.duracion_minima]
    ent = idx[(idx.split == "desarrollo") & (idx.fold != args.fold)]
    val = idx[idx.split == "test"] if es_test else idx[idx.fold == args.fold]
    if args.limite:
        ent, val = ent.sample(min(args.limite, len(ent)), random_state=0), val.sample(min(args.limite, len(val)), random_state=0)
    if set(ent.ventana) & set(val.ventana):
        raise SystemExit("DETENIDO: hay ventanas en entrenamiento y validacion a la vez")

    red, cabeza, mean, std, nombre_pesos = construir(args.modelo, len(CLASES), not args.sin_preentrenar)
    congelada = not args.ajuste_completo
    if congelada:
        for p in red.parameters():
            p.requires_grad = False
        for p in cabeza.parameters():
            p.requires_grad = True
    red.to(disp)
    dir_img = os.path.join(args.datos, "imagenes")
    if args.lote % args.acumular:
        raise SystemExit("DETENIDO: --lote tiene que ser multiplo de --acumular")
    kw = dict(batch_size=args.lote // args.acumular, num_workers=args.trabajadores, pin_memory=disp.type == "cuda")
    gen = torch.Generator().manual_seed(args.semilla)
    c_ent = DataLoader(Recortes(dir_img, ent.archivo, ent.y, args.tamano, mean, std), shuffle=True, generator=gen, **kw)
    c_val = DataLoader(Recortes(dir_img, val.archivo, val.y, args.tamano, mean, std), shuffle=False, **kw)

    n_por_clase = np.bincount(ent.y, minlength=len(CLASES))
    peso = torch.tensor(n_por_clase.sum() / (len(CLASES) * n_por_clase), dtype=torch.float32) if args.pesos_clase else None
    perdida_suma = nn.CrossEntropyLoss(weight=peso.to(disp) if peso is not None else None, reduction="sum")
    # AdamW con sus valores por defecto de PyTorch (lr 0,001, weight_decay 0,01); --lr cambia solo la tasa
    entrenables = [p for p in red.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(entrenables) if args.lr is None else torch.optim.AdamW(entrenables, lr=args.lr)

    conjunto = conjunto_de(args.datos)
    nombre = nombre_resultado(args.modelo, args.fold, args.tamano, conjunto, args.pesos_clase, args.ajuste_completo,
                              args.lr, args.duracion_minima)
    cfg = dict(vars(args), conjunto=conjunto or "recorte", evaluacion="test" if es_test else "validacion", lr_usada=opt.defaults["lr"], clases=CLASES, mapa=MAPA,
               lado=LADO, pesos_preentrenados=nombre_pesos, mean=mean, std=std,
               congelada=congelada, cortos_fuera_de_validacion=cortos, n_entrenamiento={k: int(n) for k, n in zip(CLASES, n_por_clase)},
               n_validacion={k: int(n) for k, n in zip(CLASES, np.bincount(val.y, minlength=len(CLASES)))},
               versiones=dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__),
               dispositivo=torch.cuda.get_device_name(0) if disp.type == "cuda" else "cpu")
    print(f"{nombre}: entrenamiento {len(ent)}, validacion {len(val)} (ventanas {ent.ventana.nunique()} y "
          f"{val.ventana.nunique()}); {cfg['dispositivo']}; pesos {nombre_pesos}", flush=True)
    print(f"  por clase en entrenamiento: {cfg['n_entrenamiento']}", flush=True)
    if cortos:
        print(f"  recortes de validacion que quedan fuera por cortos: {cortos}", flush=True)

    historia, mejor = [], None
    for ep in range(1, args.epocas + 1):
        t0 = time.time()
        modo_entrenamiento(red, cabeza, congelada)
        suma, n, grupo = 0.0, 0, []
        for x, y in c_ent:
            grupo.append((x, y))
            if len(grupo) == args.acumular:
                suma += paso(red, perdida_suma, peso, grupo, opt, disp) * sum(len(g[1]) for g in grupo)
                n += sum(len(g[1]) for g in grupo)
                grupo = []
        if grupo:
            suma += paso(red, perdida_suma, peso, grupo, opt, disp) * sum(len(g[1]) for g in grupo)
            n += sum(len(g[1]) for g in grupo)
        c = evaluar(red, c_val, disp)
        m = metricas(c)
        mem = torch.cuda.max_memory_allocated() / 1e9 if disp.type == "cuda" else None
        historia.append(dict(epoca=ep, perdida=suma / n, segundos=time.time() - t0, memoria_gb=mem, matriz=c.tolist(), **m))
        print(f"  epoca {ep}: perdida {suma / n:.3f}; {cfg['evaluacion']}: F1 macro volcanicas {m['f1_macro_volcanicas']:.3f}; "
              + ", ".join(f"{k} sens {m[k]['sensibilidad']:.3f}" for k in CLASES) + f"; {time.time() - t0:.0f} s"
              + (f"; memoria GPU {mem:.1f} GB" if mem is not None else ""), flush=True)
        # en test no se elige epoca: se informa la ultima, fijada antes con los folds
        if es_test or mejor is None or m["f1_macro_volcanicas"] > mejor["f1_macro_volcanicas"]:
            mejor = historia[-1]
            if args.guardar_pesos and (not es_test or ep == args.epocas):
                os.makedirs(args.salida, exist_ok=True)
                torch.save(red.state_dict(), os.path.join(args.salida, f"{nombre}.pt"))

    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, f"{nombre}.json"), "w") as f:
        json.dump(dict(configuracion=cfg, historia=historia, mejor_epoca=mejor["epoca"]), f, indent=1)
    print(f"{'Ultima' if es_test else 'Mejor'} epoca {mejor['epoca']} (F1 macro volcanicas {mejor['f1_macro_volcanicas']:.3f}); "
          f"matriz de confusion (filas real, columnas prediccion; {', '.join(CLASES)}):")
    for k, fila in zip(CLASES, mejor["matriz"]):
        print(f"  {k:>8} {fila}")
    print(f"Resultados en {os.path.join(args.salida, nombre + '.json')}")


if __name__ == "__main__":
    main()
