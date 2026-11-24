#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ajuste por transferencia de una CNN preentrenada en ImageNet (VGG16, ResNet50 o EfficientNet-B0 de
torchvision) para clasificar los recortes en VT, LP, TR y rechazo, con un fold de la validacion
cruzada de Jose. Con --paciencia se detiene cuando el F1 macro de VT, LP y TR en validacion no mejora
en esa cantidad de epocas; con --evaluar-test, el mejor modelo del fold se evalua en las ventanas de
test. Con --fold 0 entrena con todo desarrollo y evalua en test la ultima epoca. Las metricas son sobre
los eventos de Jose (VT, LP, TR, AV, IC); los recortes sin caja se informan aparte. Guarda un JSON con
la configuracion, las versiones y las metricas de cada epoca, y un CSV con la prediccion de cada recorte.

Uso: python entrenar_cnn.py --datos <carpeta> --modelo vgg16 --fold 1 --salida <carpeta>
     [--tamano estirar|rellenar] [--pesos-clase] [--ajuste-completo] [--duracion-minima 5]
     [--sin-falsas-alarmas] [--optimizador adamw|sgd] [--lr 0.0001] [--momentum 0.9]
     [--weight-decay 0.0001] [--dropout 0.3] [--perdida entropia|focal] [--lote 32]
     [--epocas 50] [--paciencia 5] [--evaluar-test] [--acumular 2] [--guardar-pesos]
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
from torchvision.ops import sigmoid_focal_loss

CLASES = ["VT", "LP", "TR", "rechazo"]
VOLCANICAS = ["VT", "LP", "TR"]
# AV, IC y los recortes que no tocan ninguna caja van al rechazo (los "sin caja" se pueden dejar fuera del
# entrenamiento con --sin-falsas-alarmas); los que tocan una caja con IoU menor que 0,3 ("fuera") no se usan
MAPA = {"VT": "VT", "LP": "LP", "TR": "TR", "AV": "rechazo", "IC": "rechazo", "sin caja": "rechazo"}
# lado de entrada con que se preentrenaron las tres redes (crop_size 224 de sus transforms); no se usa el
# recorte central de esas transforms porque cortaria el inicio y el fin del evento
LADO = 224
# columnas del espectrograma de una ventana completa de 300 s (465 en las imagenes de Jose)
ANCHO_VENTANA = 465
CABEZAS = {"vgg16": ("classifier", 6), "resnet50": ("fc", None), "efficientnet_b0": ("classifier", 1)}
# lote con que se nombraban las corridas sin --lote
LOTE_BASE = 64


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
                     duracion_minima=None, sin_falsas=False, optimizador="adamw", momentum=None, weight_decay=None,
                     dropout=None, lote=LOTE_BASE, perdida="entropia", paciencia=None):
    """Nombre del JSON (y de los pesos) de una corrida; fold 0 es el entrenamiento final evaluado en test. Las opciones
    nuevas solo se agregan cuando no tienen su valor base, para que los nombres de las corridas anteriores no cambien."""
    return (f"{modelo}_{'test' if fold == 0 else f'fold{fold}'}_{tamano}" + (f"_{conjunto}" if conjunto else "")
            + ("_pesos" if pesos_clase else "") + ("_completo" if ajuste_completo else "")
            + (f"_lr{lr:g}" if lr is not None else "") + (f"_min{duracion_minima:g}s" if duracion_minima is not None else "")
            + ("_sin_falsas" if sin_falsas else "") + (f"_{optimizador}" if optimizador != "adamw" else "")
            + (f"_mom{momentum:g}" if momentum is not None else "") + (f"_wd{weight_decay:g}" if weight_decay is not None else "")
            + (f"_do{dropout:g}" if dropout is not None else "") + (f"_lote{lote}" if lote != LOTE_BASE else "")
            + (f"_{perdida}" if perdida != "entropia" else "") + (f"_pac{paciencia}" if paciencia is not None else ""))


def conjunto_de(datos):
    """El conjunto de imagenes sale del nombre de la carpeta: clasificacion_colab_fijo60s -> fijo60s."""
    base = os.path.basename(os.path.normpath(datos))
    return base[len("clasificacion_colab"):].strip("_") if base.startswith("clasificacion_colab") else ""


def construir(modelo, n_clases, preentrenado=True, dropout=None):
    """Red de torchvision con sus pesos por defecto y la ultima capa cambiada por una de n_clases salidas. Con dropout,
    VGG16 y EfficientNet-B0 lo reciben en su parametro 'dropout'; ResNet50 no lo tiene y se le pone una capa Dropout
    antes de la capa nueva."""
    pesos = torchvision.models.get_model_weights(modelo).DEFAULT if preentrenado else None
    propio = dropout is not None and modelo != "resnet50"
    red = torchvision.models.get_model(modelo, weights=pesos, **({"dropout": dropout} if propio else {}))
    attr, i = CABEZAS[modelo]
    vieja = getattr(red, attr) if i is None else getattr(red, attr)[i]
    nueva = nn.Linear(vieja.in_features, n_clases)
    cabeza = nn.Sequential(nn.Dropout(dropout), nueva) if dropout is not None and not propio else nueva
    if i is None:
        setattr(red, attr, cabeza)
    else:
        getattr(red, attr)[i] = cabeza
    t = pesos.transforms() if pesos is not None else None
    # media y desviacion con que se normalizaron las imagenes al preentrenar la red
    mean = list(t.mean) if t is not None else [0.485, 0.456, 0.406]
    std = list(t.std) if t is not None else [0.229, 0.224, 0.225]
    return red, cabeza, mean, std, (pesos.name if pesos is not None else None)


def dropouts(red):
    """Probabilidad de cada capa Dropout de la red, en orden."""
    return [m.p for m in red.modules() if isinstance(m, nn.Dropout)]


def modo_entrenamiento(red, cabeza, congelada):
    """Con la red congelada solo la capa nueva entrena; el resto queda en modo evaluacion (BatchNorm y Dropout fijos)."""
    red.train()
    if congelada:
        for m in red.modules():
            m.eval()
        cabeza.train()


def optimizador(nombre, parametros, lr=None, momentum=None, weight_decay=None):
    """AdamW o SGD de PyTorch; lo que no se da queda con su valor por defecto."""
    kw = {k: v for k, v in dict(lr=lr, weight_decay=weight_decay).items() if v is not None}
    if nombre == "sgd":
        return torch.optim.SGD(parametros, **kw, **({"momentum": momentum} if momentum is not None else {}))
    if momentum is not None:
        raise SystemExit("DETENIDO: --momentum es un parametro de SGD, no de AdamW")
    return torch.optim.AdamW(parametros, **kw)


def perdida_y_cuenta(perdida, peso, n_clases):
    """Perdida sumada sobre un trozo del lote y lo que se divide para tener la media del lote completo.
    Entropia cruzada: suma de los pesos de clase de los ejemplos (o su numero), como su reduction='mean'.
    Focal de torchvision: cada clase es un si o no con la etiqueta en one-hot; su 'mean' promedia ejemplos x clases."""
    if perdida == "focal":
        def suma(logits, y):
            return sigmoid_focal_loss(logits, nn.functional.one_hot(y, n_clases).float(), reduction="sum")
        return suma, lambda y: len(y) * n_clases
    ce = nn.CrossEntropyLoss(weight=peso, reduction="sum")
    return ce, (lambda y: float(peso.cpu()[y].sum())) if peso is not None else (lambda y: len(y))


def paso(red, perdida_suma, cuenta, grupo, opt, disp):
    """Un paso del optimizador con un lote dividido en partes: suma los gradientes de las partes y divide por la cuenta
    del lote completo, igual que la perdida media del lote entero."""
    opt.zero_grad()
    total = sum(cuenta(y) for _, y in grupo)
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


def detener(historia, paciencia):
    """True si el F1 macro de VT, LP y TR lleva 'paciencia' epocas sin superar su mejor valor."""
    if paciencia is None:
        return False
    f1 = [h["f1_macro_volcanicas"] for h in historia]
    return len(f1) - (int(np.argmax(f1)) + 1) >= paciencia


def predecir(red, carga, disp, prob):
    """Probabilidad de cada clase para cada recorte, en el orden de la carga."""
    red.eval()
    with torch.no_grad():
        return np.concatenate([prob(red(x.to(disp))).cpu().numpy() for x, _ in carga])


def matriz(y, pred):
    c = np.zeros((len(CLASES), len(CLASES)), dtype=np.int64)
    np.add.at(c, (np.asarray(y), np.asarray(pred)), 1)
    return c


def evaluar(prob, sub):
    """Matriz y metricas sobre los eventos de Jose; los sin caja solo se cuentan por la clase que reciben."""
    evento = (sub.etiqueta != "sin caja").to_numpy()
    y, pred = sub.y.to_numpy(), prob.argmax(1)
    c = matriz(y[evento], pred[evento])
    sin_caja = {k: int(n) for k, n in zip(CLASES, np.bincount(pred[~evento], minlength=len(CLASES)))}
    return dict(matriz=c.tolist(), sin_caja_predichos=sin_caja, **metricas(c))


def tabla_predicciones(prob, sub):
    p = pd.DataFrame(prob, columns=[f"p_{k}" for k in CLASES])
    p.insert(0, "prediccion", [CLASES[i] for i in prob.argmax(1)])
    p.insert(0, "clase", [CLASES[i] for i in sub.y.to_numpy()])
    for col in ("etiqueta", "ventana", "archivo"):
        p.insert(0, col, sub[col].to_numpy())
    return p


def cortos_de(sub, minimo):
    """Eventos de Jose por clase y sin caja que quedan fuera por durar menos que el minimo."""
    c = sub[sub.dur_s < minimo]
    return {k: int(((c.y == i) & (c.etiqueta != "sin caja")).sum()) for i, k in enumerate(CLASES)}, \
        int((c.etiqueta == "sin caja").sum())


def semilla(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def argumentos(lista=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", required=True, help="carpeta con indice.csv e imagenes/")
    ap.add_argument("--modelo", required=True, choices=sorted(CABEZAS))
    ap.add_argument("--fold", type=int, required=True, help="1 a 5: fold de validacion; 0: todo desarrollo y test")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--tamano", default="estirar", choices=["estirar", "rellenar"])
    ap.add_argument("--pesos-clase", action="store_true", help="pesa la entropia cruzada por el inverso de la frecuencia de cada clase")
    ap.add_argument("--ajuste-completo", action="store_true", help="entrena toda la red, no solo la capa nueva")
    ap.add_argument("--optimizador", default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--lr", type=float, default=None, help="tasa de aprendizaje; sin el argumento, la del optimizador")
    ap.add_argument("--momentum", type=float, default=None, help="momentum de SGD; sin el argumento, el de PyTorch (0)")
    ap.add_argument("--weight-decay", type=float, default=None, help="sin el argumento, el del optimizador")
    ap.add_argument("--dropout", type=float, default=None, help="sin el argumento, el de cada red (ResNet50 no tiene)")
    ap.add_argument("--perdida", default="entropia", choices=["entropia", "focal"])
    ap.add_argument("--duracion-minima", type=float, default=None,
                    help="deja fuera los recortes mas cortos que esto (s), en entrenamiento, validacion y test")
    ap.add_argument("--epocas", type=int, default=10, help="maximo de epocas")
    ap.add_argument("--paciencia", type=int, default=None,
                    help="se detiene si el F1 macro de VT, LP y TR no mejora en esta cantidad de epocas")
    ap.add_argument("--evaluar-test", action="store_true", help="evalua en test el mejor modelo del fold")
    ap.add_argument("--lote", type=int, default=LOTE_BASE)
    ap.add_argument("--acumular", type=int, default=1,
                    help="pasa cada lote en K partes y suma sus gradientes: mismo lote efectivo con menos memoria")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--trabajadores", type=int, default=2)
    ap.add_argument("--limite", type=int, default=None, help="solo N recortes por particion, para probar")
    ap.add_argument("--sin-preentrenar", action="store_true", help="pesos al azar, solo para probar sin descargar")
    ap.add_argument("--sin-falsas-alarmas", action="store_true",
                    help="entrena sin los recortes sin caja (no tocan ningun evento del catalogo)")
    ap.add_argument("--guardar-pesos", action="store_true")
    return ap.parse_args(lista)


def entrenar(args):
    """Entrena con los argumentos dados, guarda JSON y predicciones y devuelve el resultado de la mejor epoca."""
    es_test = args.fold == 0
    if es_test and (args.paciencia is not None or args.evaluar_test):
        raise SystemExit("DETENIDO: con --fold 0 no hay validacion para --paciencia ni un mejor modelo para --evaluar-test")
    if args.perdida == "focal" and args.pesos_clase:
        raise SystemExit("DETENIDO: --pesos-clase es de la entropia cruzada; la focal de torchvision no recibe pesos por clase")
    if args.lote % args.acumular:
        raise SystemExit("DETENIDO: --lote tiene que ser multiplo de --acumular")
    semilla(args.semilla)
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = pd.read_csv(os.path.join(args.datos, "indice.csv"))
    idx = idx[idx.etiqueta.isin(MAPA)].copy()
    idx["y"] = idx.etiqueta.map(MAPA).map({k: i for i, k in enumerate(CLASES)})
    cortos, cortos_sin_caja, cortos_test = {}, 0, None
    if args.duracion_minima is not None:
        cortos, cortos_sin_caja = cortos_de(idx[idx.split == "test"] if es_test else idx[idx.fold == args.fold],
                                            args.duracion_minima)
        if args.evaluar_test:
            cortos_test = cortos_de(idx[idx.split == "test"], args.duracion_minima)
        idx = idx[idx.dur_s >= args.duracion_minima]
    ent = idx[(idx.split == "desarrollo") & (idx.fold != args.fold)]
    if args.sin_falsas_alarmas:
        ent = ent[ent.etiqueta != "sin caja"]
    val = idx[idx.split == "test"] if es_test else idx[idx.fold == args.fold]
    prueba = idx[idx.split == "test"] if args.evaluar_test else None
    if args.limite:
        ent, val = ent.sample(min(args.limite, len(ent)), random_state=0), val.sample(min(args.limite, len(val)), random_state=0)
        if prueba is not None:
            prueba = prueba.sample(min(args.limite, len(prueba)), random_state=0)
    if set(ent.ventana) & set(val.ventana) or (prueba is not None and set(prueba.ventana) & (set(ent.ventana) | set(val.ventana))):
        raise SystemExit("DETENIDO: hay ventanas compartidas entre entrenamiento, validacion y test")

    red, cabeza, mean, std, nombre_pesos = construir(args.modelo, len(CLASES), not args.sin_preentrenar, args.dropout)
    congelada = not args.ajuste_completo
    if congelada:
        for p in red.parameters():
            p.requires_grad = False
        for p in cabeza.parameters():
            p.requires_grad = True
    red.to(disp)
    dir_img = os.path.join(args.datos, "imagenes")
    kw = dict(batch_size=args.lote // args.acumular, num_workers=args.trabajadores, pin_memory=disp.type == "cuda")
    gen = torch.Generator().manual_seed(args.semilla)
    c_ent = DataLoader(Recortes(dir_img, ent.archivo, ent.y, args.tamano, mean, std), shuffle=True, generator=gen, **kw)
    c_val = DataLoader(Recortes(dir_img, val.archivo, val.y, args.tamano, mean, std), shuffle=False, **kw)

    n_por_clase = np.bincount(ent.y, minlength=len(CLASES))
    peso = torch.tensor(n_por_clase.sum() / (len(CLASES) * n_por_clase), dtype=torch.float32) if args.pesos_clase else None
    perdida_suma, cuenta = perdida_y_cuenta(args.perdida, peso.to(disp) if peso is not None else None, len(CLASES))
    # la focal trata cada clase como un si o no: su probabilidad es la sigmoide de cada salida, no el softmax
    prob = torch.sigmoid if args.perdida == "focal" else (lambda z: torch.softmax(z, 1))
    opt = optimizador(args.optimizador, [p for p in red.parameters() if p.requires_grad], args.lr, args.momentum,
                      args.weight_decay)

    conjunto = conjunto_de(args.datos)
    nombre = nombre_resultado(args.modelo, args.fold, args.tamano, conjunto, args.pesos_clase, args.ajuste_completo,
                              args.lr, args.duracion_minima, args.sin_falsas_alarmas, args.optimizador, args.momentum,
                              args.weight_decay, args.dropout, args.lote, args.perdida, args.paciencia)
    evento = (val.etiqueta != "sin caja").to_numpy()
    yv = val.y.to_numpy()
    cfg = dict(vars(args), conjunto=conjunto or "recorte", evaluacion="test" if es_test else "validacion",
               lr_usada=opt.defaults["lr"], weight_decay_usado=opt.defaults["weight_decay"],
               momentum_usado=opt.defaults.get("momentum"), dropout_red=dropouts(red),
               probabilidad="sigmoide" if args.perdida == "focal" else "softmax", clases=CLASES, mapa=MAPA,
               lado=LADO, pesos_preentrenados=nombre_pesos, mean=mean, std=std,
               congelada=congelada, cortos_fuera_de_validacion=cortos, cortos_sin_caja=cortos_sin_caja,
               n_entrenamiento={k: int(n) for k, n in zip(CLASES, n_por_clase)},
               n_validacion={k: int(n) for k, n in zip(CLASES, np.bincount(yv[evento], minlength=len(CLASES)))},
               n_sin_caja_validacion=int((~evento).sum()),
               versiones=dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__),
               dispositivo=torch.cuda.get_device_name(0) if disp.type == "cuda" else "cpu")
    print(f"{nombre}: entrenamiento {len(ent)}, validacion {len(val)} (ventanas {ent.ventana.nunique()} y "
          f"{val.ventana.nunique()}); {cfg['dispositivo']}; pesos {nombre_pesos}", flush=True)
    print(f"  {args.optimizador} lr {cfg['lr_usada']:g}, weight_decay {cfg['weight_decay_usado']:g}"
          + (f", momentum {cfg['momentum_usado']:g}" if args.optimizador == "sgd" else "")
          + f"; dropout {cfg['dropout_red']}; perdida {args.perdida}; lote {args.lote}", flush=True)
    print(f"  por clase en entrenamiento: {cfg['n_entrenamiento']}", flush=True)
    print(f"  eventos de Jose en {cfg['evaluacion']}: {cfg['n_validacion']}; sin caja aparte: {cfg['n_sin_caja_validacion']}",
          flush=True)
    if cortos:
        print(f"  recortes de {cfg['evaluacion']} que quedan fuera por cortos: {cortos}; sin caja: {cortos_sin_caja}", flush=True)

    historia, mejor, mejor_estado, prob_mejor = [], None, None, None
    for ep in range(1, args.epocas + 1):
        t0 = time.time()
        modo_entrenamiento(red, cabeza, congelada)
        suma, n, grupo = 0.0, 0, []
        for x, y in c_ent:
            grupo.append((x, y))
            if len(grupo) == args.acumular:
                suma += paso(red, perdida_suma, cuenta, grupo, opt, disp) * sum(len(g[1]) for g in grupo)
                n += sum(len(g[1]) for g in grupo)
                grupo = []
        if grupo:
            suma += paso(red, perdida_suma, cuenta, grupo, opt, disp) * sum(len(g[1]) for g in grupo)
            n += sum(len(g[1]) for g in grupo)
        p = predecir(red, c_val, disp, prob)
        m = evaluar(p, val)
        mem = torch.cuda.max_memory_allocated() / 1e9 if disp.type == "cuda" else None
        historia.append(dict(epoca=ep, perdida=suma / n, segundos=time.time() - t0, memoria_gb=mem, **m))
        sc = m["sin_caja_predichos"]
        print(f"  epoca {ep}: perdida {suma / n:.3f}; {cfg['evaluacion']}: F1 macro volcanicas {m['f1_macro_volcanicas']:.3f}; "
              + ", ".join(f"{k} sens {m[k]['sensibilidad']:.3f}" for k in CLASES) + f"; {time.time() - t0:.0f} s"
              + (f"; memoria GPU {mem:.1f} GB" if mem is not None else "")
              + (f"; sin caja como volcanico {sum(sc[k] for k in VOLCANICAS)} de {int((~evento).sum())}"
                 if (~evento).any() else ""), flush=True)
        # en test no se elige epoca: se informa la ultima
        if es_test or mejor is None or m["f1_macro_volcanicas"] > mejor["f1_macro_volcanicas"]:
            mejor, prob_mejor = historia[-1], p
            mejor_estado = {k: v.detach().to("cpu", copy=True) for k, v in red.state_dict().items()}
        if detener(historia, args.paciencia):
            print(f"  sin mejora en {args.paciencia} epocas: se detiene en la epoca {ep}", flush=True)
            break

    os.makedirs(args.salida, exist_ok=True)
    if args.guardar_pesos:
        torch.save(mejor_estado, os.path.join(args.salida, f"{nombre}.pt"))
    res = dict(configuracion=cfg, historia=historia, mejor_epoca=mejor["epoca"], epocas_corridas=len(historia))
    if args.evaluar_test:
        red.load_state_dict(mejor_estado)
        c_prueba = DataLoader(Recortes(dir_img, prueba.archivo, prueba.y, args.tamano, mean, std), shuffle=False, **kw)
        pt = predecir(red, c_prueba, disp, prob)
        mt = evaluar(pt, prueba)
        ev_t = (prueba.etiqueta != "sin caja").to_numpy()
        res["test"] = dict(epoca=mejor["epoca"], n={k: int(n) for k, n in zip(CLASES, np.bincount(prueba.y.to_numpy()[ev_t],
                                                                                                 minlength=len(CLASES)))},
                           n_sin_caja=int((~ev_t).sum()), cortos=cortos_test[0] if cortos_test else {},
                           cortos_sin_caja=cortos_test[1] if cortos_test else 0, **mt)
        tabla_predicciones(pt, prueba).to_csv(os.path.join(args.salida, f"{nombre}_test_predicciones.csv"), index=False)
    with open(os.path.join(args.salida, f"{nombre}.json"), "w") as f:
        json.dump(res, f, indent=1)
    tabla_predicciones(prob_mejor, val).to_csv(os.path.join(args.salida, f"{nombre}_predicciones.csv"), index=False)
    print(f"{'Ultima' if es_test else 'Mejor'} epoca {mejor['epoca']} de {len(historia)} (F1 macro volcanicas "
          f"{mejor['f1_macro_volcanicas']:.3f}); matriz de confusion (filas real, columnas prediccion; {', '.join(CLASES)}):")
    for k, fila in zip(CLASES, mejor["matriz"]):
        print(f"  {k:>8} {fila}")
    if args.evaluar_test:
        t = res["test"]
        print(f"Test con el modelo de la epoca {mejor['epoca']}: F1 macro volcanicas {t['f1_macro_volcanicas']:.3f}; "
              + ", ".join(f"{k} sens {t[k]['sensibilidad']:.3f}" for k in CLASES))
        for k, fila in zip(CLASES, t["matriz"]):
            print(f"  {k:>8} {fila}")
    print(f"Resultados en {os.path.join(args.salida, nombre + '.json')}")
    return res


def main():
    entrenar(argumentos())


if __name__ == "__main__":
    main()
