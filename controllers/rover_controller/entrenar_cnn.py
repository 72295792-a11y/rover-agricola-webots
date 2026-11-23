# -*- coding: utf-8 -*-
"""Entrenamiento de CultivoCNN (PyTorch) para clasificacion de estres hidrico."""
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from vision import CultivoCNN, CLASES


def muestra(cls, rng):
    br = rng.uniform(0.50, 1.25)
    if cls == 0:
        base = np.array([rng.uniform(18, 65), rng.uniform(115, 205), rng.uniform(22, 75)], np.float32)
    elif cls == 1:
        base = np.array([rng.uniform(165, 240), rng.uniform(130, 210), rng.uniform(15, 65)], np.float32)
    else:
        sub = int(rng.integers(0, 4))
        if sub == 0:
            base = np.array([rng.uniform(100, 195), rng.uniform(75, 165), rng.uniform(60, 145)], np.float32)
        elif sub == 1:
            g = rng.uniform(80, 175)
            base = np.array([g, g + rng.uniform(-8, 8), g + rng.uniform(-8, 8)], np.float32)
        elif sub == 2:
            base = np.array([rng.uniform(160, 235), rng.uniform(20, 65), rng.uniform(20, 65)], np.float32)
        else:
            base = np.array([rng.uniform(90, 165), rng.uniform(120, 195), rng.uniform(155, 235)], np.float32)
    gx = np.linspace(rng.uniform(0.75, 1.05), rng.uniform(0.75, 1.05), 32, dtype=np.float32)[None, :, None]
    gy = np.linspace(rng.uniform(0.85, 1.15), rng.uniform(0.70, 1.00), 32, dtype=np.float32)[:, None, None]
    img = base[None, None, :] * br * gx * gy + rng.normal(0, 8, (32, 32, 3)).astype(np.float32)
    return (np.clip(img, 0, 255) / 255.0).transpose(2, 0, 1)


def main():
    t0 = time.time()
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    X_tr, y_tr = [], []
    for c in range(3):
        for _ in range(500):
            X_tr.append(muestra(c, rng)); y_tr.append(c)
    X_va, y_va = [], []
    for c in range(3):
        for _ in range(100):
            X_va.append(muestra(c, rng)); y_va.append(c)

    loader = DataLoader(TensorDataset(torch.tensor(np.array(X_tr)), torch.tensor(y_tr, dtype=torch.long)), batch_size=64, shuffle=True)
    modelo = CultivoCNN(len(CLASES))
    opt = torch.optim.Adam(modelo.parameters(), lr=1.5e-3)
    crit = nn.CrossEntropyLoss()

    for _ in range(12):
        modelo.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(modelo(xb), yb); loss.backward(); opt.step()

    modelo.eval()
    with torch.no_grad():
        preds = modelo(torch.tensor(np.array(X_va))).argmax(dim=1).numpy()
    acc = float(np.mean(preds == np.array(y_va)))

    base = os.path.dirname(os.path.abspath(__file__))
    out_m = os.path.join(base, "modelo_cultivos_cnn.pth")
    torch.save(modelo.state_dict(), out_m)
    res_dir = os.path.abspath(os.path.join(base, "..", "..", "resultados"))
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, "metricas_cnn.json"), "w", encoding="utf-8") as f:
        json.dump({"modelo": "CultivoCNN", "clases": CLASES, "muestras_train": 1500, "muestras_val": 300, "epocas": 12, "val_accuracy": round(acc, 4), "tiempo_s": round(time.time() - t0, 2)}, f, indent=2)
    print(f"CultivoCNN entrenada OK | val_accuracy={acc*100:.2f}% | tiempo={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
