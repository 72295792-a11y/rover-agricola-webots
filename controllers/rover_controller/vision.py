# -*- coding: utf-8 -*-
"""Modulo 4: Vision Computacional (Unidad 2) - OpenCV + CNN PyTorch + Mapa de Calor."""
import csv
import json
import math
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

CLASES = ["planta_sana", "planta_estres_hidrico", "fondo_otro"]


class CultivoCNN(nn.Module):
    def __init__(self, num_clases=3):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.clasificador = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 2 * 2, 64), nn.ReLU(inplace=True), nn.Linear(64, num_clases)
        )

    def forward(self, x):
        return self.clasificador(self.extractor(x))


def _norm_ang(a):
    return math.atan2(math.sin(a), math.cos(a))


class ModuloVision:
    def __init__(self, resultados_dir=None, modelo_path=None):
        base_ctrl = os.path.dirname(os.path.abspath(__file__))
        root = os.path.abspath(os.path.join(base_ctrl, "..", ".."))
        self.resultados_dir = resultados_dir or os.path.join(root, "resultados")
        os.makedirs(self.resultados_dir, exist_ok=True)
        self.modelo_path = modelo_path or os.path.join(base_ctrl, "modelo_cultivos_cnn.pth")

        self.modelo = CultivoCNN(len(CLASES))
        self.modelo.load_state_dict(torch.load(self.modelo_path, map_location="cpu", weights_only=True))
        self.modelo.eval()

        self.detecciones_raw = []
        self.clusters = []
        self._capturas = 0
        self._kernel = np.ones((5, 5), np.uint8)

    def _estimar_distancia(self, alpha, lidar_ranges, bh, img_h, fov_h):
        n = len(lidar_ranges)
        if n > 0:
            fov_l = 2.0 * math.pi
            cand = []
            for i, d in enumerate(lidar_ranges):
                if math.isnan(d) or math.isinf(d) or d < 0.35 or d > 4.2:
                    continue
                ang = _norm_ang((fov_l / 2.0) - (i + 0.5) * (fov_l / n))
                if abs(_norm_ang(ang - alpha)) <= 0.08:
                    cand.append(d)
            if cand:
                return min(cand)
        fov_v = fov_h * (img_h / 320.0)
        focal = (img_h / 2.0) / max(1e-3, math.tan(fov_v / 2.0))
        return max(0.5, min(4.0, (0.40 * focal) / max(10.0, float(bh))))

    def _clasificar_cnn(self, roi_bgr):
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi32 = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
        ten = torch.from_numpy(roi32).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            probs = torch.softmax(self.modelo(ten), dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
        return CLASES[int(idx.item())], float(conf.item())

    def _registrar_cluster(self, clase, conf, px, py, t):
        mejor, mejor_d = None, 1.05
        for c in self.clusters:
            d = math.hypot(c["x"] - px, c["y"] - py)
            if d < mejor_d:
                mejor_d, mejor = d, c
        if mejor is None:
            nuevo = {
                "id": len(self.clusters) + 1, "x": px, "y": py,
                "votos": {"planta_sana": 0.0, "planta_estres_hidrico": 0.0},
                "conf_max": conf, "obs": 1, "ultimo_t": t,
            }
            nuevo["votos"][clase] += conf
            self.clusters.append(nuevo)
            return nuevo["id"], True
        n = mejor["obs"]
        mejor["x"] = (mejor["x"] * n + px) / (n + 1)
        mejor["y"] = (mejor["y"] * n + py) / (n + 1)
        mejor["votos"][clase] += conf
        mejor["conf_max"] = max(mejor["conf_max"], conf)
        mejor["obs"] = n + 1
        mejor["ultimo_t"] = t
        return mejor["id"], False

    def procesar_frame(self, img_bgr, pose_ekf, lidar_ranges, fov_h, t):
        if img_bgr is None:
            return []
        h, w = img_bgr.shape[:2]
        blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask_v = cv2.inRange(hsv, np.array([38, 60, 35]), np.array([92, 255, 255]))
        mask_a = cv2.inRange(hsv, np.array([14, 75, 50]), np.array([37, 255, 255]))
        mask = cv2.morphologyEx(cv2.bitwise_or(mask_v, mask_a), cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nuevas, vis = [], (img_bgr.copy() if self._capturas < 6 else None)
        x_ekf, y_ekf, th_ekf = pose_ekf

        for cnt in contornos:
            if cv2.contourArea(cnt) < 200:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx, cy = bx + 0.5 * bw, by + 0.5 * bh
            if cy < h * 0.20 or bw < 10 or bh < 10:
                continue
            clase, conf = self._clasificar_cnn(img_bgr[by:by + bh, bx:bx + bw])
            if clase == "fondo_otro" or conf < 0.72:
                continue
            alpha = (0.5 - (cx / float(w))) * fov_h
            dist = self._estimar_distancia(alpha, lidar_ranges, bh, h, fov_h)
            if dist > 4.2:
                continue
            px = x_ekf + 0.25 * math.cos(th_ekf) + dist * math.cos(th_ekf + alpha)
            py = y_ekf + 0.25 * math.sin(th_ekf) + dist * math.sin(th_ekf + alpha)
            if abs(px) > 4.8 or abs(py) > 4.8:
                continue

            cid, es_nueva = self._registrar_cluster(clase, conf, px, py, t)
            det = {
                "tiempo_s": round(t, 3), "cluster_id": cid, "clase": clase,
                "confianza": round(conf, 4), "pos_x": round(px, 3), "pos_y": round(py, 3),
                "distancia_m": round(dist, 3), "es_nueva": es_nueva,
            }
            self.detecciones_raw.append(det)
            if es_nueva:
                nuevas.append(det)
            if vis is not None:
                col = (0, 210, 0) if clase == "planta_sana" else (0, 215, 255)
                txt = f"{'SANA' if clase == 'planta_sana' else 'ESTRES'} {conf*100:.0f}% ({dist:.1f}m)"
                cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), col, 2)
                cv2.putText(vis, txt, (bx, max(14, by - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

        if vis is not None and len(nuevas) > 0:
            self._capturas += 1
            cv2.imwrite(os.path.join(self.resultados_dir, f"captura_vision_{self._capturas}.png"), vis)
        return nuevas

    def consolidar(self):
        out = []
        for c in self.clusters:
            vs, ve = c["votos"]["planta_sana"], c["votos"]["planta_estres_hidrico"]
            clase = "planta_estres_hidrico" if ve >= vs else "planta_sana"
            out.append({
                "id": c["id"], "clase": clase, "confianza_max": round(c["conf_max"], 4),
                "indice_estres": round(ve / max(1e-6, vs + ve), 3),
                "pos_x": round(c["x"], 3), "pos_y": round(c["y"], 3), "observaciones": c["obs"],
            })
        return out

    def exportar_resultados(self, trayectoria_ekf=None):
        csv_p = os.path.join(self.resultados_dir, "detecciones.csv")
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["tiempo_s", "cluster_id", "clase", "confianza", "pos_x", "pos_y", "distancia_m"])
            for d in self.detecciones_raw:
                w.writerow([d["tiempo_s"], d["cluster_id"], d["clase"], d["confianza"], d["pos_x"], d["pos_y"], d["distancia_m"]])

        plantas = self.consolidar()
        payload = {
            "observaciones_totales": len(self.detecciones_raw),
            "plantas_detectadas": len(plantas),
            "plantas_sanas": sum(1 for p in plantas if p["clase"] == "planta_sana"),
            "plantas_estres_hidrico": sum(1 for p in plantas if p["clase"] == "planta_estres_hidrico"),
            "plantas": plantas,
        }
        json_p = os.path.join(self.resultados_dir, "detecciones.json")
        with open(json_p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        heat_p = os.path.join(self.resultados_dir, "mapa_calor_estres.png")
        gx, gy = np.linspace(-5, 5, 120), np.linspace(-5, 5, 120)
        XX, YY = np.meshgrid(gx, gy)
        Z = np.zeros_like(XX)
        for p in plantas:
            if p["clase"] == "planta_estres_hidrico":
                Z += max(0.5, p["confianza_max"]) * np.exp(-((XX - p["pos_x"]) ** 2 + (YY - p["pos_y"]) ** 2) / (2 * 0.85 ** 2))

        fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=150)
        im = ax.imshow(Z, extent=[-5, 5, -5, 5], origin="lower", cmap="YlOrRd", alpha=0.85, vmin=0.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Indice de Estres Hidrico (CNN)")
        if trayectoria_ekf:
            ax.plot([p[0] for p in trayectoria_ekf], [p[1] for p in trayectoria_ekf], "b--", lw=1.5, label="Trayectoria EKF")
        sanas = [p for p in plantas if p["clase"] == "planta_sana"]
        estres = [p for p in plantas if p["clase"] == "planta_estres_hidrico"]
        if sanas:
            ax.scatter([p["pos_x"] for p in sanas], [p["pos_y"] for p in sanas], c="lime", ec="k", s=110, marker="s", label="Planta Sana (CNN)", zorder=5)
        if estres:
            ax.scatter([p["pos_x"] for p in estres], [p["pos_y"] for p in estres], c="red", ec="k", s=130, marker="^", label="Estres Hidrico (CNN)", zorder=6)
        ax.set(title="Mapa de Calor de Estres Hidrico - Meseta del Mantaro", xlabel="X [m]", ylabel="Y [m]", xlim=(-5, 5), ylim=(-5, 5))
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(heat_p)
        plt.close(fig)
        return csv_p, json_p, heat_p, payload
