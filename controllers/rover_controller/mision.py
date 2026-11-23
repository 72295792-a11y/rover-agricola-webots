# -*- coding: utf-8 -*-
"""Modulo 5: Mision Autonoma Agricola (Integra Percepcion, EKF, Planificacion y Vision CNN)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from percepcion import Percepcion
from localizacion import LocalizacionEKF
from planificacion import Planificador
from vision import ModuloVision

METAS_MISION = [
    (-3.5, -1.25),
    (3.5, -1.25),
    (3.5, 1.25),
    (-3.5, 1.25),
    (-3.5, 3.5),
    (3.5, 3.5),
]


class MisionAgricola:
    def __init__(self, robot):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.percepcion = Percepcion(robot, self.timestep)
        self.ekf = LocalizacionEKF(x0=-3.5, y0=-3.5, theta0=0.0)
        self.planificador = Planificador()
        self.vision = ModuloVision()
        self.tray_ekf = []
        self.tray_gps = []

    def _graficar_ekf_vs_gps(self):
        out_p = os.path.join(self.vision.resultados_dir, "trayectoria_ekf_vs_gps.png")
        fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=150)
        ax.plot([p[0] for p in self.tray_gps], [p[1] for p in self.tray_gps], "g-", lw=2.0, label="GPS Ground Truth")
        ax.plot([p[0] for p in self.tray_ekf], [p[1] for p in self.tray_ekf], "b--", lw=1.6, label="Localizacion EKF (Encoders+IMU)")
        ax.scatter([m[0] for m in METAS_MISION], [m[1] for m in METAS_MISION], c="orange", ec="k", s=70, label="Metas de Mision", zorder=5)
        ax.set(title="Comparativa de Trayectoria: EKF Propio vs. GPS Ground Truth", xlabel="X [m]", ylabel="Y [m]", xlim=(-5, 5), ylim=(-5, 5))
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="lower right", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(out_p)
        plt.close(fig)
        return out_p

    def ejecutar(self):
        if self.robot.step(self.timestep) == -1:
            return
        sens = self.percepcion.leer_sensores()
        self.ekf.actualizar_imu(sens["imu_yaw"])

        meta_idx = 0
        pose0 = self.ekf.obtener_estado()
        wps = self.planificador.planificar_a_estrella((pose0[0], pose0[1]), METAS_MISION[meta_idx])
        print("==============================================================")
        print("MISION AGRICOLA INICIADA | 5 Modulos Activos (EKF + A* + CNN)")
        print(f"--> Meta 1/{len(METAS_MISION)}: {METAS_MISION[meta_idx]} ({len(wps)} waypoints)")
        print("==============================================================")

        t0 = self.robot.getTime()
        ultimo_log = -1.0
        paso = 0

        while self.robot.step(self.timestep) != -1:
            paso += 1
            t = self.robot.getTime() - t0
            sens = self.percepcion.leer_sensores()

            # 1. Localizacion EKF (0% GPS para navegacion)
            self.ekf.predecir(sens["encoder_delta_izq"], sens["encoder_delta_der"])
            self.ekf.actualizar_imu(sens["imu_yaw"])
            pose = self.ekf.obtener_estado()
            err = self.ekf.calcular_error_gps((sens["gps_x"], sens["gps_y"]), t=t)

            if paso % 4 == 0:
                self.tray_ekf.append((pose[0], pose[1]))
                self.tray_gps.append((sens["gps_x"], sens["gps_y"]))

            # 2. Vision Computacional (OpenCV + CNN PyTorch a ~6 Hz)
            if paso % 5 == 0:
                img = self.percepcion.leer_imagen_bgr()
                nuevas = self.vision.procesar_frame(img, pose, sens["lidar_range"], sens["camera_fov"], t)
                for d in nuevas:
                    etq = "ALERTA ESTRES HIDRICO" if d["clase"] == "planta_estres_hidrico" else "PLANTA SANA"
                    print(f"[t={t:5.1f}s] [VISION CNN] #{d['cluster_id']} {etq} (conf={d['confianza']*100:.1f}%) en ({d['pos_x']:+.2f}, {d['pos_y']:+.2f})")

            # 3. Planificacion A* + Evitacion Dinamica LiDAR
            v, w, llego, min_f = self.planificador.calcular_comando(pose, sens["lidar_range"])
            if llego:
                print(f"[t={t:5.1f}s] Meta {meta_idx+1}/{len(METAS_MISION)} ALCANZADA {METAS_MISION[meta_idx]} | Err EKF={err:.3f}m")
                meta_idx += 1
                if meta_idx >= len(METAS_MISION):
                    self.percepcion.detener()
                    err_prom, err_max = self.ekf.resumen_errores()
                    self.ekf.cerrar()
                    csv_p, json_p, heat_p, res = self.vision.exportar_resultados(self.tray_ekf)
                    tray_p = self._graficar_ekf_vs_gps()
                    print("==============================================================")
                    print(f"MISION AGRICOLA COMPLETADA en {t:.1f} s")
                    print(f"Localizacion EKF -> Error promedio: {err_prom:.3f} m | Maximo: {err_max:.3f} m")
                    print(f"Vision CNN       -> Plantas detectadas: {res['plantas_detectadas']} (Sanas: {res['plantas_sanas']} | Estres hidrico: {res['plantas_estres_hidrico']})")
                    print(f"Evidencias guardadas en: {self.vision.resultados_dir}")
                    print("==============================================================")
                    break
                wps = self.planificador.planificar_a_estrella((pose[0], pose[1]), METAS_MISION[meta_idx])
                print(f"--> Meta {meta_idx+1}/{len(METAS_MISION)}: {METAS_MISION[meta_idx]} ({len(wps)} waypoints)")
                continue

            self.percepcion.set_velocidades(v, w)

            if t - ultimo_log >= 2.0:
                ultimo_log = t
                ev = " [EVASION LIDAR]" if self.planificador.evasion_activa else ""
                print(f"[t={t:5.1f}s] M{meta_idx+1}/{len(METAS_MISION)} | EKF=({pose[0]:+.2f},{pose[1]:+.2f}) | Err={err:.3f}m | Plantas={len(self.vision.clusters)}{ev}")
