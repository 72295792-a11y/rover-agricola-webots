# -*- coding: utf-8 -*-
"""Módulo de localización propia con Filtro de Kalman Extendido (EKF): Encoders + IMU."""
import csv
import math
import os

RADIO_RUEDA = 0.10
ANCHO_EJES = 0.44


def normalizar_angulo(a):
    return math.atan2(math.sin(a), math.cos(a))


class LocalizacionEKF:
    def __init__(self, x0=-3.5, y0=-3.5, theta0=0.0, log_csv=None):
        self.x = [float(x0), float(y0), float(theta0)]
        self.theta0 = float(theta0)
        self._imu_offset = None

        self.P = [
            [1e-3, 0.0, 0.0],
            [0.0, 1e-3, 0.0],
            [0.0, 0.0, 1e-3],
        ]
        self.R_imu = 1e-4

        if log_csv is None:
            log_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "localizacion_log.csv")
        self.log_csv = log_csv
        self._errores = []

        self._f = open(self.log_csv, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)
        self._writer.writerow(["tiempo_s", "ekf_x", "ekf_y", "ekf_theta", "gps_x", "gps_y", "error_m"])

    def predecir(self, delta_izq, delta_der):
        d_izq = delta_izq * RADIO_RUEDA
        d_der = delta_der * RADIO_RUEDA
        ds = 0.5 * (d_izq + d_der)
        dth = (d_der - d_izq) / ANCHO_EJES

        th_mid = self.x[2] + 0.5 * dth
        c = math.cos(th_mid)
        s = math.sin(th_mid)

        self.x[0] += ds * c
        self.x[1] += ds * s
        self.x[2] = normalizar_angulo(self.x[2] + dth)

        # Jacobiano F y actualización de covarianza P = F P F^T + Q
        f02 = -ds * s
        f12 = ds * c
        P = self.P

        fp00 = P[0][0] + f02 * P[2][0]
        fp01 = P[0][1] + f02 * P[2][1]
        fp02 = P[0][2] + f02 * P[2][2]

        fp10 = P[1][0] + f12 * P[2][0]
        fp11 = P[1][1] + f12 * P[2][1]
        fp12 = P[1][2] + f12 * P[2][2]

        fp20, fp21, fp22 = P[2][0], P[2][1], P[2][2]

        q_pos = 1e-5 + 0.008 * (ds * ds)
        q_th = 1e-4 + 0.10 * (dth * dth)

        self.P = [
            [fp00 + fp02 * f02 + q_pos, fp01 + fp02 * f12, fp02],
            [fp10 + fp12 * f02, fp11 + fp12 * f12 + q_pos, fp12],
            [fp20 + fp22 * f02, fp21 + fp22 * f12, fp22 + q_th],
        ]

    def actualizar_imu(self, yaw_imu):
        if self._imu_offset is None:
            self._imu_offset = yaw_imu - self.theta0
        yaw_med = normalizar_angulo(yaw_imu - self._imu_offset)

        innov = normalizar_angulo(yaw_med - self.x[2])
        S = self.P[2][2] + self.R_imu
        if S <= 1e-12:
            return

        k0 = self.P[0][2] / S
        k1 = self.P[1][2] / S
        k2 = self.P[2][2] / S

        self.x[0] += k0 * innov
        self.x[1] += k1 * innov
        self.x[2] = normalizar_angulo(self.x[2] + k2 * innov)

        p20, p21, p22 = self.P[2][0], self.P[2][1], self.P[2][2]
        self.P[0][0] -= k0 * p20
        self.P[0][1] -= k0 * p21
        self.P[0][2] -= k0 * p22
        self.P[1][0] -= k1 * p20
        self.P[1][1] -= k1 * p21
        self.P[1][2] -= k1 * p22
        self.P[2][0] -= k2 * p20
        self.P[2][1] -= k2 * p21
        self.P[2][2] -= k2 * p22

    def obtener_estado(self):
        return self.x[0], self.x[1], self.x[2]

    def calcular_error_gps(self, gps_xy, t=0.0):
        """Solo evaluación contra ground truth (NUNCA modifica el estado del EKF)."""
        err = math.hypot(self.x[0] - gps_xy[0], self.x[1] - gps_xy[1])
        self._errores.append(err)
        self._writer.writerow([
            f"{t:.3f}", f"{self.x[0]:.4f}", f"{self.x[1]:.4f}", f"{self.x[2]:.4f}",
            f"{gps_xy[0]:.4f}", f"{gps_xy[1]:.4f}", f"{err:.4f}"
        ])
        return err

    def resumen_errores(self):
        if not self._errores:
            return 0.0, 0.0
        return sum(self._errores) / len(self._errores), max(self._errores)

    def cerrar(self):
        if self._f and not self._f.closed:
            self._f.flush()
            self._f.close()
