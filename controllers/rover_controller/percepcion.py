# -*- coding: utf-8 -*-
"""Modulo 1: Percepcion y actuadores del Rover Agricola 4x4 (Webots R2025a)."""
import math
import numpy as np

RADIO_RUEDA = 0.10
ANCHO_EJES = 0.44
VEL_MAX_RUEDA = 8.0


class Percepcion:
    def __init__(self, robot, timestep):
        self.robot = robot
        self.timestep = timestep
        self.motores = {}
        for nombre in ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
            m = robot.getDevice(nombre)
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
            self.motores[nombre] = m

        self.encoders, self._enc_prev = {}, {}
        for nombre in ("enc_fl", "enc_fr", "enc_rl", "enc_rr"):
            e = robot.getDevice(nombre)
            e.enable(timestep)
            self.encoders[nombre] = e
            self._enc_prev[nombre] = None

        self.lidar = robot.getDevice("lidar")
        self.lidar.enable(timestep)
        self.camera = robot.getDevice("camera")
        self.camera.enable(timestep)
        self.cam_w = int(self.camera.getWidth())
        self.cam_h = int(self.camera.getHeight())
        self.cam_fov = float(self.camera.getFov())
        self.imu = robot.getDevice("imu")
        self.imu.enable(timestep)
        self.gps_ground_truth = robot.getDevice("gps_ground_truth")
        self.gps_ground_truth.enable(timestep)

    def _delta_encoder(self, nombre):
        val = float(self.encoders[nombre].getValue())
        if math.isnan(val):
            return 0.0
        prev = self._enc_prev[nombre]
        self._enc_prev[nombre] = val
        return 0.0 if prev is None else (val - prev)

    def set_velocidades(self, v_lineal, w_angular):
        v_izq = v_lineal - (w_angular * ANCHO_EJES / 2.0)
        v_der = v_lineal + (w_angular * ANCHO_EJES / 2.0)
        w_izq = max(-VEL_MAX_RUEDA, min(VEL_MAX_RUEDA, v_izq / RADIO_RUEDA))
        w_der = max(-VEL_MAX_RUEDA, min(VEL_MAX_RUEDA, v_der / RADIO_RUEDA))
        self.motores["wheel_fl"].setVelocity(w_izq)
        self.motores["wheel_rl"].setVelocity(w_izq)
        self.motores["wheel_fr"].setVelocity(w_der)
        self.motores["wheel_rr"].setVelocity(w_der)

    def detener(self):
        self.set_velocidades(0.0, 0.0)

    def leer_imagen_bgr(self):
        raw = self.camera.getImage()
        if raw is None:
            return None
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape((self.cam_h, self.cam_w, 4))
        return bgra[:, :, :3].copy()

    def leer_sensores(self):
        delta_izq = 0.5 * (self._delta_encoder("enc_fl") + self._delta_encoder("enc_rl"))
        delta_der = 0.5 * (self._delta_encoder("enc_fr") + self._delta_encoder("enc_rr"))
        rpy = self.imu.getRollPitchYaw()
        yaw = float(rpy[2]) if (rpy and not math.isnan(rpy[2])) else 0.0
        raw_lidar = self.lidar.getRangeImage()
        lidar_range = [float(r) for r in raw_lidar] if raw_lidar else []
        pos = self.gps_ground_truth.getValues()
        gps_x = float(pos[0]) if not math.isnan(pos[0]) else 0.0
        gps_y = float(pos[1]) if not math.isnan(pos[1]) else 0.0
        return {
            "encoder_delta_izq": delta_izq,
            "encoder_delta_der": delta_der,
            "imu_yaw": yaw,
            "lidar_range": lidar_range,
            "gps_x": gps_x,
            "gps_y": gps_y,
            "camera_fov": self.cam_fov,
        }
