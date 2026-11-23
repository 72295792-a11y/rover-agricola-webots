# -*- coding: utf-8 -*-
"""Modulo 3: Planificacion (Unidad 3) - Mapa de Ocupacion Dinamico + A* + Campos Potenciales (0% GPS)."""
import heapq
import math


def normalizar_angulo(a):
    return math.atan2(math.sin(a), math.cos(a))


class MapaOcupacion:
    def __init__(self, x_min=-5.0, x_max=5.0, y_min=-5.0, y_max=5.0, resolucion=0.20, margen_seguridad=0.22):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.res = resolucion
        self.margen = margen_seguridad
        self.nx = int(round((x_max - x_min) / resolucion))
        self.ny = int(round((y_max - y_min) / resolucion))
        self.static_grid = [[0] * self.nx for _ in range(self.ny)]
        self.dynamic_ttl = [[0] * self.nx for _ in range(self.ny)]
        self._plantas = [(px, py) for px in (-2.0, 0.0, 2.0) for py in (-2.5, 0.0, 2.5)]
        self._rocas = [(-1.0, 1.25), (1.5, -1.25)]
        self._construir_mapa_estatico()

    def mundo_a_celda(self, x, y):
        gx = max(0, min(self.nx - 1, int((x - self.x_min) / self.res)))
        gy = max(0, min(self.ny - 1, int((y - self.y_min) / self.res)))
        return gx, gy

    def celda_a_mundo(self, gx, gy):
        return self.x_min + (gx + 0.5) * self.res, self.y_min + (gy + 0.5) * self.res

    def esta_ocupada(self, gx, gy, incluir_dinamicos=True):
        if gx < 0 or gx >= self.nx or gy < 0 or gy >= self.ny:
            return True
        if self.static_grid[gy][gx] == 1:
            return True
        return incluir_dinamicos and self.dynamic_ttl[gy][gx] > 0

    def es_celda_dinamica(self, gx, gy):
        return 0 <= gx < self.nx and 0 <= gy < self.ny and self.dynamic_ttl[gy][gx] > 0

    def _marcar_estatico(self, cx, cy, sx, sy):
        rx, ry = sx + self.margen, sy + self.margen
        for gy in range(self.ny):
            wy = self.y_min + (gy + 0.5) * self.res
            if abs(wy - cy) > ry:
                continue
            for gx in range(self.nx):
                wx = self.x_min + (gx + 0.5) * self.res
                if abs(wx - cx) <= rx:
                    self.static_grid[gy][gx] = 1

    def _construir_mapa_estatico(self):
        for gy in range(self.ny):
            for gx in range(self.nx):
                wx, wy = self.celda_a_mundo(gx, gy)
                if abs(wx) > 4.55 or abs(wy) > 4.55:
                    self.static_grid[gy][gx] = 1
        for px, py in self._plantas:
            self._marcar_estatico(px, py, 0.20, 0.20)
        for rx, ry in self._rocas:
            self._marcar_estatico(rx, ry, 0.225, 0.225)

    def es_estatico_cercano(self, wx, wy):
        if abs(wx) > 4.45 or abs(wy) > 4.45:
            return True
        for px, py in self._plantas:
            if abs(wx - px) <= 0.42 and abs(wy - py) <= 0.42:
                return True
        for rx, ry in self._rocas:
            if abs(wx - rx) <= 0.44 and abs(wy - ry) <= 0.44:
                return True
        return False

    def decaer_dinamicos(self):
        for fila in self.dynamic_ttl:
            for gx in range(self.nx):
                if fila[gx] > 0:
                    fila[gx] -= 1

    def marcar_dinamico(self, ox, oy, rover_xy, ttl=85):
        rx, ry = 0.45, 0.24
        rover_x, rover_y = rover_xy
        g0x, g0y = self.mundo_a_celda(ox - rx, oy - ry)
        g1x, g1y = self.mundo_a_celda(ox + rx, oy + ry)
        for gy in range(g0y, g1y + 1):
            wy = self.y_min + (gy + 0.5) * self.res
            for gx in range(g0x, g1x + 1):
                wx = self.x_min + (gx + 0.5) * self.res
                if math.hypot(wx - rover_x, wy - rover_y) < 0.24:
                    continue
                if self.dynamic_ttl[gy][gx] < ttl:
                    self.dynamic_ttl[gy][gx] = ttl

    def celda_libre_cercana(self, gx, gy, incluir_dinamicos=True):
        if not self.esta_ocupada(gx, gy, incluir_dinamicos=incluir_dinamicos):
            return gx, gy
        for radio in range(1, 12):
            for dy in range(-radio, radio + 1):
                for dx in range(-radio, radio + 1):
                    nx, ny = gx + dx, gy + dy
                    if not self.esta_ocupada(nx, ny, incluir_dinamicos=incluir_dinamicos):
                        return nx, ny
        return gx, gy


class Planificador:
    """Planificador global A* con capa dinamica LiDAR + escudo reactivo (0% GPS)."""

    def __init__(self, mapa=None):
        self.mapa = mapa if mapa is not None else MapaOcupacion()
        self.waypoints = []
        self.wp_idx = 0
        self.meta_actual = None
        self.evasion_activa = False
        self._replan_cooldown = 0
        self._pasos_retirada = 0
        self._pasos_estancado = 0
        self._mejor_dist_wp = float("inf")

    def planificar_a_estrella(self, inicio_xy, meta_xy, usar_dinamicos=True):
        self.meta_actual = meta_xy
        inicio = self.mapa.celda_libre_cercana(*self.mapa.mundo_a_celda(*inicio_xy), incluir_dinamicos=False)
        meta = self.mapa.celda_libre_cercana(*self.mapa.mundo_a_celda(*meta_xy), incluir_dinamicos=usar_dinamicos)

        vecinos = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ]

        abiertos = [(0.0, inicio)]
        vino_de = {}
        g_score = {inicio: 0.0}
        cerrados = set()
        encontrado = False

        while abiertos:
            _, actual = heapq.heappop(abiertos)
            if actual in cerrados:
                continue
            if actual == meta:
                encontrado = True
                break
            cerrados.add(actual)

            ax, ay = actual
            for dx, dy, costo in vecinos:
                nx, ny = ax + dx, ay + dy
                if (nx, ny) != meta and self.mapa.esta_ocupada(nx, ny, incluir_dinamicos=usar_dinamicos):
                    continue
                if dx != 0 and dy != 0:
                    if (self.mapa.esta_ocupada(ax + dx, ay, incluir_dinamicos=usar_dinamicos) or
                            self.mapa.esta_ocupada(ax, ay + dy, incluir_dinamicos=usar_dinamicos)):
                        continue
                vecino = (nx, ny)
                if vecino in cerrados:
                    continue
                tg = g_score[actual] + costo
                if tg < g_score.get(vecino, float("inf")):
                    vino_de[vecino] = actual
                    g_score[vecino] = tg
                    heapq.heappush(abiertos, (tg + math.hypot(meta[0] - nx, meta[1] - ny), vecino))

        if not encontrado and usar_dinamicos:
            return self.planificar_a_estrella(inicio_xy, meta_xy, usar_dinamicos=False)
        if not encontrado:
            self.waypoints = [meta_xy]
            self.wp_idx = 0
            return self.waypoints

        ruta = []
        nodo = meta
        while nodo in vino_de:
            ruta.append(nodo)
            nodo = vino_de[nodo]
        ruta.reverse()

        simplificada = []
        prev_dir = None
        for i, c in enumerate(ruta):
            if i == 0 or i == len(ruta) - 1:
                simplificada.append(c)
                if i > 0:
                    prev_dir = (c[0] - ruta[i - 1][0], c[1] - ruta[i - 1][1])
            else:
                cur_dir = (c[0] - ruta[i - 1][0], c[1] - ruta[i - 1][1])
                nxt_dir = (ruta[i + 1][0] - c[0], ruta[i + 1][1] - c[1])
                if cur_dir != nxt_dir or cur_dir != prev_dir:
                    simplificada.append(c)
                prev_dir = cur_dir

        self.waypoints = [self.mapa.celda_a_mundo(gx, gy) for gx, gy in simplificada]
        if self.waypoints:
            self.waypoints[-1] = meta_xy
        self.wp_idx = 0
        self._pasos_estancado = 0
        self._mejor_dist_wp = float("inf")
        return self.waypoints

    def _procesar_lidar(self, pose_ekf, lidar_ranges):
        self.mapa.decaer_dinamicos()
        x, y, theta = pose_ekf
        min_frontal = 8.0
        fx_rep, fy_rep, activos = 0.0, 0.0, 0
        detecto_din = False

        n = len(lidar_ranges)
        if n == 0:
            return fx_rep, fy_rep, min_frontal, False

        fov = 2.0 * math.pi
        d0 = 0.52
        lx = x + 0.15 * math.cos(theta)
        ly = y + 0.15 * math.sin(theta)

        for i, d in enumerate(lidar_ranges):
            if math.isnan(d) or math.isinf(d) or d < 0.18 or d > 7.9:
                continue
            ang = normalizar_angulo((fov / 2.0) - (i + 0.5) * (fov / n))

            if abs(ang) <= 0.36 and d < min_frontal:
                min_frontal = d

            if abs(ang) <= 1.65 and 0.22 <= d <= 2.80 and (i % 2 == 0):
                ox = lx + d * math.cos(theta + ang)
                oy = ly + d * math.sin(theta + ang)
                if not self.mapa.es_estatico_cercano(ox, oy):
                    self.mapa.marcar_dinamico(ox, oy, (x, y), ttl=85)
                    detecto_din = True

            if abs(ang) <= 1.40 and d < d0:
                mag = min(1.2, 0.12 * ((1.0 / d) - (1.0 / d0)) / (d * d))
                fx_rep -= mag * math.cos(ang)
                fy_rep -= mag * math.sin(ang)
                activos += 1

        if activos > 0:
            fx_rep /= activos
            fy_rep /= activos
        return fx_rep, fy_rep, min_frontal, detecto_din

    def _ruta_bloqueada(self, x, y):
        if not self.waypoints or self.wp_idx >= len(self.waypoints):
            return False
        px, py = x, y
        recorrido = 0.0
        for k in range(self.wp_idx, min(len(self.waypoints), self.wp_idx + 5)):
            wx, wy = self.waypoints[k]
            dist = math.hypot(wx - px, wy - py)
            pasos = max(1, int(dist / 0.15))
            for s in range(1, pasos + 1):
                sx = px + (wx - px) * (s / pasos)
                sy = py + (wy - py) * (s / pasos)
                gx, gy = self.mapa.mundo_a_celda(sx, sy)
                if self.mapa.es_celda_dinamica(gx, gy):
                    return True
            recorrido += dist
            if recorrido >= 2.80:
                break
            px, py = wx, wy
        return False

    def _actualizar_waypoint(self, x, y):
        cambio = False
        while self.wp_idx < len(self.waypoints):
            wx, wy = self.waypoints[self.wp_idx]
            tol = 0.30 if self.wp_idx == len(self.waypoints) - 1 else 0.36
            if math.hypot(wx - x, wy - y) <= tol:
                self.wp_idx += 1
                cambio = True
                continue
            break
        if cambio:
            self._pasos_estancado = 0
            self._mejor_dist_wp = float("inf")

    def calcular_comando(self, pose_ekf, lidar_ranges):
        if not self.waypoints or self.wp_idx >= len(self.waypoints):
            return 0.0, 0.0, True, 8.0

        x, y, theta = pose_ekf
        if self._replan_cooldown > 0:
            self._replan_cooldown -= 1

        if self.meta_actual is not None and math.hypot(self.meta_actual[0] - x, self.meta_actual[1] - y) <= 0.32:
            return 0.0, 0.0, True, 8.0

        self._actualizar_waypoint(x, y)
        if self.wp_idx >= len(self.waypoints):
            return 0.0, 0.0, True, 8.0

        fx_rep, fy_rep, min_frontal, detecto_din = self._procesar_lidar(pose_ekf, lidar_ranges)

        # 1. Replanificacion anticipada A* si el obstaculo dinamico invade la ruta
        if self._replan_cooldown == 0 and self.meta_actual is not None and self._ruta_bloqueada(x, y):
            self.planificar_a_estrella((x, y), self.meta_actual, usar_dinamicos=True)
            self._replan_cooldown = 30
            self.evasion_activa = True
        else:
            self.evasion_activa = detecto_din and (min_frontal < 2.20)

        # 2. Escudo de emergencia: retroceso recto controlado (nunca gira contra plantas)
        if min_frontal < 0.62:
            self.evasion_activa = True
            if self._pasos_retirada == 0:
                self._pasos_retirada = 16
                if self._replan_cooldown == 0 and self.meta_actual is not None:
                    self.planificar_a_estrella((x, y), self.meta_actual, usar_dinamicos=True)
                    self._replan_cooldown = 25

        if self._pasos_retirada > 0:
            self._pasos_retirada -= 1
            return -0.24, 0.0, False, min_frontal

        # 3. Anti-estancamiento
        wx, wy = self.waypoints[self.wp_idx]
        dist_wp = math.hypot(wx - x, wy - y)
        if dist_wp < self._mejor_dist_wp - 0.05:
            self._mejor_dist_wp = dist_wp
            self._pasos_estancado = 0
        else:
            self._pasos_estancado += 1
            if self._pasos_estancado > 80 and self._replan_cooldown == 0 and self.meta_actual is not None:
                self.planificar_a_estrella((x, y), self.meta_actual, usar_dinamicos=True)
                self._replan_cooldown = 35
                wx, wy = self.waypoints[self.wp_idx]
                dist_wp = math.hypot(wx - x, wy - y)

        # 4. Seguimiento suave de waypoints (limita |w| para reducir derrape del 4x4)
        ang_meta = normalizar_angulo(math.atan2(wy - y, wx - x) - theta)
        fx_tot = math.cos(ang_meta) + 0.35 * fx_rep
        fy_tot = math.sin(ang_meta) + 0.35 * fy_rep
        err_theta = math.atan2(fy_tot, fx_tot)

        if abs(err_theta) > 0.55:
            v = 0.0
            w = max(-1.35, min(1.35, 1.6 * err_theta))
        else:
            v_base = 0.42
            if self.wp_idx == len(self.waypoints) - 1 and dist_wp < 0.65:
                v_base *= max(0.35, dist_wp / 0.65)
            v = v_base * max(0.30, math.cos(err_theta))
            w = max(-1.35, min(1.35, 1.6 * err_theta))

        return v, w, False, min_frontal
