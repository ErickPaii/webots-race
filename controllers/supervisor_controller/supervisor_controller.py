"""Controlador Supervisor de Webots para la carrera del DARwIn-OP.

Corre DENTRO de Webots (lo lanza el nodo Robot con controller
"supervisor_controller" y supervisor TRUE). Abre un servidor TCP en localhost
y atiende las peticiones del entorno Gymnasium (ver envs/protocol.py):

    reset(config) -> reposiciona el robot en la salida, coloca obstáculos
                     según la configuración y devuelve la observación inicial.
    step(action)  -> traduce la acción al gait manager, avanza la simulación
                     `control_step_ms`, lee sensores y posición, calcula el
                     reward y detecta eventos de seguridad.
    close()       -> cierra la conexión.

El balance de la marcha lo resuelve el gait manager nativo del DARwIn-OP
(managers RobotisOp2GaitManager / MotionManager), por lo que las acciones solo
fijan amplitud de avance (X) y de giro (A). Así NO se entrena el equilibrio
desde cero (justificado en el informe).

IMPORTANTE: este archivo usa la API `controller` de Webots y los managers de
`managers` (compilados con Webots). Solo se ejecuta dentro de Webots; fuera de
Webots esos imports fallan a propósito. La lógica pura (reward, observación)
se mantiene en funciones testeables sin Webots en este mismo archivo.
"""

from __future__ import annotations

import math
import os
import socket
import sys

# Permite importar envs/ y safety/ desde la raíz del proyecto webots_race/.
# __file__ = .../webots_race/controllers/supervisor_controller/supervisor_controller.py
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.config import (  # noqa: E402
    RaceConfig,
    RewardConfig,
    DEFAULT_HOST,
    DEFAULT_PORT,
    ACTION_GAIT,
)
from envs.protocol import send_msg, LineReader  # noqa: E402


# --------------------------------------------------------------------------
# Lógica pura, sin Webots: se puede importar y testear fuera del simulador.
# --------------------------------------------------------------------------
def compute_observation(
    robot_x: float,
    robot_y: float,
    heading: float,
    goal_x: float,
    goal_y: float,
    roll: float,
    pitch: float,
    yaw_rate: float,
    accel: tuple,
    nearest_dist: float,
    nearest_angle: float,
    track_length: float,
    track_width: float,
) -> list:
    """Construye el vector de observación normalizado de 12 dimensiones.

    Plano de la pista: XY (X avance, Y lateral). accel = (ax, ay, az) con la
    gravedad nominal sobre Z (eje vertical del PROTO Darwin-op en R2025a).
    """
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    dist = math.hypot(dx, dy)
    goal_bearing = math.atan2(dy, dx)
    heading_error = math.atan2(
        math.sin(goal_bearing - heading), math.cos(goal_bearing - heading)
    )
    norm_len = max(track_length, 1e-6)
    norm_w = max(track_width, 1e-6)
    g = 9.81
    obs = [
        _clip(dx / norm_len),
        _clip(dy / norm_w),
        _clip(dist / norm_len),
        _clip(heading_error / math.pi),
        _clip(nearest_dist / norm_len),
        _clip(nearest_angle / math.pi),
        _clip(roll / math.pi),
        _clip(pitch / math.pi),
        _clip(yaw_rate / 10.0),
        _clip(accel[0] / (4 * g)),
        _clip(accel[1] / (4 * g)),
        _clip(accel[2] / (4 * g)),
    ]
    return obs


def compute_reward(
    prev_dist: float,
    cur_dist: float,
    reached_goal: bool,
    collided: bool,
    fell: bool,
    out_of_track: bool,
    rc: RewardConfig,
) -> float:
    """Reward shaping: progreso + bonus - penalizaciones."""
    r = rc.w_progress * (prev_dist - cur_dist)  # avanzar reduce la distancia -> r>0
    r -= rc.pen_time
    if collided:
        r -= rc.pen_collision
    if fell:
        r -= rc.pen_fall
    if out_of_track:
        r -= rc.pen_out_of_track
    if reached_goal:
        r += rc.bonus_goal
    return r


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------
# A partir de aquí: código que SOLO corre dentro de Webots.
# --------------------------------------------------------------------------
def main() -> None:  # pragma: no cover - requiere Webots
    from controller import Supervisor  # type: ignore

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    # Nodo del robot y sus campos (DEF DARWIN en el mundo).
    robot_node = supervisor.getFromDef("DARWIN")
    if robot_node is None:
        raise RuntimeError("No se encontró el nodo DEF DARWIN en el mundo.")
    trans_field = robot_node.getField("translation")
    rot_field = robot_node.getField("rotation")

    # Emitter para enviar comandos de marcha al controlador del robot (canal 0).
    emitter = supervisor.getDevice("emitter")
    if emitter is None:
        print("[supervisor] AVISO: no se encontró el Emitter 'emitter'. "
              "El robot no recibirá comandos de marcha.")

    # Nodos de obstáculos predefinidos en el mundo (DEF OBSTACLE_i). El
    # Supervisor los reposiciona o los esconde según la RaceConfig, así se
    # parametriza el número/posición de obstáculos sin editar el .wbt.
    MAX_WORLD_OBSTACLES = 8
    obstacle_nodes = []
    for i in range(MAX_WORLD_OBSTACLES):
        n = supervisor.getFromDef(f"OBSTACLE_{i}")
        if n is not None:
            obstacle_nodes.append(n)

    def place_obstacles(obstacles):
        """Coloca los obstáculos de la config; esconde los sobrantes bajo el suelo."""
        for i, node in enumerate(obstacle_nodes):
            tf = node.getField("translation")
            if i < len(obstacles):
                ox, oz, _orad = obstacles[i]
                tf.setSFVec3f([ox, 0.15, oz])
            else:
                tf.setSFVec3f([0.0, -5.0, 0.0])  # fuera de la pista, invisible

    # Sensores leídos a través del propio supervisor (si están habilitados en
    # el PROTO). Para IMU/acelerómetro reales se usa el controlador del robot;
    # aquí aproximamos roll/pitch desde la orientación del nodo supervisor, que
    # es suficiente para detectar caídas y para la observación.
    rc = RewardConfig()

    # Servidor TCP.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((DEFAULT_HOST, DEFAULT_PORT))
    srv.listen(1)
    print(f"[supervisor] escuchando en {DEFAULT_HOST}:{DEFAULT_PORT}")
    conn, _ = srv.accept()
    reader = LineReader(conn)
    print("[supervisor] entorno conectado")

    cfg = RaceConfig()
    prev_dist = 0.0

    # NOTA de ejes (R2025a, PROTO Darwin-op): el eje VERTICAL es Z. La pista
    # está en el plano XY: X = avance (salida->meta), Y = lateral (ancho).
    # Los campos de RaceConfig nombrados (x, z) representan el par planar; aquí
    # el segundo valor se mapea al eje Y de Webots (lateral) y Z se reserva a la
    # altura. Para mantener coherencia, en este controlador usamos: posición
    # planar = (x, y), altura = z.
    STAND_HEIGHT = 0.227  # altura de pie del Darwin-op según el PROTO

    def read_pose():
        """Devuelve (x, y, z_altura, roll, pitch, yaw) usando la matriz de orientación."""
        t = trans_field.getSFVec3f()  # [x, y, z]
        # getOrientation() devuelve la matriz de rotación 3x3 (fila mayor) del nodo.
        R = robot_node.getOrientation()
        # Con Z vertical: yaw = atan2(R[3], R[0]) (rotación alrededor de Z).
        yaw = math.atan2(R[3], R[0])
        # roll y pitch a partir de la componente vertical de la matriz.
        pitch = math.asin(max(-1.0, min(1.0, -R[6])))
        roll = math.atan2(R[7], R[8])
        return t[0], t[1], t[2], roll, pitch, yaw

    def nearest_obstacle(rx, ry, heading):
        """Distancia y ángulo relativo al obstáculo más cercano en el plano XY."""
        best_d, best_a = 1e9, 0.0
        for (ox, oy, orad) in cfg.obstacles:
            d = math.hypot(ox - rx, oy - ry) - orad
            if d < best_d:
                best_d = d
                bearing = math.atan2(oy - ry, ox - rx)
                best_a = math.atan2(
                    math.sin(bearing - heading), math.cos(bearing - heading)
                )
        if best_d > 1e8:
            best_d = cfg.track_length  # sin obstáculos
        return max(best_d, 0.0), best_a

    def do_reset(new_cfg: RaceConfig):
        nonlocal cfg, prev_dist
        cfg = new_cfg
        sx, sy = cfg.start_pos
        trans_field.setSFVec3f([sx, sy, STAND_HEIGHT])
        rot_field.setSFRotation([0, 0, 1, 0])  # de pie, mirando +X
        robot_node.resetPhysics()
        place_obstacles(cfg.obstacles)
        # Detener la marcha al reiniciar.
        if emitter is not None:
            emitter.send("0.0 0.0".encode("utf-8"))
        supervisor.step(timestep)
        rx, ry, rz, roll, pitch, yaw = read_pose()
        gx, gy = cfg.goal_pos
        prev_dist = math.hypot(gx - rx, gy - ry)
        nd, na = nearest_obstacle(rx, ry, yaw)
        obs = compute_observation(
            rx, ry, yaw, gx, gy, roll, pitch, 0.0, (0.0, 0.0, 9.81),
            nd, na, cfg.track_length, cfg.track_width,
        )
        return obs, rx, ry

    def do_step(action: int):
        nonlocal prev_dist
        x_amp, a_amp = ACTION_GAIT.get(int(action), (0.0, 0.0))
        # Enviar el comando de marcha (amplitud avance X, amplitud giro A) al
        # controlador del robot por el Emitter. El controlador darwin_agent lo
        # recibe y lo aplica con el gait manager nativo del DARwIn-OP.
        if emitter is not None:
            emitter.send(f"{x_amp} {a_amp}".encode("utf-8"))
        # Avanzar la simulación los ms configurados (varios sub-pasos del
        # basicTimeStep) para que la marcha progrese entre cada step() del env.
        n_sub = max(1, cfg.control_step_ms // timestep)
        for _ in range(n_sub):
            if supervisor.step(timestep) == -1:
                break
        rx, ry, rz, roll, pitch, yaw = read_pose()
        gx, gy = cfg.goal_pos
        cur_dist = math.hypot(gx - rx, gy - ry)
        nd, na = nearest_obstacle(rx, ry, yaw)

        reached = cur_dist <= cfg.goal_radius
        fell = (abs(roll) > cfg.fall_roll_threshold
                or abs(pitch) > cfg.fall_pitch_threshold
                or rz < STAND_HEIGHT * 0.5)  # también caído si baja mucho el torso
        out = abs(ry) > cfg.track_width / 2 or rx < -0.5 or rx > cfg.track_length + 0.5
        collided = nd < 0.12  # contacto efectivo con obstáculo

        reward = compute_reward(prev_dist, cur_dist, reached, collided, fell, out, rc)
        prev_dist = cur_dist
        terminated = reached or fell or out
        obs = compute_observation(
            rx, ry, yaw, gx, gy, roll, pitch, 0.0, (0.0, 0.0, 9.81),
            nd, na, cfg.track_length, cfg.track_width,
        )
        info = {
            "reached_goal": reached,
            "collided": collided,
            "fell": fell,
            "out_of_track": out,
            "nearest_dist": nd,
            "nearest_angle": na,
            "robot_x": rx,
            "robot_y": ry,
        }
        return obs, reward, terminated, False, info

    while True:
        msg = reader.read_msg()
        if msg is None or msg.get("cmd") == "close":
            break
        cmd = msg["cmd"]
        if cmd == "reset":
            obs, rx0, ry0 = do_reset(RaceConfig.from_dict(msg["config"]))
            send_msg(conn, {"obs": obs, "reward": 0.0, "terminated": False,
                            "truncated": False,
                            "info": {"robot_x": rx0, "robot_y": ry0}})
        elif cmd == "step":
            obs, reward, term, trunc, info = do_step(msg["action"])
            send_msg(conn, {"obs": obs, "reward": reward, "terminated": term,
                            "truncated": trunc, "info": info})

    conn.close()
    srv.close()
    print("[supervisor] terminado")


if __name__ == "__main__":
    main()
