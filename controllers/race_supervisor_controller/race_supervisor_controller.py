"""race_supervisor_controller.py — Supervisor de la carrera simultánea (4 robots).

Orquesta una carrera de 4 DARwIn-OP a la vez (evaluación, SIN aprendizaje).
Abre un servidor TCP y habla con run_race.py:

    reset(config)       -> coloca los 4 robots en la salida (carriles en Y),
                           coloca obstáculos y devuelve las 4 observaciones.
    step(actions[4])    -> envía a cada robot (canales 0..3) su comando de
                           marcha, avanza la simulación, devuelve 4 obs + flags
                           + estado de la carrera (quién llegó, en qué orden,
                           tiempo de cada uno).

El ORDEN DE LLEGADA y los TIEMPOS se registran aquí y se devuelven a run_race.py
para su CSV. La carrera termina cuando todos llegan, caen o se agota el tiempo.

Eje vertical = Z. Protocolo JSON por línea (ver envs/protocol.py).
"""

from __future__ import annotations

import math
import os
import socket
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.config import RaceConfig, DEFAULT_HOST, ACTION_GAIT  # noqa: E402
from envs.protocol import send_msg, LineReader  # noqa: E402
# Reutilizamos la observación del supervisor individual (misma definición).
from controllers.supervisor_controller.supervisor_controller import compute_observation  # noqa: E402

RACE_PORT = 10002  # puerto distinto al supervisor individual
ROBOT_DEFS = ["DARWIN_CUSTOM", "DARWIN_PPO", "DARWIN_DQN", "DARWIN_A2C"]
ROBOT_LABELS = ["CUSTOM", "PPO", "DQN", "A2C"]
STAND_HEIGHT = 0.227
LANES = [1.2, 0.4, -0.4, -1.2]


def main() -> None:  # pragma: no cover - requiere Webots
    from controller import Supervisor  # type: ignore

    sv = Supervisor()
    timestep = int(sv.getBasicTimeStep())

    robots = [sv.getFromDef(d) for d in ROBOT_DEFS]
    emitters = [sv.getDevice(f"emitter{i}") for i in range(4)]

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((DEFAULT_HOST, RACE_PORT))
    srv.listen(1)
    print(f"[race_supervisor] escuchando en {DEFAULT_HOST}:{RACE_PORT}")
    conn, _ = srv.accept()
    reader = LineReader(conn)
    print("[race_supervisor] run_race conectado")

    cfg = RaceConfig()
    finish_order = []       # labels en orden de llegada
    finish_time = {}        # label -> paso de llegada
    sim_step = 0

    def pose(node):
        t = node.getField("translation").getSFVec3f()
        R = node.getOrientation()
        yaw = math.atan2(R[3], R[0])
        pitch = math.asin(max(-1.0, min(1.0, -R[6])))
        roll = math.atan2(R[7], R[8])
        return t[0], t[1], t[2], roll, pitch, yaw

    def nearest(rx, ry, heading):
        best_d, best_a = cfg.track_length, 0.0
        for (ox, oy, orad) in cfg.obstacles:
            d = math.hypot(ox - rx, oy - ry) - orad
            if d < best_d:
                best_d = max(0.0, d)
                b = math.atan2(oy - ry, ox - rx)
                best_a = math.atan2(math.sin(b - heading), math.cos(b - heading))
        return best_d, best_a

    def obs_for(node):
        rx, ry, rz, roll, pitch, yaw = pose(node)
        gx, gy = cfg.goal_pos
        nd, na = nearest(rx, ry, yaw)
        o = compute_observation(rx, ry, yaw, gx, gy, roll, pitch, 0.0,
                                (0.0, 0.0, 9.81), nd, na,
                                cfg.track_length, cfg.track_width)
        return o, rx, ry, rz, roll, pitch

    def do_reset(new_cfg):
        nonlocal cfg, finish_order, finish_time, sim_step
        cfg = new_cfg
        finish_order = []
        finish_time = {}
        sim_step = 0
        for i, node in enumerate(robots):
            node.getField("translation").setSFVec3f([0.0, LANES[i], STAND_HEIGHT])
            node.getField("rotation").setSFRotation([0, 0, 1, 0])
            node.resetPhysics()
            if emitters[i] is not None:
                emitters[i].send("0.0 0.0".encode("utf-8"))
        sv.step(timestep)
        return [obs_for(n)[0] for n in robots]

    def do_step(actions):
        nonlocal sim_step
        for i, a in enumerate(actions):
            x_amp, a_amp = ACTION_GAIT.get(int(a), (0.0, 0.0))
            if emitters[i] is not None:
                emitters[i].send(f"{x_amp} {a_amp}".encode("utf-8"))
        n_sub = max(1, cfg.control_step_ms // timestep)
        for _ in range(n_sub):
            if sv.step(timestep) == -1:
                break
        sim_step += 1

        obs_list, finished, fell_list = [], [], []
        gx, gy = cfg.goal_pos
        for i, node in enumerate(robots):
            o, rx, ry, rz, roll, pitch = obs_for(node)
            obs_list.append(o)
            label = ROBOT_LABELS[i]
            dist = math.hypot(gx - rx, gy - ry)
            reached = dist <= cfg.goal_radius
            fell = abs(roll) > cfg.fall_roll_threshold or abs(pitch) > cfg.fall_pitch_threshold or rz < STAND_HEIGHT * 0.5
            fell_list.append(fell)
            if reached and label not in finish_time:
                finish_time[label] = sim_step
                finish_order.append(label)
            finished.append(label in finish_time)

        race_over = all(finished) or sim_step >= cfg.max_steps
        return obs_list, finished, fell_list, race_over

    while True:
        msg = reader.read_msg()
        if msg is None or msg.get("cmd") == "close":
            break
        if msg["cmd"] == "reset":
            obs = do_reset(RaceConfig.from_dict(msg["config"]))
            send_msg(conn, {"obs": obs, "info": {"labels": ROBOT_LABELS}})
        elif msg["cmd"] == "step":
            obs, finished, fell, over = do_step(msg["actions"])
            send_msg(conn, {
                "obs": obs,
                "finished": finished,
                "fell": fell,
                "race_over": over,
                "step": sim_step,
                "finish_order": finish_order,
                "finish_time": finish_time,
            })

    conn.close()
    srv.close()
    print("[race_supervisor] terminado. Orden de llegada:", finish_order)


if __name__ == "__main__":
    main()
