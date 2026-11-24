"""race_supervisor_controller.py — Carrera simultanea 4 robots (cuasi-estatico, carriles paralelos)."""

from __future__ import annotations

import math
import os
import socket
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.config import (  # noqa: E402
    RaceConfig, DEFAULT_HOST, ACTION_GAIT,
    ACTION_FORWARD, ACTION_FORWARD_FAST, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
)
from envs.protocol import send_msg, LineReader  # noqa: E402
from controllers.supervisor_controller.supervisor_controller import compute_observation  # noqa: E402

RACE_PORT = 10002
ROBOT_DEFS = ["DARWIN_CUSTOM", "DARWIN_PPO", "DARWIN_DQN", "DARWIN_A2C"]
ROBOT_LABELS = ["CUSTOM", "PPO", "DQN", "A2C"]
STAND_HEIGHT = 0.30
LANES = [1.2, 0.4, -0.4, -1.2]

STEP_FORWARD = 0.18
STEP_FORWARD_FAST = 0.28
STEP_TURN = 0.30

def main():
    from controller import Supervisor  # type: ignore

    sv = Supervisor()
    timestep = int(sv.getBasicTimeStep())

    robots = [sv.getFromDef(d) for d in ROBOT_DEFS]
    trans = [n.getField("translation") if n else None for n in robots]
    rots = [n.getField("rotation") if n else None for n in robots]
    emitters = [sv.getDevice(f"emitter{i}") for i in range(4)]

    st = [{"x": 0.0, "y": LANES[i], "yaw": 0.0} for i in range(4)]

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((DEFAULT_HOST, RACE_PORT))
    srv.listen(1)
    srv.setblocking(False)
    print(f"[race_supervisor] escuchando en {DEFAULT_HOST}:{RACE_PORT}", flush=True)

    def wait_for_client():
        print("[race_supervisor] esperando conexion de run_race.py...", flush=True)
        while True:
            try:
                c, _ = srv.accept()
                c.setblocking(True)
                print("[race_supervisor] run_race conectado", flush=True)
                return c, LineReader(c)
            except BlockingIOError:
                if sv.step(timestep) == -1:
                    return None

    cfg = RaceConfig()
    finish_order = []
    finish_time = {}
    sim_step = 0

    def apply_pose(i):
        if trans[i] is not None:
            trans[i].setSFVec3f([st[i]["x"], st[i]["y"], STAND_HEIGHT])
            rots[i].setSFRotation([0.0, 0.0, 1.0, st[i]["yaw"]])
            try:
                robots[i].resetPhysics()
            except Exception:
                pass

    def lane_obstacles(i):
        return [(ox, LANES[i] + oy, orad) for (ox, oy, orad) in cfg.obstacles]

    def nearest(i, rx, ry, heading):
        best_d, best_a = cfg.track_length, 0.0
        for (ox, oy, orad) in lane_obstacles(i):
            d = math.hypot(ox - rx, oy - ry) - orad
            if d < best_d:
                best_d = max(0.0, d)
                b = math.atan2(oy - ry, ox - rx)
                best_a = math.atan2(math.sin(b - heading), math.cos(b - heading))
        return best_d, best_a

    def goal_for(i):
        # Cada robot corre en SU carril: meta recta al frente, misma Y de salida.
        return cfg.goal_pos[0], LANES[i]

    def obs_for(i):
        rx, ry, yaw = st[i]["x"], st[i]["y"], st[i]["yaw"]
        gx, gy = goal_for(i)
        nd, na = nearest(i, rx, ry, yaw)
        return compute_observation(rx, ry, yaw, gx, gy, 0.0, 0.0, 0.0,
                                   (0.0, 0.0, 9.81), nd, na,
                                   cfg.track_length, cfg.track_width)

    def do_reset(new_cfg):
        nonlocal cfg, finish_order, finish_time, sim_step
        cfg = new_cfg
        finish_order = []
        finish_time = {}
        sim_step = 0
        for i in range(4):
            st[i] = {"x": 0.0, "y": LANES[i], "yaw": 0.0}
            apply_pose(i)
            if emitters[i] is not None:
                emitters[i].send("0.0 0.0".encode("utf-8"))
        all_obs = []
        for i in range(4):
            all_obs.extend(lane_obstacles(i))
        for j in range(16):
            node = sv.getFromDef(f"OBSTACLE_{j}")
            if node is None:
                continue
            tf = node.getField("translation")
            if j < len(all_obs):
                ox, oy, _r = all_obs[j]
                tf.setSFVec3f([ox, oy, 0.15])
            else:
                tf.setSFVec3f([0.0, 0.0, -5.0])
        sv.step(timestep)
        for i in range(4):
            apply_pose(i)
        return [obs_for(i) for i in range(4)]

    def integrate(i, action):
        action = int(action)
        if action == ACTION_FORWARD:
            d = STEP_FORWARD
        elif action == ACTION_FORWARD_FAST:
            d = STEP_FORWARD_FAST
        else:
            d = 0.0
        if action == ACTION_TURN_LEFT:
            st[i]["yaw"] += STEP_TURN
        elif action == ACTION_TURN_RIGHT:
            st[i]["yaw"] -= STEP_TURN
        st[i]["x"] += d * math.cos(st[i]["yaw"])
        st[i]["y"] += d * math.sin(st[i]["yaw"])

    def do_step(actions):
        nonlocal sim_step
        for i, a in enumerate(actions):
            if ROBOT_LABELS[i] in finish_time:
                continue
            integrate(i, a)
            apply_pose(i)
            x_amp, a_amp = ACTION_GAIT.get(int(a), (0.0, 0.0))
            if emitters[i] is not None:
                emitters[i].send(f"{x_amp} {a_amp}".encode("utf-8"))

        n_sub = max(1, cfg.control_step_ms // timestep)
        for _ in range(n_sub):
            if sv.step(timestep) == -1:
                break
            for i in range(4):
                apply_pose(i)
        sim_step += 1

        obs_list, finished, fell_list = [], [], []
        for i in range(4):
            obs_list.append(obs_for(i))
            label = ROBOT_LABELS[i]
            gx, gy = goal_for(i)
            dist = math.hypot(gx - st[i]["x"], gy - st[i]["y"])
            reached = dist <= cfg.goal_radius
            fell_list.append(False)
            if reached and label not in finish_time:
                finish_time[label] = sim_step
                finish_order.append(label)
            finished.append(label in finish_time)

        race_over = all(finished) or sim_step >= cfg.max_steps
        return obs_list, finished, fell_list, race_over

    client = wait_for_client()
    while client is not None:
        conn, reader = client
        try:
            while True:
                msg = reader.read_msg()
                if msg is None or msg.get("cmd") == "close":
                    print("[race_supervisor] cliente desconectado; espero uno nuevo.", flush=True)
                    break
                if msg["cmd"] == "reset":
                    obs = do_reset(RaceConfig.from_dict(msg["config"]))
                    send_msg(conn, {"obs": obs, "info": {"labels": ROBOT_LABELS}})
                elif msg["cmd"] == "step":
                    obs, finished, fell, over = do_step(msg["actions"])
                    send_msg(conn, {
                        "obs": obs, "finished": finished, "fell": fell,
                        "race_over": over, "step": sim_step,
                        "finish_order": finish_order, "finish_time": finish_time,
                    })
        except (ConnectionError, OSError) as e:
            print(f"[race_supervisor] conexion perdida ({e}); espero uno nuevo.", flush=True)
        finally:
            try:
                conn.close()
            except OSError:
                pass
        client = wait_for_client()

    srv.close()
    print("[race_supervisor] simulacion detenida; controlador finalizado.", flush=True)

if __name__ == "__main__":
    main()
