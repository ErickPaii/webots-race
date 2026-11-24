"""Controlador Supervisor de Webots para la carrera del DARwIn-OP (locomocion cuasi-estatica)."""

from __future__ import annotations

import math
import os
import socket
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.config import (  # noqa: E402
    RaceConfig, RewardConfig, DEFAULT_HOST, DEFAULT_PORT, ACTION_GAIT,
    ACTION_FORWARD, ACTION_FORWARD_FAST, ACTION_TURN_LEFT, ACTION_TURN_RIGHT,
)
from envs.protocol import send_msg, LineReader  # noqa: E402

def compute_observation(robot_x, robot_y, heading, goal_x, goal_y, roll, pitch,
                        yaw_rate, accel, nearest_dist, nearest_angle,
                        track_length, track_width):
    dx = goal_x - robot_x
    dy = goal_y - robot_y
    dist = math.hypot(dx, dy)
    goal_bearing = math.atan2(dy, dx)
    heading_error = math.atan2(math.sin(goal_bearing - heading),
                               math.cos(goal_bearing - heading))
    norm_len = max(track_length, 1e-6)
    norm_w = max(track_width, 1e-6)
    g = 9.81
    return [
        _clip(dx / norm_len), _clip(dy / norm_w), _clip(dist / norm_len),
        _clip(heading_error / math.pi), _clip(nearest_dist / norm_len),
        _clip(nearest_angle / math.pi), _clip(roll / math.pi),
        _clip(pitch / math.pi), _clip(yaw_rate / 10.0),
        _clip(accel[0] / (4 * g)), _clip(accel[1] / (4 * g)),
        _clip(accel[2] / (4 * g)),
    ]

def compute_reward(prev_dist, cur_dist, reached_goal, collided, fell,
                   out_of_track, rc):
    r = rc.w_progress * (prev_dist - cur_dist)
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

def _clip(v, lo=-1.0, hi=1.0):
    return max(lo, min(hi, v))

def main():
    from controller import Supervisor  # type: ignore

    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    robot_node = supervisor.getFromDef("DARWIN")
    if robot_node is None:
        raise RuntimeError("No se encontro el nodo DEF DARWIN en el mundo.")
    trans_field = robot_node.getField("translation")
    rot_field = robot_node.getField("rotation")

    emitter = supervisor.getDevice("emitter")
    if emitter is None:
        print("[supervisor] AVISO: no se encontro el Emitter 'emitter'.", flush=True)

    MAX_WORLD_OBSTACLES = 8
    obstacle_nodes = []
    for i in range(MAX_WORLD_OBSTACLES):
        n = supervisor.getFromDef(f"OBSTACLE_{i}")
        if n is not None:
            obstacle_nodes.append(n)

    def place_obstacles(obstacles):
        for i, node in enumerate(obstacle_nodes):
            tf = node.getField("translation")
            if i < len(obstacles):
                ox, oy, _orad = obstacles[i]
                tf.setSFVec3f([ox, oy, 0.15])
            else:
                tf.setSFVec3f([0.0, 0.0, -5.0])

    rc = RewardConfig()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((DEFAULT_HOST, DEFAULT_PORT))
    srv.listen(1)
    srv.setblocking(False)
    print(f"[supervisor] escuchando en {DEFAULT_HOST}:{DEFAULT_PORT}", flush=True)

    def wait_for_client():
        print("[supervisor] esperando conexion del entorno "
              "(lanza train.py/test_walk.py)...", flush=True)
        while True:
            try:
                c, _ = srv.accept()
                c.setblocking(True)
                print("[supervisor] entorno conectado", flush=True)
                return c, LineReader(c)
            except BlockingIOError:
                if supervisor.step(timestep) == -1:
                    return None

    cfg = RaceConfig()
    prev_dist = 0.0
    STAND_HEIGHT = 0.30

    STEP_FORWARD = 0.18
    STEP_FORWARD_FAST = 0.28
    STEP_TURN = 0.30
    state = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    def apply_pose():
        trans_field.setSFVec3f([state["x"], state["y"], STAND_HEIGHT])
        rot_field.setSFRotation([0.0, 0.0, 1.0, state["yaw"]])
        try:
            robot_node.resetPhysics()
        except Exception:
            pass

    def nearest_obstacle(rx, ry, heading):
        best_d, best_a = 1e9, 0.0
        for (ox, oy, orad) in cfg.obstacles:
            d = math.hypot(ox - rx, oy - ry) - orad
            if d < best_d:
                best_d = d
                bearing = math.atan2(oy - ry, ox - rx)
                best_a = math.atan2(math.sin(bearing - heading),
                                    math.cos(bearing - heading))
        if best_d > 1e8:
            best_d = cfg.track_length
        return max(best_d, 0.0), best_a

    def do_reset(new_cfg):
        nonlocal cfg, prev_dist
        cfg = new_cfg
        sx, sy = cfg.start_pos
        state["x"], state["y"], state["yaw"] = sx, sy, 0.0
        apply_pose()
        place_obstacles(cfg.obstacles)
        if emitter is not None:
            emitter.send("0.0 0.0".encode("utf-8"))
        supervisor.step(timestep)
        apply_pose()
        rx, ry, yaw = state["x"], state["y"], state["yaw"]
        gx, gy = cfg.goal_pos
        prev_dist = math.hypot(gx - rx, gy - ry)
        nd, na = nearest_obstacle(rx, ry, yaw)
        obs = compute_observation(rx, ry, yaw, gx, gy, 0.0, 0.0, 0.0,
                                  (0.0, 0.0, 9.81), nd, na,
                                  cfg.track_length, cfg.track_width)
        return obs, rx, ry

    def do_step(action):
        nonlocal prev_dist
        action = int(action)
        x_amp, a_amp = ACTION_GAIT.get(action, (0.0, 0.0))
        if emitter is not None:
            emitter.send(f"{x_amp} {a_amp}".encode("utf-8"))

        if action == ACTION_FORWARD:
            d = STEP_FORWARD
        elif action == ACTION_FORWARD_FAST:
            d = STEP_FORWARD_FAST
        else:
            d = 0.0
        if action == ACTION_TURN_LEFT:
            state["yaw"] += STEP_TURN
        elif action == ACTION_TURN_RIGHT:
            state["yaw"] -= STEP_TURN
        state["x"] += d * math.cos(state["yaw"])
        state["y"] += d * math.sin(state["yaw"])
        apply_pose()

        n_sub = max(1, cfg.control_step_ms // timestep)
        for _ in range(n_sub):
            if supervisor.step(timestep) == -1:
                break
            apply_pose()

        rx, ry, yaw = state["x"], state["y"], state["yaw"]
        gx, gy = cfg.goal_pos
        cur_dist = math.hypot(gx - rx, gy - ry)
        nd, na = nearest_obstacle(rx, ry, yaw)

        reached = cur_dist <= cfg.goal_radius
        fell = False
        out = abs(ry) > cfg.track_width / 2 or rx < -0.5 or rx > cfg.track_length + 0.5
        collided = nd < 0.12
        roll = pitch = 0.0

        reward = compute_reward(prev_dist, cur_dist, reached, collided, fell, out, rc)
        prev_dist = cur_dist
        terminated = reached or out
        obs = compute_observation(rx, ry, yaw, gx, gy, roll, pitch, 0.0,
                                  (0.0, 0.0, 9.81), nd, na,
                                  cfg.track_length, cfg.track_width)
        info = {"reached_goal": reached, "collided": collided, "fell": fell,
                "out_of_track": out, "nearest_dist": nd, "nearest_angle": na,
                "robot_x": rx, "robot_y": ry}
        return obs, reward, terminated, False, info

    client = wait_for_client()
    while client is not None:
        conn, reader = client
        try:
            while True:
                msg = reader.read_msg()
                if msg is None:
                    print("[supervisor] cliente desconectado; espero uno nuevo.", flush=True)
                    break
                if msg.get("cmd") == "close":
                    print("[supervisor] cliente cerro la sesion; espero uno nuevo.", flush=True)
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
        except (ConnectionError, OSError) as e:
            print(f"[supervisor] conexion perdida ({e}); espero uno nuevo.", flush=True)
        finally:
            try:
                conn.close()
            except OSError:
                pass
        client = wait_for_client()

    srv.close()
    print("[supervisor] simulacion detenida; controlador finalizado.", flush=True)

if __name__ == "__main__":
    main()
