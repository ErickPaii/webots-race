"""run_race.py — Carrera SIMULTÁNEA de los 4 agentes (evaluación, sin aprendizaje).

Carga los 4 modelos entrenados (CUSTOM, PPO, DQN, A2C) para un escenario y los
hace competir A LA VEZ en el mundo worlds/darwin_race_4.wbt, controlados por el
race_supervisor_controller vía TCP (puerto 10002). Registra el ORDEN DE LLEGADA
y los TIEMPOS en results/race_<exp>.csv.

Decisión de diseño (justificada en el informe): cada agente se ENTRENA por
separado en pistas idénticas (RL estacionario), pero la CARRERA FINAL es con los
4 robots simultáneamente, como exige el enunciado ("competir contra otros robots
DARwIn-OP"). Además guardamos la comparación por tiempos de las corridas
separadas (ver analyze.py, que cruza los CSV de evaluate.py).

Uso (con worlds/darwin_race_4.wbt abierto en Webots):
    python run_race.py --exp A2 --seed 0

Modo DEPURACION (sin Webots; NO valido para el informe): simula la carrera con
4 instancias del entorno de juguete.
    python run_race.py --exp A2 --seed 0 --debug
"""

from __future__ import annotations

import argparse
import csv
import os
import socket

from experiments.experiment_grid import build_scenarios
from envs.config import DEFAULT_HOST
from envs.protocol import send_msg, LineReader

RACE_PORT = 10002
ALGOS = ["CUSTOM", "PPO", "DQN", "A2C"]
SOURCE_WEBOTS = "WEBOTS_REAL"
SOURCE_DEBUG = "DEBUG_SIM_NO_WEBOTS"
RACE_COLUMNS = ["source", "exp_id", "config_id", "seed", "algorithm",
                "finish_rank", "finish_step", "finished"]


def _scenario(exp_id):
    for s in build_scenarios():
        if s["exp_id"] == exp_id:
            return s
    raise ValueError(f"exp_id desconocido: {exp_id}")


def _load_policies(exp_id, seed, models_dir, obs_space, act_space):
    """Carga las 4 políticas deterministas. Devuelve dict algo -> policy(obs)->a."""
    policies = {}
    from stable_baselines3 import PPO, DQN, A2C
    from agents.custom_agent import CustomAgent
    for algo in ALGOS:
        path = os.path.join(models_dir, f"{algo}_{exp_id}_seed{seed}")
        if algo == "CUSTOM":
            agent = CustomAgent.load(path + ".pt", obs_space, act_space)
            policies[algo] = lambda o, ag=agent: ag.select_action(o, deterministic=True)
        else:
            cls = {"PPO": PPO, "DQN": DQN, "A2C": A2C}[algo]
            model = cls.load(path)
            policies[algo] = lambda o, m=model: int(m.predict(o, deterministic=True)[0])
    return policies


def run_race_webots(exp_id, seed, models_dir, out_dir):
    scn = _scenario(exp_id)
    config = scn["config"]
    config.seed = seed

    # Conectar al race supervisor.
    sock = socket.create_connection((DEFAULT_HOST, RACE_PORT), timeout=30)
    reader = LineReader(sock)

    # Necesitamos los espacios para cargar las políticas: los tomamos del env real.
    from envs.darwin_race_env import DarwinRaceEnv
    tmp = DarwinRaceEnv(config=config)  # solo para los espacios (no conecta aún)
    obs_space, act_space = tmp.observation_space, tmp.action_space
    policies = _load_policies(exp_id, seed, models_dir, obs_space, act_space)

    send_msg(sock, {"cmd": "reset", "config": config.to_dict()})
    resp = reader.read_msg(timeout=60)
    obs_list = resp["obs"]

    race_over = False
    last = {}
    while not race_over:
        actions = [policies[ALGOS[i]](obs_list[i]) for i in range(4)]
        send_msg(sock, {"cmd": "step", "actions": actions})
        resp = reader.read_msg(timeout=60)
        obs_list = resp["obs"]
        race_over = resp["race_over"]
        last = resp

    send_msg(sock, {"cmd": "close"})
    sock.close()

    return config, last.get("finish_order", []), last.get("finish_time", {})


def run_race_debug(exp_id, seed, models_dir, out_dir):
    """Carrera simulada SIN Webots (4 entornos de juguete). NO para el informe."""
    scn = _scenario(exp_id)
    config = scn["config"]
    config.seed = seed
    from envs.debug_sim_env import DebugSimEnv

    envs = [DebugSimEnv(config=config) for _ in ALGOS]
    obs_space, act_space = envs[0].observation_space, envs[0].action_space
    policies = _load_policies(exp_id, seed, models_dir, obs_space, act_space)

    obs = [e.reset(seed=seed)[0] for e in envs]
    done = [False] * 4
    finish_order, finish_time = [], {}
    step = 0
    while not all(done) and step < config.max_steps:
        step += 1
        for i, e in enumerate(envs):
            if done[i]:
                continue
            o, _, term, trunc, info = e.step(policies[ALGOS[i]](obs[i]))
            obs[i] = o
            if info.get("reached_goal") and ALGOS[i] not in finish_time:
                finish_time[ALGOS[i]] = step
                finish_order.append(ALGOS[i])
            done[i] = term or trunc
    for e in envs:
        e.close()
    return config, finish_order, finish_time


def write_race_csv(out_dir, exp_id, seed, source, config, finish_order, finish_time):
    path = os.path.join(out_dir, f"race_{exp_id}_seed{seed}.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RACE_COLUMNS)
        w.writeheader()
        for algo in ALGOS:
            rank = finish_order.index(algo) + 1 if algo in finish_order else ""
            w.writerow({
                "source": source, "exp_id": exp_id,
                "config_id": config.config_id(), "seed": seed, "algorithm": algo,
                "finish_rank": rank,
                "finish_step": finish_time.get(algo, ""),
                "finished": int(algo in finish_time),
            })
    return path


def main():
    p = argparse.ArgumentParser(description="Carrera simultanea de los 4 agentes.")
    p.add_argument("--exp", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--models-dir", default="models")
    args = p.parse_args()

    if args.debug:
        print("  *** MODO DEPURACION (sin Webots): resultado de carrera NO valido para el informe ***")
        source = SOURCE_DEBUG
        config, order, times = run_race_debug(args.exp, args.seed, args.models_dir, args.out_dir)
    else:
        source = SOURCE_WEBOTS
        config, order, times = run_race_webots(args.exp, args.seed, args.models_dir, args.out_dir)

    path = write_race_csv(args.out_dir, args.exp, args.seed, source, config, order, times)
    print(f"[race] orden de llegada: {order}")
    print(f"[race] tiempos (paso): {times}")
    print(f"[race] resultado -> {path}")


if __name__ == "__main__":
    main()
