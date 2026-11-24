"""evaluate.py — Evalúa un agente entrenado SIN exploración.

Fase de "participar en la carrera" en su forma individual: carga un modelo
entrenado y lo corre de forma DETERMINISTA (sin exploración) durante N episodios
en un escenario, registrando recompensa acumulada, éxito, tiempo a meta y
métricas de seguridad en un CSV de evaluación.

Uso:
    python evaluate.py --algo PPO --exp A2 --seed 0 --episodes 10
    python evaluate.py --algo CUSTOM --exp E4 --seed 1 --episodes 10
    python evaluate.py --algo A2C --exp A2 --seed 0 --episodes 5 --debug
"""

from __future__ import annotations

import argparse
import os

from experiments.experiment_grid import build_scenarios
from common import make_env, CSVLogger, EVAL_COLUMNS


def _scenario(exp_id):
    for s in build_scenarios():
        if s["exp_id"] == exp_id:
            return s
    raise ValueError(f"exp_id desconocido: {exp_id}")


def load_policy(algo, model_path, env):
    """Devuelve una función obs -> acción determinista para el algoritmo dado."""
    algo = algo.upper()
    if algo == "CUSTOM":
        from agents.custom_agent import CustomAgent
        agent = CustomAgent.load(model_path + ".pt", env.observation_space, env.action_space)
        return lambda obs: agent.select_action(obs, deterministic=True)
    from stable_baselines3 import PPO, DQN, A2C
    cls = {"PPO": PPO, "DQN": DQN, "A2C": A2C}[algo]
    model = cls.load(model_path)
    return lambda obs: int(model.predict(obs, deterministic=True)[0])


def evaluate(algo, exp_id, seed, episodes, debug=False, out_dir="results",
             models_dir="models", tag=""):
    algo = algo.upper()
    scn = _scenario(exp_id)
    config = scn["config"]
    config.seed = seed
    suffix = f"_{tag}" if tag else ""

    env, source = make_env(config=config, debug=debug)
    model_path = os.path.join(models_dir, f"{algo}_{exp_id}_seed{seed}{suffix}")
    policy = load_policy(algo, model_path, env)

    csv_path = os.path.join(out_dir, f"eval_{algo}_{exp_id}_seed{seed}{suffix}.csv")
    logger = CSVLogger(csv_path, EVAL_COLUMNS)

    print(f"[eval] algo={algo} exp={exp_id} seed={seed} episodes={episodes} source={source}")
    if debug:
        print("  *** MODO DEPURACION (sin Webots): resultados NO validos para el informe ***")

    successes = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        info = {}
        while not done:
            obs, _, term, trunc, info = env.step(policy(obs))
            done = term or trunc
        is_succ = bool(info.get("is_success", False))
        successes += int(is_succ)
        logger.write({
            "source": source, "algorithm": algo, "exp_id": exp_id,
            "config_id": config.config_id(), "seed": seed, "phase": "eval",
            "episode": ep + 1,
            "cumulative_reward": round(float(info.get("cumulative_reward", 0.0)), 3),
            "is_success": int(is_succ),
            "time_to_goal": info.get("time_to_goal") or "",
            "falls": int(info.get("falls", 0)),
            "collisions": int(info.get("collisions", 0)),
            "out_of_track_events": int(info.get("out_of_track_events", 0)),
            "shield_interventions": int(info.get("shield_interventions", 0)),
        })
    env.close()
    print(f"[eval] tasa de exito: {successes}/{episodes} = {successes/episodes:.2%}")
    print(f"[eval] resultados -> {csv_path}")


def main():
    p = argparse.ArgumentParser(description="Evalua un agente entrenado (determinista).")
    p.add_argument("--algo", required=True, choices=["CUSTOM", "PPO", "DQN", "A2C"])
    p.add_argument("--exp", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    evaluate(args.algo, args.exp, args.seed, args.episodes, debug=args.debug,
             out_dir=args.out_dir, models_dir=args.models_dir, tag=args.tag)


if __name__ == "__main__":
    main()
