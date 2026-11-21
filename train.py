"""train.py — Entrena un agente (CUSTOM/PPO/DQN/A2C) en un escenario.

Fase de EXPLORACIÓN del enunciado: cada algoritmo explora el entorno durante el
entrenamiento. Registra cada EPISODIO de entrenamiento en un CSV (para las
curvas de aprendizaje) y guarda el modelo entrenado.

Uso (en tu máquina, con Webots abierto en worlds/darwin_race.wbt):
    python train.py --algo PPO --exp A2 --seed 0 --timesteps 200000
    python train.py --algo CUSTOM --exp E4 --seed 1 --timesteps 200000 --variant population

Modo DEPURACIÓN (sin Webots, solo para validar el pipeline; NO para el informe):
    python train.py --algo PPO --exp A2 --seed 0 --timesteps 2000 --debug
"""

from __future__ import annotations

import argparse
import os

from experiments.experiment_grid import build_scenarios
from agents.sb3_factory import make_sb3_agent, get_hyperparams
from common import make_env, CSVLogger, TRAIN_COLUMNS


def _scenario(exp_id: str):
    for s in build_scenarios():
        if s["exp_id"] == exp_id:
            return s
    raise ValueError(f"exp_id desconocido: {exp_id}")


def train(algo, exp_id, seed, timesteps, debug=False, out_dir="results",
          models_dir="models", variant="population", **algo_overrides):
    import time as _time
    _t0 = _time.time()
    algo = algo.upper()
    scn = _scenario(exp_id)
    config = scn["config"]
    config.seed = seed

    env, source = make_env(config=config, debug=debug)
    os.makedirs(models_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"train_{algo}_{exp_id}_seed{seed}.csv")
    logger = CSVLogger(csv_path, TRAIN_COLUMNS)

    print(f"[train] algo={algo} exp={exp_id} seed={seed} timesteps={timesteps} "
          f"source={source}")
    if debug:
        print("  *** MODO DEPURACION (sin Webots): resultados NO validos para el informe ***")

    model_path = os.path.join(models_dir, f"{algo}_{exp_id}_seed{seed}")

    if algo == "CUSTOM":
        from agents.custom_agent import CustomAgent
        agent = CustomAgent(env.observation_space, env.action_space, seed=seed,
                            variant=variant, **algo_overrides)

        def cb(info):
            if info.get("discovery"):
                return
            logger.write({
                "source": source, "algorithm": algo, "exp_id": exp_id,
                "config_id": config.config_id(), "seed": seed, "phase": "train",
                "episode": info["episode"], "step": info["step"],
                "ep_return": round(info["ep_return"], 3), "ep_len": info["ep_len"],
                "is_success": int(info["is_success"]),
                "falls": info["falls"], "collisions": info["collisions"],
                "time_to_goal": info.get("time_to_goal") or "",
            })

        # Entorno separado para las rondas de descubrimiento (carreras reales).
        disc_env = None
        if variant != "base":
            disc_env, _ = make_env(config=config, debug=debug)
        agent.learn(env, total_timesteps=timesteps, callback=cb, disc_env=disc_env)
        agent.save(model_path + ".pt")
        if disc_env is not None:
            disc_env.close()
        print(f"[train] modelo guardado en {model_path}.pt")
    else:
        # SB3: envolvemos en Monitor para capturar estadísticas de episodio.
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import BaseCallback

        # info_keywords hace que Monitor conserve estas claves de seguridad en
        # el dict "episode" al terminar cada episodio (si no, las descarta).
        menv = Monitor(env, info_keywords=("is_success", "falls", "collisions"))
        model = make_sb3_agent(algo, menv, seed=seed, **algo_overrides)

        class EpisodeCSVCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.ep = 0

            def _on_step(self) -> bool:
                for info in self.locals.get("infos", []):
                    ep = info.get("episode")
                    if ep is not None:
                        self.ep += 1
                        # Con info_keywords, Monitor coloca esas claves dentro
                        # del dict "episode" (ep); el resto sigue en info.
                        logger.write({
                            "source": source, "algorithm": algo, "exp_id": exp_id,
                            "config_id": config.config_id(), "seed": seed,
                            "phase": "train", "episode": self.ep,
                            "step": self.num_timesteps,
                            "ep_return": round(float(ep["r"]), 3),
                            "ep_len": int(ep["l"]),
                            "is_success": int(ep.get("is_success", info.get("is_success", False))),
                            "falls": int(ep.get("falls", info.get("falls", 0))),
                            "collisions": int(ep.get("collisions", info.get("collisions", 0))),
                            "time_to_goal": info.get("time_to_goal") or "",
                        })
                return True

        model.learn(total_timesteps=timesteps, callback=EpisodeCSVCallback())
        model.save(model_path)
        print(f"[train] modelo guardado en {model_path}.zip")

    env.close()
    _dt = _time.time() - _t0
    print(f"[train] curva de aprendizaje -> {csv_path}")
    print(f"[train] hiperparametros: {get_hyperparams(algo, **algo_overrides) if algo!='CUSTOM' else 'ver CustomHyperparams'}")
    print(f"[train] duracion: {_dt:.1f} s ({timesteps} timesteps)")


def main():
    p = argparse.ArgumentParser(description="Entrena un agente en un escenario.")
    p.add_argument("--algo", required=True, choices=["CUSTOM", "PPO", "DQN", "A2C"])
    p.add_argument("--exp", required=True, help="exp_id del escenario (p.ej. A2, E4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--variant", default="population",
                   choices=["base", "population", "control"],
                   help="solo para CUSTOM: variante del descubrimiento poblacional")
    p.add_argument("--debug", action="store_true",
                   help="usa la sim de juguete SIN Webots (NO valido para el informe)")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--models-dir", default="models")
    args = p.parse_args()

    train(args.algo, args.exp, args.seed, args.timesteps, debug=args.debug,
          out_dir=args.out_dir, models_dir=args.models_dir, variant=args.variant)


if __name__ == "__main__":
    main()
