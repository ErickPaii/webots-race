"""run_all.py — Ejecuta la rejilla de experimentos con reanudación y cronómetro.

Combina: escenarios x algoritmos (CUSTOM, PPO, DQN, A2C) x semillas. Para cada
combinación: entrena (train) y evalúa (evaluate).

Perfiles:
  --profile fast       : 4 escenarios, pocos timesteps, 1 semilla. Validación.
  --profile night-lite : 20 escenarios x 4 algos x 1 semilla x 60k timesteps.
                         Objetivo del usuario (cabe en 2-3 noches). Cumple el
                         minimo de 20 experimentos del enunciado.
  --profile full       : 23 escenarios x 4 algos x 3 semillas x 200k timesteps.

REANUDACIÓN: si un modelo ya existe en models/ (p.ej. porque una noche se cortó),
esa corrida se SALTA por defecto. Usa --force para re-entrenar todo.

CRONÓMETRO: imprime el tiempo de cada corrida, el acumulado y una ESTIMACIÓN del
tiempo restante (ETA) basada en el promedio de las corridas ya hechas, para que
sepas si cabe en la noche.

Uso:
    # validación sin Webots (NO para el informe):
    python -m experiments.run_all --profile fast --debug

    # objetivo del usuario, con Webots abierto en darwin_race.wbt:
    python -m experiments.run_all --profile night-lite

    # reanudar tras un corte: vuelve a lanzar el MISMO comando; salta lo hecho.
"""

from __future__ import annotations

import argparse
import os
import time

from experiments.experiment_grid import build_scenarios, fast_subset
from train import train
from evaluate import evaluate

ALGOS = ["CUSTOM", "PPO", "DQN", "A2C"]

# 20 escenarios para night-lite (cumple el minimo exacto del enunciado).
NIGHT_LITE_IDS = [
    "A1", "A2", "A3", "B1", "B2", "C1", "C2", "C3", "D1", "D2",
    "D3", "E1", "E2", "E3", "E4", "F1", "F2", "F3", "G1", "G3",
]


def _model_exists(models_dir, algo, exp, seed):
    base = os.path.join(models_dir, f"{algo}_{exp}_seed{seed}")
    return os.path.exists(base + ".zip") or os.path.exists(base + ".pt")


def _fmt(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["fast", "night-lite", "full"], default="fast")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--algos", nargs="+", default=ALGOS)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="re-entrena aunque el modelo ya exista (desactiva reanudacion)")
    p.add_argument("--models-dir", default="models")
    args = p.parse_args()

    if args.profile == "fast":
        scenarios = fast_subset()
        timesteps = args.timesteps or 2000
        seeds = args.seeds or [0]
    elif args.profile == "night-lite":
        wanted = set(NIGHT_LITE_IDS)
        scenarios = [s for s in build_scenarios() if s["exp_id"] in wanted]
        timesteps = args.timesteps or 60_000
        seeds = args.seeds or [0]
    else:  # full
        scenarios = build_scenarios()
        timesteps = args.timesteps or 200_000
        seeds = args.seeds or [0, 1, 2]

    total = len(scenarios) * len(args.algos) * len(seeds)
    print(f"[run_all] perfil={args.profile} escenarios={len(scenarios)} "
          f"algos={args.algos} semillas={seeds} timesteps={timesteps}")
    print(f"[run_all] total de corridas: {total}  (reanudacion: "
          f"{'OFF (--force)' if args.force else 'ON'})")
    if args.debug:
        print("  *** MODO DEPURACION (sin Webots): NADA de esto es valido para el informe ***")

    done = 0
    skipped = 0
    run_times = []
    t_start = time.time()
    for scn in scenarios:
        exp = scn["exp_id"]
        for seed in seeds:
            for algo in args.algos:
                done += 1
                tag = f"({done}/{total}) {algo} {exp} seed={seed}"
                if not args.force and _model_exists(args.models_dir, algo, exp, seed):
                    skipped += 1
                    print(f"[run_all] {tag}  -> SALTADO (modelo ya existe)")
                    continue

                t0 = time.time()
                print(f"\n[run_all] {tag}  train...")
                train(algo, exp, seed, timesteps, debug=args.debug,
                      models_dir=args.models_dir)
                print(f"[run_all] {tag}  eval...")
                evaluate(algo, exp, seed, args.eval_episodes, debug=args.debug,
                         models_dir=args.models_dir)
                dt = time.time() - t0
                run_times.append(dt)

                avg = sum(run_times) / len(run_times)
                remaining = total - done
                eta = avg * remaining
                print(f"[run_all] {tag}  OK en {_fmt(dt)}  | "
                      f"media/corrida {_fmt(avg)} | hechas {done-skipped} "
                      f"saltadas {skipped} | ETA restante ~{_fmt(eta)}")

    total_t = time.time() - t_start
    print(f"\n[run_all] COMPLETADO: {done-skipped} entrenadas, {skipped} saltadas, "
          f"tiempo total {_fmt(total_t)}.")
    print("[run_all] Ahora corre:  python analyze.py")


if __name__ == "__main__":
    main()
