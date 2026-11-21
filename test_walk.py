"""test_walk.py — Prueba de marcha SIN entrenar.

Hace avanzar al robot en linea recta con la accion "avanzar" durante unos
segundos y reporta:
  - distancia recorrida en X (hacia la meta),
  - desplazamiento lateral en Y,
  - si el robot se cayo,
  - posiciones inicial y final.

Sirve para verificar que el CPG produce una marcha estable ANTES de invertir
tiempo entrenando. Usa el mismo entorno Gymnasium que el entrenamiento, asi que
requiere Webots abierto en worlds/darwin_race.wbt y en Play.

Uso:
    python test_walk.py                 # ~8 s de marcha hacia adelante
    python test_walk.py --steps 40 --action forward_fast
    python test_walk.py --debug         # sin Webots (sim de juguete, solo pipeline)

Interpretacion:
  - Si avanza > ~0.5 m sin caerse: la marcha sirve, puedes entrenar.
  - Si no avanza o se cae enseguida: el CPG necesita ajuste (avisamelo con la
    salida y los avisos de la consola de Webots).
"""

from __future__ import annotations

import argparse

from envs.config import RaceConfig, ACTION_NAMES
from common import make_env

# nombre de accion -> indice
NAME_TO_ACTION = {v: k for k, v in ACTION_NAMES.items()}


def main():
    p = argparse.ArgumentParser(description="Prueba de marcha sin entrenar.")
    p.add_argument("--steps", type=int, default=25,
                   help="numero de pasos de entorno (cada uno ~0.32 s de sim)")
    p.add_argument("--action", default="forward",
                   choices=list(NAME_TO_ACTION.keys()),
                   help="accion a repetir (por defecto: forward)")
    p.add_argument("--debug", action="store_true",
                   help="usa la sim de juguete SIN Webots (solo prueba el pipeline)")
    args = p.parse_args()

    action = NAME_TO_ACTION[args.action]
    # Pista larga y sin obstaculos para medir avance limpio.
    config = RaceConfig(track_length=8.0, track_width=3.0,
                        start_pos=(0.0, 0.0), goal_pos=(8.0, 0.0),
                        obstacles=[], max_steps=args.steps + 5)

    env, source = make_env(config=config, debug=args.debug)
    print(f"[test_walk] fuente={source}  accion='{args.action}'  pasos={args.steps}")
    if args.debug:
        print("  *** MODO DEPURACION (sin Webots): esto NO valida la marcha real ***")

    obs, info = env.reset(seed=0)
    x0 = info.get("robot_x", 0.0)
    y0 = info.get("robot_y", 0.0)

    fell = False
    last_info = info
    rx = ry = None
    for i in range(args.steps):
        obs, r, term, trunc, info = env.step(action)
        last_info = info
        rx = info.get("robot_x", rx)
        ry = info.get("robot_y", ry)
        if info.get("fell"):
            fell = True
            print(f"[test_walk] el robot CAYO en el paso {i+1}.")
            break
        if info.get("reached_goal"):
            print(f"[test_walk] el robot LLEGO a la meta en el paso {i+1}.")
            break
        if term or trunc:
            break

    rx = rx if rx is not None else x0
    ry = ry if ry is not None else y0
    dx = rx - x0
    dy = ry - y0

    print("\n==== RESULTADO DE LA PRUEBA DE MARCHA ====")
    print(f"  posicion inicial: x={x0:.3f}  y={y0:.3f}")
    print(f"  posicion final:   x={rx:.3f}  y={ry:.3f}")
    print(f"  distancia avanzada (X): {dx:.3f} m")
    print(f"  desvio lateral (Y):     {dy:.3f} m")
    print(f"  se cayo: {'SI' if fell else 'no'}")
    if not args.debug:
        if dx > 0.5 and not fell:
            print("  >>> La marcha SIRVE: avanza sin caerse. Puedes entrenar. <<<")
        elif fell:
            print("  >>> La marcha NO sirve todavia: se cayo. Revisa el CPG. <<<")
        else:
            print("  >>> La marcha avanza poco (<0.5 m). Puede necesitar ajuste. <<<")
    env.close()


if __name__ == "__main__":
    main()
