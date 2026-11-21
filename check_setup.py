"""check_setup.py — Verifica que tu máquina está lista para correr el proyecto.

Comprueba, con mensajes claros de qué falló:
  1. Versión de Python (>= 3.10).
  2. Imports: numpy, gymnasium, torch, stable_baselines3.
  3. Que el agente propio y los agentes SB3 se construyen sobre el entorno.
  4. Un episodio corto de prueba en el entorno de DEPURACIÓN (sin Webots).
  5. (Opcional) Que el puerto TCP del supervisor de Webots responde, si pasas
     --webots (requiere tener el mundo abierto en Webots).

Uso:
    python check_setup.py            # comprobaciones sin Webots
    python check_setup.py --webots   # además prueba la conexión al supervisor

Códigos de salida: 0 = todo OK; 1 = alguna comprobación obligatoria falló.
"""

from __future__ import annotations

import argparse
import importlib
import socket
import sys

OK = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check_python():
    v = sys.version_info
    if v < (3, 10):
        print(f"{FAIL} Python {v.major}.{v.minor}: se requiere >= 3.10 "
              f"(Stable-Baselines3 2.x lo exige).")
        return False
    print(f"{OK} Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_imports():
    ok = True
    wanted = {
        "numpy": "numpy",
        "gymnasium": "gymnasium",
        "torch": "torch (PyTorch, CPU)",
        "stable_baselines3": "stable-baselines3",
    }
    for mod, desc in wanted.items():
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{OK} import {desc}  (version {ver})")
        except Exception as e:
            ok = False
            print(f"{FAIL} no se pudo importar {desc}: {e}")
            if mod == "torch":
                print("       instala: pip install torch --index-url "
                      "https://download.pytorch.org/whl/cpu")
            elif mod == "stable_baselines3":
                print('       instala: pip install "stable-baselines3>=2.3,<3.0"')
            elif mod == "gymnasium":
                print('       instala: pip install "gymnasium>=0.29,<1.0"')
            else:
                print("       instala: pip install numpy")
    return ok


def check_agents_build():
    """Construye los 4 agentes sobre el entorno de depuración (sin entrenar)."""
    try:
        from envs.config import RaceConfig
        from envs.debug_sim_env import DebugSimEnv
        env = DebugSimEnv(config=RaceConfig())
    except Exception as e:
        print(f"{FAIL} no se pudo crear el entorno de prueba: {e}")
        return False

    ok = True
    # SB3
    try:
        from agents.sb3_factory import make_sb3_agent
        for algo in ["PPO", "DQN", "A2C"]:
            make_sb3_agent(algo, env, seed=0)
        print(f"{OK} agentes SB3 (PPO/DQN/A2C) se construyen sobre el entorno")
    except Exception as e:
        ok = False
        print(f"{FAIL} no se pudieron construir los agentes SB3: {e}")
    # Custom
    try:
        from agents.custom_agent import CustomAgent
        CustomAgent(env.observation_space, env.action_space, seed=0, variant="base")
        print(f"{OK} agente propio (A2C + population discovery) se construye")
    except Exception as e:
        ok = False
        print(f"{FAIL} no se pudo construir el agente propio: {e}")
    env.close()
    return ok


def check_short_episode():
    """Corre un episodio corto en el entorno de depuración (sin Webots)."""
    try:
        from envs.config import RaceConfig
        from envs.debug_sim_env import DebugSimEnv
        env = DebugSimEnv(config=RaceConfig(max_steps=50))
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        steps = 0
        done = False
        while not done and steps < 60:
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            steps += 1
            done = term or trunc
        env.close()
        print(f"{OK} episodio corto de prueba (DEBUG, sin Webots): {steps} pasos, "
              f"source={info.get('source')}")
        print("       (recordatorio: los datos DEBUG NO son validos para el informe)")
        return True
    except Exception as e:
        print(f"{FAIL} el episodio de prueba fallo: {e}")
        return False


def check_webots_port(host="127.0.0.1", port=10001, timeout=3.0):
    """Comprueba que el supervisor de Webots está escuchando en el puerto TCP."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            print(f"{OK} el supervisor de Webots responde en {host}:{port}")
            return True
    except OSError as e:
        print(f"{FAIL} no hay respuesta en {host}:{port}: {e}")
        print("       Abre worlds/darwin_race.wbt en Webots y pulsa Play. El "
              "controlador supervisor_controller abre ese puerto al arrancar.")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--webots", action="store_true",
                   help="ademas prueba la conexion TCP al supervisor de Webots")
    args = p.parse_args()

    print("== check_setup: verificando entorno ==")
    results = []
    results.append(("Python >= 3.10", check_python()))
    imports_ok = check_imports()
    results.append(("imports", imports_ok))
    if imports_ok:
        results.append(("construccion de agentes", check_agents_build()))
        results.append(("episodio de prueba (DEBUG)", check_short_episode()))
    else:
        print(f"{WARN} se omiten pruebas de agentes/episodio porque faltan imports.")
    if args.webots:
        results.append(("puerto Webots", check_webots_port()))

    print("\n== resumen ==")
    all_ok = True
    for name, ok in results:
        print(f"  {OK if ok else FAIL} {name}")
        all_ok = all_ok and ok
    if all_ok:
        print("\nTodo listo. Puedes entrenar (ver README).")
        sys.exit(0)
    else:
        print("\nHay comprobaciones que fallaron. Revisa los mensajes de arriba.")
        sys.exit(1)


if __name__ == "__main__":
    main()
