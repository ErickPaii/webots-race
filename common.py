"""Utilidades compartidas por los scripts (factoría de entorno y CSV).

Centraliza:
  - la creación del entorno (real de Webots o el de DEPURACIÓN sin Webots),
  - la escritura de CSV con columnas estables,
  - el registro del origen de los datos (real vs. depuración) para que el
    informe nunca mezcle resultados de juguete con resultados reales.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

from envs.config import RaceConfig, RewardConfig, SafetyConfig, DEFAULT_PORT

# Etiqueta de origen de datos.
SOURCE_WEBOTS = "WEBOTS_REAL"
SOURCE_DEBUG = "DEBUG_SIM_NO_WEBOTS"


def make_env(
    config: Optional[RaceConfig] = None,
    debug: bool = False,
    reward_config: Optional[RewardConfig] = None,
    safety_config: Optional[SafetyConfig] = None,
    port: int = DEFAULT_PORT,
):
    """Crea el entorno. Si debug=True usa la sim de juguete SIN Webots.

    Devuelve (env, source_tag).
    """
    reward_config = reward_config or RewardConfig()
    safety_config = safety_config or SafetyConfig()
    if debug:
        from envs.debug_sim_env import DebugSimEnv
        env = DebugSimEnv(config=config, reward_config=reward_config,
                          safety_config=safety_config)
        return env, SOURCE_DEBUG
    from envs.darwin_race_env import DarwinRaceEnv
    env = DarwinRaceEnv(config=config, reward_config=reward_config,
                        safety_config=safety_config, port=port)
    return env, SOURCE_WEBOTS


# Columnas del CSV de EPISODIOS de entrenamiento (para curvas de aprendizaje).
TRAIN_COLUMNS = [
    "source", "algorithm", "exp_id", "config_id", "seed", "phase",
    "episode", "step", "ep_return", "ep_len", "is_success",
    "falls", "collisions", "time_to_goal",
]

# Columnas del CSV de EVALUACIÓN / carrera (resumen por corrida).
EVAL_COLUMNS = [
    "source", "algorithm", "exp_id", "config_id", "seed", "phase",
    "episode", "cumulative_reward", "is_success", "time_to_goal",
    "falls", "collisions", "out_of_track_events", "shield_interventions",
]


class CSVLogger:
    """Escribe filas a un CSV, creando la cabecera si no existe."""

    def __init__(self, path: str, columns: List[str]):
        self.path = path
        self.columns = columns
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._new = not os.path.exists(path) or os.path.getsize(path) == 0

    def write(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            if self._new:
                w.writeheader()
                self._new = False
            w.writerow({c: row.get(c, "") for c in self.columns})
