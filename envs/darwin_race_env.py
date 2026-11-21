"""Entorno Gymnasium para la carrera del DARwIn-OP en Webots.

Este wrapper expone la interfaz estándar de Gymnasium (reset, step,
observation_space, action_space) y delega la física al Supervisor de Webots a
través de un socket TCP (ver envs/protocol.py). El Supervisor lanza/posiciona
el robot, avanza la simulación y devuelve observación + reward + flags.

Diseño (aprobado en la Fase 0):

  observation_space: Box(12,) normalizado ~[-1, 1]
      [ dx_meta, dz_meta, dist_meta, heading_error,
        dist_obstaculo, angulo_obstaculo,
        roll, pitch, yaw_rate, accel_x, accel_y, accel_z ]

  action_space: Discrete(5)
      0=stop, 1=forward, 2=forward_fast, 3=turn_left, 4=turn_right
      (cada acción se traduce a parámetros del gait manager del DARwIn-OP;
       así no entrenamos el equilibrio desde cero).

  reward: progreso hacia la meta + bonus por llegar
          - penalización por colisión / caída / fuera de pista / tiempo.

  safe learning: el shield filtra la acción (anti-colisión) y la CostFunction
          acumula el costo del CMDP y las métricas de seguridad por separado.

El entorno es tolerante a la ausencia de Webots: si no logra conectar con el
Supervisor lanza una excepción clara indicando cómo arrancar Webots. Esto
permite construir/validar la lógica sin el simulador, pero NO fabrica datos:
nunca devuelve observaciones falsas.
"""

from __future__ import annotations

import socket
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "Se requiere gymnasium. Instala las dependencias con "
        "`pip install -r requirements.txt`."
    ) from exc

from envs.config import (
    RaceConfig,
    RewardConfig,
    SafetyConfig,
    DEFAULT_HOST,
    DEFAULT_PORT,
    N_ACTIONS,
)
from envs.protocol import send_msg, LineReader
from safety.shield import ActionShield
from safety.cost import CostFunction


OBS_DIM = 12


class DarwinRaceEnv(gym.Env):
    """Entorno de carrera de un solo DARwIn-OP contra obstáculos y el reloj.

    La competencia simultánea de 4 robots se maneja aparte en run_race.py
    (evaluación, sin aprendizaje); para entrenar, cada agente usa este entorno
    de un robot en pistas idénticas, lo que mantiene el RL estacionario.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[RaceConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        safety_config: Optional[SafetyConfig] = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        connect_timeout: float = 30.0,
    ):
        super().__init__()
        self.config = config or RaceConfig()
        self.reward_config = reward_config or RewardConfig()
        self.safety_config = safety_config or SafetyConfig()
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout

        # Espacios Gymnasium.
        self.action_space = spaces.Discrete(N_ACTIONS)
        high = np.ones(OBS_DIM, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Safe learning.
        self.shield = ActionShield(self.safety_config)
        self.cost_fn = CostFunction(self.safety_config)

        # Estado de conexión.
        self._sock: Optional[socket.socket] = None
        self._reader: Optional[LineReader] = None

        # Acumuladores por episodio (para info / CSV).
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_start_time = 0.0
        self._reached_goal = False

    # ------------------------------------------------------------------ conexión
    def _ensure_connection(self) -> None:
        if self._sock is not None:
            return
        deadline = time.time() + self.connect_timeout
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                s = socket.create_connection((self.host, self.port), timeout=5.0)
                self._sock = s
                self._reader = LineReader(s)
                return
            except OSError as e:
                last_err = e
                time.sleep(0.5)
        raise ConnectionError(
            f"No se pudo conectar al Supervisor de Webots en {self.host}:{self.port}.\n"
            f"Asegúrate de haber abierto el mundo worlds/darwin_race.wbt en Webots "
            f"(el controlador Supervisor abre el servidor TCP). Último error: {last_err}"
        )

    # ------------------------------------------------------------------ gym API
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.config.seed = seed
        self._ensure_connection()

        self.shield.reset()
        self.cost_fn.reset()
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._reached_goal = False
        self._ep_start_time = time.time()

        send_msg(self._sock, {"cmd": "reset", "config": self.config.to_dict()})
        resp = self._reader.read_msg(timeout=60.0)
        if resp is None:
            raise ConnectionError("El Supervisor cerró la conexión durante reset().")
        obs = np.asarray(resp["obs"], dtype=np.float32)
        info = resp.get("info", {})
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._sock is None:
            raise RuntimeError("Llama a reset() antes de step().")

        action = int(action)
        self._ep_steps += 1

        # --- Safe learning: filtrar la acción con el shield, usando la última
        #     observación de distancia/ángulo al obstáculo más cercano. El
        #     Supervisor nos reenvía esos valores en el info del paso previo;
        #     en el primer paso usamos la observación inicial almacenada.
        nearest_dist = getattr(self, "_last_nearest_dist", 1.0)
        nearest_angle = getattr(self, "_last_nearest_angle", 0.0)
        safe_action = self.shield.filter(action, nearest_dist, nearest_angle)

        send_msg(self._sock, {"cmd": "step", "action": safe_action})
        resp = self._reader.read_msg(timeout=60.0)
        if resp is None:
            raise ConnectionError("El Supervisor cerró la conexión durante step().")

        obs = np.asarray(resp["obs"], dtype=np.float32)
        base_info = resp.get("info", {})

        # Guardamos distancia/ángulo para el shield del siguiente paso.
        self._last_nearest_dist = float(base_info.get("nearest_dist", 1.0))
        self._last_nearest_angle = float(base_info.get("nearest_angle", 0.0))

        # El Supervisor ya entrega el reward shaping calculado (progreso, bonus,
        # penalizaciones). Aquí solo acumulamos y adjuntamos métricas de seguridad.
        reward = float(resp["reward"])
        terminated = bool(resp["terminated"])
        truncated = bool(resp["truncated"]) or self._ep_steps >= self.config.max_steps

        # Costo separado (CMDP) + métricas de seguridad.
        self.cost_fn.step_cost(
            collided=bool(base_info.get("collided", False)),
            fell=bool(base_info.get("fell", False)),
            out_of_track=bool(base_info.get("out_of_track", False)),
        )

        self._ep_reward += reward
        if base_info.get("reached_goal", False):
            self._reached_goal = True

        info = dict(base_info)
        info.update(
            {
                "episode_steps": self._ep_steps,
                "cumulative_reward": self._ep_reward,
                "shield_interventions": self.shield.stats.interventions,
                "falls": self.cost_fn.stats.falls,
                "collisions": self.cost_fn.stats.collisions,
                "out_of_track_events": self.cost_fn.stats.out_of_track,
                "cumulative_cost": self.cost_fn.stats.cumulative_cost,
                "action_requested": action,
                "action_executed": safe_action,
            }
        )
        if terminated or truncated:
            info["is_success"] = self._reached_goal
            info["time_to_goal"] = (
                time.time() - self._ep_start_time if self._reached_goal else None
            )
            # Gymnasium Monitor espera la clave "episode" cuando se usa RecordEpisodeStatistics;
            # SB3 la añade con su propio Monitor, así que no la duplicamos aquí.

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._sock is not None:
            try:
                send_msg(self._sock, {"cmd": "close"})
            except OSError:
                pass
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._reader = None
