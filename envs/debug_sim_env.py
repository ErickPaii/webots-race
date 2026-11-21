"""Entorno de DEPURACIÓN ligero (SIN Webots) — NO para el informe.

======================================================================
  ADVERTENCIA: Este entorno es una SIMULACIÓN CINEMÁTICA DE JUGUETE.
  Sirve ÚNICAMENTE para comprobar que el pipeline (train / evaluate /
  analyze / agente propio / agentes SB3) corre de punta a punta sin
  necesidad de Webots. NO reproduce la física del DARwIn-OP ni la marcha.

  >>> SUS RESULTADOS NO DEBEN USARSE EN EL INFORME. <<<
  Todo CSV generado con este entorno se marca con la columna
  `source="DEBUG_SIM_NO_WEBOTS"` para que sea imposible confundirlo con
  datos reales de Webots.
======================================================================

Modela un punto (el "robot") que se mueve en el plano XY de la pista con las
mismas acciones Discrete(5), observación Box(12,) y función de recompensa que
el entorno real, de modo que la interfaz y el flujo de datos sean idénticos.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.config import (
    RaceConfig,
    RewardConfig,
    SafetyConfig,
    N_ACTIONS,
    ACTION_GAIT,
)
from envs.darwin_race_env import OBS_DIM
from safety.shield import ActionShield
from safety.cost import CostFunction

SOURCE_TAG = "DEBUG_SIM_NO_WEBOTS"


class DebugSimEnv(gym.Env):
    """Simulación cinemática de juguete (SIN Webots). Solo para probar el pipeline."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[RaceConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        safety_config: Optional[SafetyConfig] = None,
        **_ignored: Any,
    ):
        super().__init__()
        self.config = config or RaceConfig()
        self.reward_config = reward_config or RewardConfig()
        self.safety_config = safety_config or SafetyConfig()

        self.action_space = spaces.Discrete(N_ACTIONS)
        high = np.ones(OBS_DIM, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.shield = ActionShield(self.safety_config)
        self.cost_fn = CostFunction(self.safety_config)

        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._prev_dist = 0.0
        self._steps = 0
        self._ep_return = 0.0
        self._reached = False
        self._rng = np.random.default_rng(self.config.seed)

    # ----------------------------------------------------------------- helpers
    def _nearest_obstacle(self):
        best_d, best_a = self.config.track_length, 0.0
        for (ox, oy, orad) in self.config.obstacles:
            d = math.hypot(ox - self._x, oy - self._y) - orad
            if d < best_d:
                best_d = max(0.0, d)
                bearing = math.atan2(oy - self._y, ox - self._x)
                best_a = math.atan2(
                    math.sin(bearing - self._heading), math.cos(bearing - self._heading)
                )
        return best_d, best_a

    def _obs(self):
        gx, gy = self.config.goal_pos
        dx, dy = gx - self._x, gy - self._y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        he = math.atan2(math.sin(bearing - self._heading), math.cos(bearing - self._heading))
        nd, na = self._nearest_obstacle()
        L = max(self.config.track_length, 1e-6)
        W = max(self.config.track_width, 1e-6)

        def clip(v):
            return float(max(-1.0, min(1.0, v)))

        return np.array([
            clip(dx / L), clip(dy / W), clip(dist / L), clip(he / math.pi),
            clip(nd / L), clip(na / math.pi),
            0.0, 0.0, 0.0, 0.0, 0.0, clip(1.0),  # roll/pitch/yawrate/accel (sin física)
        ], dtype=np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.config.seed = seed
            self._rng = np.random.default_rng(seed)
        self._x, self._y = self.config.start_pos
        self._heading = 0.0
        gx, gy = self.config.goal_pos
        self._prev_dist = math.hypot(gx - self._x, gy - self._y)
        self._steps = 0
        self._ep_return = 0.0
        self._reached = False
        self.shield.reset()
        self.cost_fn.reset()
        return self._obs(), {"source": SOURCE_TAG}

    def step(self, action: int):
        action = int(action)
        self._steps += 1
        nd, na = self._nearest_obstacle()
        safe_action = self.shield.filter(action, nd, na)

        x_amp, a_amp = ACTION_GAIT.get(safe_action, (0.0, 0.0))
        # Cinemática de juguete: avanza una fracción por x_amp, gira por a_amp.
        self._heading += 0.3 * a_amp
        speed = 0.15 * x_amp
        self._x += speed * math.cos(self._heading)
        self._y += speed * math.sin(self._heading)
        # Ruido pequeño para que las semillas difieran.
        self._y += float(self._rng.normal(0, 0.005))

        gx, gy = self.config.goal_pos
        cur_dist = math.hypot(gx - self._x, gy - self._y)
        nd2, _ = self._nearest_obstacle()

        reached = cur_dist <= self.config.goal_radius
        out = (abs(self._y) > self.config.track_width / 2
               or self._x < -0.5 or self._x > self.config.track_length + 0.5)
        collided = nd2 < 0.12
        fell = False  # sin caídas en la sim de juguete

        rc = self.reward_config
        reward = rc.w_progress * (self._prev_dist - cur_dist) - rc.pen_time
        if collided:
            reward -= rc.pen_collision
        if out:
            reward -= rc.pen_out_of_track
        if reached:
            reward += rc.bonus_goal
        self._prev_dist = cur_dist
        self._ep_return += reward

        self.cost_fn.step_cost(collided=collided, fell=fell, out_of_track=out)

        terminated = reached or out
        truncated = self._steps >= self.config.max_steps
        if reached:
            self._reached = True

        info = {
            "source": SOURCE_TAG,
            "reached_goal": reached,
            "collided": collided,
            "fell": fell,
            "out_of_track": out,
            "nearest_dist": nd2,
            "nearest_angle": na,
            "episode_steps": self._steps,
            "cumulative_reward": self._ep_return,
            "shield_interventions": self.shield.stats.interventions,
            "falls": self.cost_fn.stats.falls,
            "collisions": self.cost_fn.stats.collisions,
            "out_of_track_events": self.cost_fn.stats.out_of_track,
            "cumulative_cost": self.cost_fn.stats.cumulative_cost,
            "action_requested": action,
            "action_executed": safe_action,
        }
        if terminated or truncated:
            info["is_success"] = self._reached
            info["time_to_goal"] = self._steps if self._reached else None
        return self._obs(), reward, terminated, truncated, info

    def close(self):
        pass
