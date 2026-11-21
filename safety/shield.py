"""Shield de seguridad: filtro de acciones (safe learning).

Implementa una restricción de seguridad EXPLÍCITA sobre la política: antes de
ejecutar la acción elegida por el agente, el shield comprueba si esa acción
llevaría al robot a colisionar con el obstáculo/robot más cercano dentro de un
umbral de distancia. Si es así, sustituye la acción por una alternativa segura
(girar en dirección libre o detenerse) y registra la intervención.

Esto corresponde al componente "shield / filtro de acciones que impida
colisionar" del requisito de safe learning. Es independiente del algoritmo de
RL: se aplica igual a PPO, DQN, A2C y al agente propio, de modo que la
comparación sea justa.

El shield razona con la observación ya calculada por el entorno (distancia y
ángulo al obstáculo más cercano), por lo que no necesita acceso a Webots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from envs.config import (
    SafetyConfig,
    ACTION_STOP,
    ACTION_FORWARD,
    ACTION_FORWARD_FAST,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
)


@dataclass
class ShieldStats:
    """Métricas de seguridad acumuladas durante un episodio."""
    interventions: int = 0      # nº de veces que el shield cambió la acción
    blocked_forward: int = 0    # avances bloqueados por riesgo de colisión

    def reset(self) -> None:
        self.interventions = 0
        self.blocked_forward = 0


class ActionShield:
    """Filtro de acciones basado en distancia al obstáculo más cercano.

    Args:
        cfg: configuración de seguridad (umbral, límites).
    """

    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.stats = ShieldStats()

    def reset(self) -> None:
        self.stats.reset()

    def filter(self, action: int, nearest_dist: float, nearest_angle: float) -> int:
        """Devuelve una acción segura.

        Args:
            action: acción propuesta por el agente (0..4).
            nearest_dist: distancia (m) al obstáculo/robot más cercano.
            nearest_angle: ángulo relativo (rad) al obstáculo más cercano.
                Positivo = a la izquierda del robot, negativo = a la derecha.

        Returns:
            La misma acción si es segura, o una acción segura alternativa.
        """
        if not self.cfg.enable_shield:
            return action

        is_forward = action in (ACTION_FORWARD, ACTION_FORWARD_FAST)
        # Solo intervenimos si avanzamos hacia un obstáculo cercano y casi al frente.
        if is_forward and nearest_dist < self.cfg.shield_min_dist and abs(nearest_angle) < 0.6:
            self.stats.interventions += 1
            self.stats.blocked_forward += 1
            # Gira hacia el lado con más espacio (opuesto al obstáculo).
            if nearest_angle >= 0:   # obstáculo a la izquierda -> gira derecha
                return ACTION_TURN_RIGHT
            else:                    # obstáculo a la derecha -> gira izquierda
                return ACTION_TURN_LEFT
        return action
