"""Función de costo separada del Constrained-MDP (safe learning).

En un CMDP el agente maximiza la recompensa sujeto a mantener el costo esperado
bajo un umbral. Aquí mantenemos el costo SEPARADO de la recompensa para poder
reportarlo como métrica de seguridad independiente (nº de caídas, colisiones,
violaciones), tal como pide el enunciado.

El costo NO modifica la recompensa que ve el algoritmo de RL base (PPO/DQN/A2C
de Stable-Baselines3 optimizan solo el reward). El costo se acumula y se guarda
en el CSV para el análisis de seguridad. El agente propio, al ser implementación
propia, SÍ puede usar este costo (p. ej. para una variante lagrangiana) del
algoritmo del Taller 2.
"""

from __future__ import annotations

from dataclasses import dataclass

from envs.config import SafetyConfig


@dataclass
class CostStats:
    """Métricas de seguridad acumuladas por episodio."""
    falls: int = 0
    collisions: int = 0
    out_of_track: int = 0
    cumulative_cost: float = 0.0

    def reset(self) -> None:
        self.falls = 0
        self.collisions = 0
        self.out_of_track = 0
        self.cumulative_cost = 0.0


class CostFunction:
    """Calcula el costo c_t de un paso según los eventos de seguridad."""

    def __init__(self, cfg: SafetyConfig):
        self.cfg = cfg
        self.stats = CostStats()

    def reset(self) -> None:
        self.stats.reset()

    def step_cost(self, *, collided: bool, fell: bool, out_of_track: bool) -> float:
        """Acumula eventos y devuelve el costo del paso."""
        c = 0.0
        if collided:
            self.stats.collisions += 1
            c += self.cfg.cost_collision
        if fell:
            self.stats.falls += 1
            c += self.cfg.cost_fall
        if out_of_track:
            self.stats.out_of_track += 1
            c += self.cfg.cost_out_of_track
        self.stats.cumulative_cost += c
        return c
