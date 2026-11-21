"""Configuración parametrizable del entorno de carreras DARwIn-OP.

Toda la parametrización pedida en los requisitos técnicos vive aquí:
tamaño de la pista, posición inicial, posición de la meta, número y
posición de obstáculos, y semilla. Los scripts (train/evaluate/run_race)
y la rejilla de experimentos construyen instancias de `RaceConfig`.

Las unidades son metros y radianes (coherentes con Webots).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import json


# ---------------------------------------------------------------------------
# Puerto del canal supervisor <-> entorno Gymnasium.
# El Supervisor de Webots abre un servidor TCP en localhost y el proceso
# Python que corre Stable-Baselines3 / el agente propio se conecta como
# cliente. Ver envs/protocol.py para el formato de los mensajes.
# ---------------------------------------------------------------------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 10001


@dataclass
class RaceConfig:
    """Configuración completa de un escenario de carrera.

    Attributes:
        track_length: Largo de la pista (eje X, de salida a meta), en metros.
        track_width: Ancho de la pista (eje Y lateral), en metros. Salir de
            [-w/2, w/2] cuenta como "fuera de pista".
        start_pos: Posición inicial del robot (x, y) planar, en metros.
        goal_pos: Posición de la meta (x, y) planar, en metros.
        goal_radius: Radio alrededor de la meta que cuenta como "llegada".
        obstacles: Lista de obstáculos como (x, y, radio) en metros.

    Nota de ejes: en Webots R2025a el PROTO Darwin-op usa Z como eje vertical,
    por lo que la pista se despliega en el plano XY (Y = lateral). El par
    planar se nombra (x, y) para evitar confusión con la altura Z.
        seed: Semilla para reproducibilidad.
        max_steps: Máximo de pasos de entorno por episodio (corte por tiempo).
        control_step_ms: Milisegundos de simulación avanzados por cada step()
            del entorno. Debe ser múltiplo del basicTimeStep del mundo (32 ms).
        fall_pitch_threshold: Umbral |pitch| (rad) que cuenta como caída.
        fall_roll_threshold: Umbral |roll| (rad) que cuenta como caída.
    """

    track_length: float = 6.0
    track_width: float = 2.0
    start_pos: Tuple[float, float] = (0.0, 0.0)
    goal_pos: Tuple[float, float] = (6.0, 0.0)
    goal_radius: float = 0.4
    obstacles: List[Tuple[float, float, float]] = field(default_factory=list)
    seed: int = 0
    max_steps: int = 500
    control_step_ms: int = 320  # 10 * basicTimeStep(32ms) -> ~0.32 s por step
    fall_pitch_threshold: float = 0.7
    fall_roll_threshold: float = 0.7

    # ----------------------------- serialización ---------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "RaceConfig":
        # las tuplas se serializan como listas en JSON; las reconvertimos
        d = dict(d)
        if "start_pos" in d:
            d["start_pos"] = tuple(d["start_pos"])
        if "goal_pos" in d:
            d["goal_pos"] = tuple(d["goal_pos"])
        if "obstacles" in d:
            d["obstacles"] = [tuple(o) for o in d["obstacles"]]
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "RaceConfig":
        return cls.from_dict(json.loads(s))

    def config_id(self) -> str:
        """Identificador corto y legible de la configuración, para el CSV."""
        return (
            f"L{self.track_length:g}_W{self.track_width:g}"
            f"_G{self.goal_pos[0]:g}-{self.goal_pos[1]:g}"
            f"_S{self.start_pos[0]:g}-{self.start_pos[1]:g}"
            f"_O{len(self.obstacles)}_seed{self.seed}"
        )


# ---------------------------------------------------------------------------
# Parámetros del espacio de acciones (Discrete(5)) traducidos a comandos del
# gait manager del DARwIn-OP. Cada acción fija amplitud de paso en X (avance)
# y ángulo de giro A. Son valores conservadores para no desestabilizar la
# marcha; el shield de seguridad puede reducirlos aún más.
# ---------------------------------------------------------------------------
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_FORWARD_FAST = 2
ACTION_TURN_LEFT = 3
ACTION_TURN_RIGHT = 4

ACTION_NAMES = {
    ACTION_STOP: "stop",
    ACTION_FORWARD: "forward",
    ACTION_FORWARD_FAST: "forward_fast",
    ACTION_TURN_LEFT: "turn_left",
    ACTION_TURN_RIGHT: "turn_right",
}

# (x_amplitude, a_amplitude) para el gait manager. x>0 avanza, a>0 gira izq.
ACTION_GAIT = {
    ACTION_STOP:         (0.0,  0.0),
    ACTION_FORWARD:      (0.6,  0.0),
    ACTION_FORWARD_FAST: (1.0,  0.0),
    ACTION_TURN_LEFT:    (0.3,  0.5),
    ACTION_TURN_RIGHT:   (0.3, -0.5),
}

N_ACTIONS = len(ACTION_NAMES)


# ---------------------------------------------------------------------------
# Pesos de la función de recompensa (reward shaping) y de la función de costo
# separada (enfoque Constrained-MDP). Ver envs/darwin_race_env.py y safety/.
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    w_progress: float = 1.0        # recompensa por metro avanzado hacia la meta
    bonus_goal: float = 100.0      # bonificación por llegar a la meta
    pen_collision: float = 5.0     # penalización por colisión
    pen_fall: float = 50.0         # penalización por caída (termina episodio)
    pen_out_of_track: float = 50.0 # penalización por salir de la pista (termina)
    pen_time: float = 0.1          # penalización por paso (incentiva rapidez)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SafetyConfig:
    """Parámetros del safe learning (shield + límites + costo)."""
    enable_shield: bool = True
    shield_min_dist: float = 0.35      # distancia (m) a la que el shield actúa
    max_forward_amplitude: float = 1.0 # límite de velocidad de marcha (gait X)
    max_turn_amplitude: float = 0.5    # límite de giro
    # costos del Constrained-MDP (separados del reward)
    cost_collision: float = 1.0
    cost_fall: float = 1.0
    cost_out_of_track: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)
