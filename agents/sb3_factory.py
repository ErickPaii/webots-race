"""Fábrica de agentes base con Stable-Baselines3: PPO, DQN y A2C.

Los tres agentes base del enunciado se implementan con la librería
Stable-Baselines3 (SB3), que es la referencia estándar y auditable para estos
algoritmos. NO son implementación propia: en el informe se declara
explícitamente "librería: Stable-Baselines3" para PPO, DQN y A2C.

Todos operan sobre el mismo entorno Gymnasium (DarwinRaceEnv) con
action_space = Discrete(5) y observation_space = Box(12,), lo que permite una
comparación justa. DQN solo admite acciones discretas, por eso todo el diseño
se mantuvo discreto (decisión justificada en el informe).

Los hiperparámetros por defecto quedan centralizados en DEFAULT_HYPERPARAMS para
documentarlos en la tabla del informe y poder variarlos en los experimentos.
"""

from __future__ import annotations

from typing import Any, Dict

# Hiperparámetros por defecto de cada algoritmo. Se documentan en el informe y
# la rejilla de experimentos puede sobrescribirlos (p. ej. learning_rate, gamma).
DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "PPO": {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy": "MlpPolicy",
    },
    "DQN": {
        "learning_rate": 1e-3,
        "buffer_size": 50_000,
        "learning_starts": 1_000,
        "batch_size": 64,
        "gamma": 0.99,
        "train_freq": 4,
        "target_update_interval": 1_000,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "policy": "MlpPolicy",
    },
    "A2C": {
        "learning_rate": 7e-4,
        "n_steps": 20,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy": "MlpPolicy",
    },
}

ALGO_IS_LIBRARY = {"PPO": True, "DQN": True, "A2C": True}


def make_sb3_agent(algo: str, env, seed: int = 0, **overrides: Any):
    """Construye un modelo SB3 (PPO/DQN/A2C) para el entorno dado.

    Args:
        algo: "PPO", "DQN" o "A2C".
        env: entorno Gymnasium (idealmente ya envuelto en Monitor).
        seed: semilla.
        **overrides: sobrescribe hiperparámetros por defecto (para experimentos).

    Returns:
        Un modelo SB3 con .learn(), .predict(), .save(), .load().
    """
    algo = algo.upper()
    if algo not in DEFAULT_HYPERPARAMS:
        raise ValueError(f"Algoritmo SB3 no soportado: {algo}. Usa PPO/DQN/A2C.")

    try:
        from stable_baselines3 import PPO, DQN, A2C
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Se requiere stable-baselines3. Instala con "
            "`pip install -r requirements.txt`."
        ) from exc

    params = dict(DEFAULT_HYPERPARAMS[algo])
    params.update(overrides)
    policy = params.pop("policy", "MlpPolicy")

    cls = {"PPO": PPO, "DQN": DQN, "A2C": A2C}[algo]
    model = cls(policy, env, seed=seed, verbose=0, **params)
    return model


def get_hyperparams(algo: str, **overrides: Any) -> Dict[str, Any]:
    """Devuelve el diccionario de hiperparámetros efectivo (para el CSV/informe)."""
    params = dict(DEFAULT_HYPERPARAMS[algo.upper()])
    params.update(overrides)
    return params
