"""Agentes: base (SB3: PPO/DQN/A2C) y propio (A2C + population discovery)."""

from agents.sb3_factory import make_sb3_agent, get_hyperparams, DEFAULT_HYPERPARAMS, ALGO_IS_LIBRARY
from agents.custom_agent import CustomAgent, CustomHyperparams

ALL_ALGOS = ["CUSTOM", "PPO", "DQN", "A2C"]

__all__ = [
    "make_sb3_agent",
    "get_hyperparams",
    "DEFAULT_HYPERPARAMS",
    "ALGO_IS_LIBRARY",
    "CustomAgent",
    "CustomHyperparams",
    "ALL_ALGOS",
]
