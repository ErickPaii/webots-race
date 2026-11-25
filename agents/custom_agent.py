"""Agente propio (algoritmo del Taller 2): Actor-Critic + Population Discovery.

ALGORITMO (adaptado del Taller 2 sobre Atari):
    Actor-Critic (A2C a n pasos) con bonus de entropía, AUMENTADO con
    "descubrimiento poblacional": cada `disc_every` pasos se crea una población
    de copias mutadas del Actor (pares antitéticos +eps / -eps, estilo Evolution
    Strategies), cada variante JUEGA CARRERAS REALES, y el Actor se mueve hacia
    las mejores según el RETORNO REAL del episodio (no según el Critic).
    Variante "control": mueve el Actor en dirección aleatoria de igual magnitud.

    Actualización ES:
        theta <- theta + es_step * sigma * sum_k w_k * eps_k ,  w_k = rango en [-0.5, 0.5]

Es IMPLEMENTACIÓN PROPIA (no se usa la A2C de Stable-Baselines3). Todos los
hiperparámetros están en CustomHyperparams y se documentan en el informe.

Adaptación al problema de carrera:
  - Observación: Box(12,) del DarwinRaceEnv (no RAM de 128 bytes).
  - Acciones: Discrete(5) (gait manager).
  - "Score real" de una variante = retorno real del episodio de carrera.
  - Redes: MLP (2 capas ocultas de 128, ReLU) para Actor y Critic.

Interfaz pública estable: select_action / update / discover / learn / save / load.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class CustomHyperparams:
    """Hiperparámetros del agente propio (se documentan en el informe)."""
    # --- Actor-Critic ---
    gamma: float = 0.99
    lr: float = 7e-4
    n_steps: int = 20
    beta: float = 0.01          # bonus de entropía
    hidden: int = 128
    max_grad_norm: float = 0.5
    # --- Descubrimiento poblacional (Evolution Strategies) ---
    variant: str = "population"  # "base" | "population" | "control"
    pop: int = 6                 # variantes por ronda (pares antitéticos)
    sigma: float = 0.01          # tamaño de mutación por peso
    pop_episodes: int = 2        # carreras reales por variante (fitness=media)
    disc_every: int = 40_000     # pasos entre rondas de descubrimiento
    es_step: float = 1.0         # cuánto se mueve el Actor hacia las mejores

    def to_dict(self) -> dict:
        return asdict(self)


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.distributions import Categorical
        return torch, nn, Categorical
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "El agente propio requiere PyTorch. Instala con "
            "`pip install torch --index-url https://download.pytorch.org/whl/cpu`."
        ) from exc


def _mlp(nn, n_in: int, n_out: int, hidden: int):
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, n_out),
    )


class CustomAgent:
    """Agente propio: A2C n-step + descubrimiento poblacional."""

    def __init__(self, observation_space, action_space, seed: int = 0, **hyperparams: Any):
        self.torch, self.nn, self.Categorical = _import_torch()
        torch = self.torch

        self.observation_space = observation_space
        self.action_space = action_space
        self.seed = seed
        self.hp = CustomHyperparams(**hyperparams)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        self.obs_dim = int(np.prod(observation_space.shape))
        self.n_actions = int(action_space.n)

        self.actor = _mlp(self.nn, self.obs_dim, self.n_actions, self.hp.hidden)
        self.critic = _mlp(self.nn, self.obs_dim, 1, self.hp.hidden)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.hp.lr
        )

    # ----------------------------------------------------------------- helpers
    def _prep(self, obs: np.ndarray):
        return self.torch.as_tensor(np.asarray(obs, dtype=np.float32))

    # --------------------------------------------------------------- inferencia
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        torch = self.torch
        with torch.no_grad():
            logits = self.actor(self._prep(obs))
            if deterministic:
                return int(torch.argmax(logits).item())
            return int(self.Categorical(logits=logits).sample().item())

    # ------------------------------------------------------------ entrenamiento
    def update(self, rollout: Dict[str, list]) -> Dict[str, float]:
        """Actualización Actor-Critic a n pasos.

        rollout: dict con listas 'logps','ents','vals','rews','dones' (largo
        n_steps) y 'last_obs' (observación tras el último paso) para bootstrap.
        """
        torch = self.torch
        with torch.no_grad():
            R = self.critic(self._prep(rollout["last_obs"])).squeeze()
        returns = []
        for t in reversed(range(len(rollout["rews"]))):
            R = rollout["rews"][t] + self.hp.gamma * R * (1.0 - rollout["dones"][t])
            returns.append(R)
        returns = torch.stack(returns[::-1])
        vals = torch.stack(rollout["vals"])
        adv = returns - vals

        logps = torch.stack(rollout["logps"])
        ents = torch.stack(rollout["ents"])
        actor_loss = -(logps * adv.detach()).mean() - self.hp.beta * ents.mean()
        critic_loss = 0.5 * adv.pow(2).mean()
        loss = actor_loss + critic_loss

        self.opt.zero_grad()
        loss.backward()
        self.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.hp.max_grad_norm,
        )
        self.opt.step()
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(ents.mean().item()),
        }

    def discover(self, env) -> Dict[str, Any]:
        """Una ronda de descubrimiento poblacional (Evolution Strategies).

        Crea pop variantes antitéticas del Actor, cada una corre pop_episodes
        carreras reales (su fitness = retorno medio real), y mueve el Actor
        hacia las mejores por rango (o en dirección aleatoria si variant=='control').
        Devuelve fitnesses y pasos consumidos (cuentan al presupuesto).
        """
        torch = self.torch
        from torch.nn.utils import parameters_to_vector, vector_to_parameters

        theta = parameters_to_vector(self.actor.parameters()).detach().clone()
        cand = _mlp(self.nn, self.obs_dim, self.n_actions, self.hp.hidden)
        half = torch.randn(self.hp.pop // 2, theta.numel())
        eps = torch.cat([half, -half])  # pares antitéticos (pop, n_weights)

        fits: List[float] = []
        used = 0
        for k in range(self.hp.pop):
            vector_to_parameters(theta + self.hp.sigma * eps[k], cand.parameters())
            scores, u = self._rollout_real(cand, env, self.hp.pop_episodes)
            fits.append(float(np.mean(scores)))
            used += u

        ranks = np.argsort(np.argsort(fits)) / max(1, (self.hp.pop - 1)) - 0.5
        if self.hp.variant == "control":
            ranks = self.rng.permutation(ranks)
        delta = self.hp.es_step * self.hp.sigma * (
            torch.as_tensor(ranks, dtype=torch.float32) @ eps
        )
        vector_to_parameters(theta + delta, self.actor.parameters())
        return {
            "fit_mean": float(np.mean(fits)),
            "fit_std": float(np.std(fits)),
            "fit_best": float(np.max(fits)),
            "steps_used": used,
            "fits": fits,
        }

    def _rollout_real(self, actor_net, env, episodes: int):
        """Juega `episodes` carreras REALES con `actor_net` (sin aprender).

        Devuelve (lista de retornos, pasos usados). Muestrea la política (como
        en el Taller 2) para medir el score real de la variante.
        """
        torch = self.torch
        scores, used = [], 0
        for _ in range(episodes):
            obs, _ = env.reset()
            done, total = False, 0.0
            while not done:
                with torch.no_grad():
                    a = self.Categorical(logits=actor_net(self._prep(obs))).sample().item()
                obs, r, term, trunc, _ = env.step(a)
                total += r
                used += 1
                done = term or trunc
            scores.append(total)
        return scores, used

    def learn(
        self,
        env,
        total_timesteps: int,
        callback: Optional[Callable[[dict], None]] = None,
        disc_env=None,
    ) -> "CustomAgent":
        """Bucle de entrenamiento (análogo a SB3.learn()).

        - Juega n_steps decisiones con exploración (muestreo).
        - Llama update() Actor-Critic.
        - Cada disc_every pasos (solo variantes != 'base'), llama discover() en
          un entorno separado `disc_env` (o `env` si no se pasa).
        - Invoca `callback(ep_info)` al terminar cada episodio para registrar la
          curva de aprendizaje en el CSV.
        """
        torch = self.torch
        use_pop = self.hp.variant != "base"
        disc_env = disc_env or env
        next_disc = self.hp.disc_every

        step = 0
        ep_idx = 0
        ep_return = 0.0
        ep_len = 0
        obs, _ = env.reset()

        while step < total_timesteps:
            rollout = {"logps": [], "ents": [], "vals": [], "rews": [], "dones": []}
            for _ in range(self.hp.n_steps):
                s = self._prep(obs)
                dist = self.Categorical(logits=self.actor(s))
                a = dist.sample()
                next_obs, r, term, trunc, info = env.step(int(a.item()))
                done = term or trunc
                rollout["logps"].append(dist.log_prob(a))
                rollout["ents"].append(dist.entropy())
                rollout["vals"].append(self.critic(s).squeeze())
                rollout["rews"].append(float(r))
                rollout["dones"].append(float(done))
                ep_return += float(r)
                ep_len += 1
                step += 1
                obs = next_obs
                if done:
                    ep_idx += 1
                    if callback is not None:
                        callback({
                            "episode": ep_idx,
                            "step": step,
                            "ep_return": ep_return,
                            "ep_len": ep_len,
                            "is_success": bool(info.get("is_success", False)),
                            "falls": int(info.get("falls", 0)),
                            "collisions": int(info.get("collisions", 0)),
                            "time_to_goal": info.get("time_to_goal", None),
                        })
                    ep_return, ep_len = 0.0, 0
                    obs, _ = env.reset()
                if step >= total_timesteps:
                    break

            rollout["last_obs"] = obs
            if len(rollout["rews"]) > 0:
                self.update(rollout)

            if use_pop and step >= next_disc:
                info_disc = self.discover(disc_env)
                step += info_disc["steps_used"]
                next_disc += self.hp.disc_every
                if callback is not None:
                    callback({"discovery": True, "step": step, **{
                        k: info_disc[k] for k in ("fit_mean", "fit_std", "fit_best")
                    }})

        return self

    # ------------------------------------------------------------ persistencia
    def save(self, path: str) -> None:
        self.torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "hyperparams": self.hp.to_dict(),
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, observation_space=None, action_space=None) -> "CustomAgent":
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if observation_space is None or action_space is None:
            raise ValueError(
                "load() necesita observation_space y action_space para reconstruir el agente."
            )
        agent = cls(observation_space, action_space, **ckpt.get("hyperparams", {}))
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        return agent

    def get_hyperparams(self) -> Dict[str, Any]:
        return self.hp.to_dict()
