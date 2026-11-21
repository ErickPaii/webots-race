"""Rejilla de experimentos (>= 20 configuraciones).

El enunciado exige >= 20 experimentos variando: tamaño del espacio, posición
inicial, posición de la meta y parámetros del algoritmo. Aquí definimos el
conjunto de CONFIGURACIONES DE ENTORNO (escenarios). Cada script de experimento
las combina con los 4 algoritmos y >= 3 semillas.

Cada experimento tiene un `exp_id` estable para el CSV y el informe.
"""

from __future__ import annotations

from typing import List, Dict, Any

from envs.config import RaceConfig


def _obstacles_line(x0: float, x1: float, n: int, y_amp: float) -> list:
    """n obstáculos repartidos entre x0 y x1, alternando lados en Y."""
    obs = []
    if n <= 0:
        return obs
    for i in range(n):
        x = x0 + (x1 - x0) * (i + 1) / (n + 1)
        y = y_amp if i % 2 == 0 else -y_amp
        obs.append((round(x, 2), y, 0.12))
    return obs


# ---------------------------------------------------------------------------
# 24 escenarios base variando tamaño, salida, meta y densidad de obstáculos.
# (24 escenarios >= 20 requeridos; con 4 algoritmos y 3 semillas da 288 corridas
#  en modo full, o un subconjunto en modo fast.)
# ---------------------------------------------------------------------------
def build_scenarios() -> List[Dict[str, Any]]:
    scenarios: List[Dict[str, Any]] = []

    def add(exp_id, desc, **cfg_kwargs):
        scenarios.append({
            "exp_id": exp_id,
            "description": desc,
            "config": RaceConfig(**cfg_kwargs),
        })

    # --- Grupo A: tamaño de pista (longitud) ---
    add("A1", "Pista corta sin obstáculos", track_length=3.0, track_width=2.0,
        start_pos=(0.0, 0.0), goal_pos=(3.0, 0.0), obstacles=[])
    add("A2", "Pista media sin obstáculos", track_length=6.0, track_width=2.0,
        start_pos=(0.0, 0.0), goal_pos=(6.0, 0.0), obstacles=[])
    add("A3", "Pista larga sin obstáculos", track_length=9.0, track_width=2.0,
        start_pos=(0.0, 0.0), goal_pos=(9.0, 0.0), obstacles=[], max_steps=700)

    # --- Grupo B: ancho de pista (tramos estrechos) ---
    add("B1", "Pista estrecha", track_length=6.0, track_width=1.0,
        start_pos=(0.0, 0.0), goal_pos=(6.0, 0.0), obstacles=[])
    add("B2", "Pista ancha", track_length=6.0, track_width=3.0,
        start_pos=(0.0, 0.0), goal_pos=(6.0, 0.0), obstacles=[])

    # --- Grupo C: posición inicial ---
    add("C1", "Salida desplazada +Y", track_length=6.0, track_width=2.0,
        start_pos=(0.0, 0.6), goal_pos=(6.0, 0.0), obstacles=[])
    add("C2", "Salida desplazada -Y", track_length=6.0, track_width=2.0,
        start_pos=(0.0, -0.6), goal_pos=(6.0, 0.0), obstacles=[])
    add("C3", "Salida adelantada", track_length=6.0, track_width=2.0,
        start_pos=(1.0, 0.0), goal_pos=(6.0, 0.0), obstacles=[])

    # --- Grupo D: posición de la meta ---
    add("D1", "Meta desplazada +Y", track_length=6.0, track_width=2.0,
        start_pos=(0.0, 0.0), goal_pos=(6.0, 0.7), obstacles=[])
    add("D2", "Meta desplazada -Y", track_length=6.0, track_width=2.0,
        start_pos=(0.0, 0.0), goal_pos=(6.0, -0.7), obstacles=[])
    add("D3", "Meta cercana", track_length=6.0, track_width=2.0,
        start_pos=(0.0, 0.0), goal_pos=(4.0, 0.0), obstacles=[])

    # --- Grupo E: densidad de obstáculos ---
    add("E1", "1 obstáculo central", track_length=6.0, track_width=2.0,
        start_pos=(0.0, 0.0), goal_pos=(6.0, 0.0), obstacles=[(3.0, 0.0, 0.12)])
    add("E2", "2 obstáculos", track_length=6.0, track_width=2.0,
        start_pos=(0.0, 0.0), goal_pos=(6.0, 0.0), obstacles=_obstacles_line(1.5, 4.5, 2, 0.4))
    add("E3", "3 obstáculos", track_length=6.0, track_width=2.0,
        start_pos=(0.0, 0.0), goal_pos=(6.0, 0.0), obstacles=_obstacles_line(1.5, 4.5, 3, 0.4))
    add("E4", "4 obstáculos (denso)", track_length=6.0, track_width=2.0,
        start_pos=(0.0, 0.0), goal_pos=(6.0, 0.0), obstacles=_obstacles_line(1.0, 5.0, 4, 0.45))
    add("E5", "6 obstáculos (muy denso)", track_length=8.0, track_width=2.5,
        start_pos=(0.0, 0.0), goal_pos=(8.0, 0.0), obstacles=_obstacles_line(1.0, 7.0, 6, 0.5),
        max_steps=700)

    # --- Grupo F: combinaciones (tamaño + obstáculos + desplazamientos) ---
    add("F1", "Larga + estrecha + 3 obstáculos", track_length=9.0, track_width=1.2,
        start_pos=(0.0, 0.3), goal_pos=(9.0, -0.3), obstacles=_obstacles_line(2.0, 7.0, 3, 0.35),
        max_steps=800)
    add("F2", "Media + 2 obstáculos + meta +Y", track_length=6.0, track_width=2.0,
        start_pos=(0.0, -0.4), goal_pos=(6.0, 0.6), obstacles=_obstacles_line(2.0, 4.0, 2, 0.4))
    add("F3", "Corta + 1 obstáculo + estrecha", track_length=4.0, track_width=1.0,
        start_pos=(0.0, 0.0), goal_pos=(4.0, 0.0), obstacles=[(2.0, 0.0, 0.12)])
    add("F4", "Larga ancha + 4 obstáculos", track_length=9.0, track_width=3.0,
        start_pos=(0.0, 0.5), goal_pos=(9.0, -0.5), obstacles=_obstacles_line(2.0, 7.0, 4, 0.7),
        max_steps=800)

    # --- Grupo G: curvas implícitas (meta muy lateral obliga a girar) ---
    add("G1", "Giro fuerte a +Y", track_length=6.0, track_width=3.0,
        start_pos=(0.0, -1.0), goal_pos=(6.0, 1.0), obstacles=[(3.0, 0.0, 0.12)])
    add("G2", "Giro fuerte a -Y", track_length=6.0, track_width=3.0,
        start_pos=(0.0, 1.0), goal_pos=(6.0, -1.0), obstacles=[(3.0, 0.0, 0.12)])
    add("G3", "Zigzag 3 obstáculos alternos", track_length=7.0, track_width=2.5,
        start_pos=(0.0, 0.0), goal_pos=(7.0, 0.0), obstacles=_obstacles_line(1.5, 5.5, 3, 0.7),
        max_steps=700)

    return scenarios


def fast_subset() -> List[Dict[str, Any]]:
    """Subconjunto reducido para el modo fast (validación del pipeline)."""
    wanted = {"A1", "A2", "E1", "E4"}
    return [s for s in build_scenarios() if s["exp_id"] in wanted]


if __name__ == "__main__":
    sc = build_scenarios()
    print(f"Total de escenarios: {len(sc)}")
    for s in sc:
        print(f"  {s['exp_id']:<4} {s['description']:<40} {s['config'].config_id()}")
