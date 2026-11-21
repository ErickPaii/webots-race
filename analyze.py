"""analyze.py — Agrega los CSV y genera tablas y figuras para el informe.

Lee todos los CSV de results/ y produce:
  - figures/learning_curves_<exp>.png : curvas de aprendizaje (recompensa por
    episodio, media móvil, promediadas entre semillas) por algoritmo.
  - figures/success_rate.png           : barras de tasa de éxito por algoritmo.
  - figures/reward_boxplot.png         : boxplots de recompensa de evaluación.
  - results/summary_metrics.csv        : métricas promedio (± std) por algoritmo.
  - results/race_summary.csv           : resumen de las carreras (orden/tiempos).

IMPORTANTE: separa SIEMPRE los datos por la columna `source`. Los datos de
depuración (source=DEBUG_SIM_NO_WEBOTS) se reportan aparte y NUNCA se mezclan
con los reales (source=WEBOTS_REAL). Si solo hay datos de depuración, lo avisa
en claro y marca las salidas como NO válidas para el informe.

Uso:
    python analyze.py                 # analiza todo results/
    python analyze.py --exp A2        # solo un escenario para las curvas
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
ALGOS = ["CUSTOM", "PPO", "DQN", "A2C"]


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_eval_rows():
    rows = []
    for p in glob.glob(os.path.join(RESULTS_DIR, "eval_*.csv")):
        rows.extend(_read_csv(p))
    return rows


def load_train_files():
    return sorted(glob.glob(os.path.join(RESULTS_DIR, "train_*.csv")))


def load_race_rows():
    rows = []
    for p in glob.glob(os.path.join(RESULTS_DIR, "race_*.csv")):
        rows.extend(_read_csv(p))
    return rows


def _warn_source(sources):
    real = any(s == "WEBOTS_REAL" for s in sources)
    debug = any(s == "DEBUG_SIM_NO_WEBOTS" for s in sources)
    if debug and not real:
        print("  *** AVISO: solo hay datos de DEPURACION (sin Webots). "
              "Estas salidas NO son validas para el informe. ***")
    elif debug and real:
        print("  *** AVISO: hay datos reales Y de depuracion. Se analizan POR SEPARADO. ***")


def summary_metrics():
    """Métricas promedio (± std) por algoritmo y origen, desde los eval CSV."""
    rows = load_eval_rows()
    if not rows:
        print("[analyze] no hay CSV de evaluacion (eval_*.csv) en results/.")
        return
    _warn_source([r["source"] for r in rows])

    groups = defaultdict(list)
    for r in rows:
        groups[(r["source"], r["algorithm"])].append(r)

    out_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    cols = ["source", "algorithm", "n_eval_episodes",
            "reward_mean", "reward_std", "success_rate",
            "time_to_goal_mean", "falls_mean", "collisions_mean"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for (source, algo), rs in sorted(groups.items()):
            rewards = [_f(r["cumulative_reward"]) for r in rs]
            succ = [_f(r["is_success"]) for r in rs]
            ttg = [_f(r["time_to_goal"]) for r in rs if r["time_to_goal"] not in ("", None)]
            falls = [_f(r["falls"]) for r in rs]
            cols_ = [_f(r["collisions"]) for r in rs]
            w.writerow({
                "source": source, "algorithm": algo, "n_eval_episodes": len(rs),
                "reward_mean": round(np.mean(rewards), 3),
                "reward_std": round(np.std(rewards), 3),
                "success_rate": round(np.mean(succ), 4),
                "time_to_goal_mean": round(np.mean(ttg), 3) if ttg else "",
                "falls_mean": round(np.mean(falls), 3),
                "collisions_mean": round(np.mean(cols_), 3),
            })
    print(f"[analyze] metricas resumen -> {out_path}")
    # eco en consola
    for row in _read_csv(out_path):
        print("   ", {k: row[k] for k in ("source", "algorithm", "reward_mean",
                                           "reward_std", "success_rate")})


def _lazy_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def learning_curves(exp_id=None):
    files = load_train_files()
    if exp_id:
        files = [f for f in files if f"_{exp_id}_" in os.path.basename(f)]
    if not files:
        print("[analyze] no hay CSV de entrenamiento (train_*.csv).")
        return
    plt = _lazy_plt()

    # agrupa por (exp, algo) -> lista de (steps, smoothed_return) por semilla
    curves = defaultdict(list)
    sources = []
    for path in files:
        rows = _read_csv(path)
        if not rows:
            continue
        sources.append(rows[0]["source"])
        exp = rows[0]["exp_id"]
        algo = rows[0]["algorithm"]
        steps = np.array([_f(r["step"]) for r in rows])
        rets = np.array([_f(r["ep_return"]) for r in rows])
        if len(rets) >= 5:
            k = min(10, len(rets))
            sm = np.convolve(rets, np.ones(k) / k, mode="valid")
            steps = steps[k - 1:]
        else:
            sm = rets
        curves[(exp, algo)].append((steps, sm))
    _warn_source(sources)

    exps = sorted({e for (e, _a) in curves})
    for exp in exps:
        plt.figure(figsize=(9, 5))
        for algo in ALGOS:
            seeds = curves.get((exp, algo), [])
            if not seeds:
                continue
            all_steps = np.unique(np.concatenate([s for s, _ in seeds]))
            interp = []
            for s, r in seeds:
                if len(s) >= 2:
                    interp.append(np.interp(all_steps, s, r, left=np.nan, right=np.nan))
            if not interp:
                continue
            m = np.nanmean(interp, axis=0)
            sd = np.nanstd(interp, axis=0)
            plt.plot(all_steps, m, label=f"{algo} (n={len(seeds)})")
            plt.fill_between(all_steps, m - sd, m + sd, alpha=0.15)
        plt.xlabel("Pasos de entrenamiento")
        plt.ylabel("Recompensa por episodio (media movil, promedio entre semillas)")
        plt.title(f"Curvas de aprendizaje - escenario {exp}")
        plt.legend()
        plt.grid(alpha=0.3)
        out = os.path.join(FIGURES_DIR, f"learning_curves_{exp}.png")
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[analyze] curva -> {out}")


def success_and_boxplots():
    rows = [r for r in load_eval_rows() if r["source"] == "WEBOTS_REAL"] or load_eval_rows()
    if not rows:
        return
    plt = _lazy_plt()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # barras de tasa de exito
    succ = defaultdict(list)
    rew = defaultdict(list)
    for r in rows:
        succ[r["algorithm"]].append(_f(r["is_success"]))
        rew[r["algorithm"]].append(_f(r["cumulative_reward"]))
    algos = [a for a in ALGOS if a in succ]

    plt.figure(figsize=(7, 4))
    plt.bar(algos, [np.mean(succ[a]) for a in algos])
    plt.ylabel("Tasa de exito")
    plt.ylim(0, 1)
    plt.title("Tasa de exito por algoritmo (evaluacion)")
    plt.grid(alpha=0.3, axis="y")
    out = os.path.join(FIGURES_DIR, "success_rate.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analyze] barras -> {out}")

    plt.figure(figsize=(7, 4))
    # matplotlib >= 3.9 renombro 'labels' a 'tick_labels' en boxplot.
    try:
        plt.boxplot([rew[a] for a in algos], tick_labels=algos)
    except TypeError:
        plt.boxplot([rew[a] for a in algos], labels=algos)
    plt.ylabel("Recompensa acumulada (evaluacion)")
    plt.title("Distribucion de recompensa por algoritmo")
    plt.grid(alpha=0.3, axis="y")
    out = os.path.join(FIGURES_DIR, "reward_boxplot.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[analyze] boxplot -> {out}")


def race_summary():
    rows = load_race_rows()
    if not rows:
        return
    _warn_source([r["source"] for r in rows])
    wins = defaultdict(int)
    for r in rows:
        if str(r.get("finish_rank")) == "1":
            wins[r["algorithm"]] += 1
    out = os.path.join(RESULTS_DIR, "race_summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "wins"])
        for a in ALGOS:
            w.writerow([a, wins.get(a, 0)])
    print(f"[analyze] resumen de carreras -> {out}  victorias={dict(wins)}")


def main():
    p = argparse.ArgumentParser(description="Agrega CSV y genera figuras/tablas.")
    p.add_argument("--exp", default=None, help="limita las curvas a un escenario")
    args = p.parse_args()
    summary_metrics()
    try:
        learning_curves(args.exp)
        success_and_boxplots()
    except ImportError:
        print("[analyze] matplotlib no instalado: se omiten las figuras "
              "(las tablas CSV si se generaron).")
    race_summary()


if __name__ == "__main__":
    main()
