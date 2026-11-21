# Carrera de DARwIn-OP con Aprendizaje por Refuerzo (Webots)

Proyecto del Parcial 02 de Inteligencia Artificial (Universidad Distrital).
Cuatro agentes de RL compiten en una carrera con el robot humanoide **DARwIn-OP**
en **Webots R2025a**: un agente propio (Actor-Critic + *population discovery*,
del Taller 2) y tres bases de **Stable-Baselines3** (PPO, DQN, A2C).

> **Sobre los resultados:** este repositorio contiene el **código completo** y el
> **esqueleto del informe**. Las cifras del informe deben generarse ejecutando
> los experimentos en TU máquina con Webots. Todo lo pendiente está marcado como
> `[PENDIENTE]`. Hay un **modo depuración** (sin Webots) solo para validar que el
> pipeline corre; sus resultados **no son válidos para el informe** y se marcan
> con `source=DEBUG_SIM_NO_WEBOTS` en los CSV.

## Estructura

```
webots_race/
├── worlds/
│   ├── darwin_race.wbt          # 1 robot (entrenamiento/evaluación individual)
│   └── darwin_race_4.wbt        # 4 robots (carrera simultánea)
├── controllers/
│   ├── darwin_agent/            # controlador del robot (gait manager / CPG)
│   ├── supervisor_controller/   # supervisor del mundo de 1 robot (TCP :10001)
│   ├── race_agent/              # controlador de cada robot en la carrera 4x
│   └── race_supervisor_controller/  # supervisor de la carrera 4x (TCP :10002)
├── envs/
│   ├── config.py                # RaceConfig / RewardConfig / SafetyConfig
│   ├── darwin_race_env.py       # entorno Gymnasium (reset/step/spaces) sobre Webots
│   ├── debug_sim_env.py         # entorno de DEPURACIÓN sin Webots (NO para informe)
│   └── protocol.py              # protocolo TCP supervisor <-> entorno
├── agents/
│   ├── custom_agent.py          # AGENTE PROPIO: A2C + population discovery (propio)
│   └── sb3_factory.py           # PPO / DQN / A2C (librería Stable-Baselines3)
├── safety/
│   ├── shield.py                # filtro de acciones anti-colisión (safe learning)
│   └── cost.py                  # función de costo separada (CMDP) + métricas
├── experiments/
│   ├── experiment_grid.py       # 23 escenarios (>= 20 requeridos)
│   └── run_all.py               # barrido completo (train+eval de todas las corridas)
├── results/                     # CSVs generados (vacío hasta ejecutar)
├── figures/                     # figuras generadas (vacío hasta ejecutar)
├── report/                      # informe IEEE (LaTeX)
├── train.py  evaluate.py  run_race.py  analyze.py
├── check_setup.py               # verifica tu entorno
├── requirements.txt
└── README.md
```

## Requisitos

- **Webots R2025a** (o la estable más reciente). Descárgalo de cyberbotics.com.
- **Python 3.10 o superior** (probado con 3.11). En Windows, la versión de
  Python 3.x de 64 bits del instalador oficial funciona bien.
- CPU suficiente (no se requiere GPU; el agente propio y las bases usan MLPs
  pequeñas). Perfil de hardware de referencia del autor: Windows 10 22H2,
  Intel HD Graphics 4600 (sin GPU dedicada), 20 GB RAM.

---

## Instalación paso a paso (Windows 10)

Abre **PowerShell** o **CMD** en la carpeta `webots_race` y ejecuta, en este orden:

```powershell
REM 1) Crear y activar un entorno virtual
python -m venv venv
venv\Scripts\activate

REM 2) Instalar PyTorch (CPU) desde su índice propio
pip install torch --index-url https://download.pytorch.org/whl/cpu

REM 3) Instalar el resto de dependencias
pip install -r requirements.txt

REM 4) Verificar que todo está listo (sin Webots todavía)
python check_setup.py
```

`check_setup.py` debe terminar con "Todo listo". Si algo falla, imprime qué
instalar.

> **Linux/Mac (referencia):** idéntico salvo que el venv se activa con
> `source venv/bin/activate`. El resto de comandos es igual.

### Instalar Webots y los controladores del DARwIn-OP

1. Instala **Webots R2025a**.
2. Este proyecto usa el PROTO nativo `Darwin-op` (ruta interna
   `robots > darwin-op > Darwin-op`, que en disco vive en
   `.../projects/robots/robotis/darwin-op/...`). Los `EXTERNPROTO` del `.wbt`
   apuntan a la versión R2025a en línea, así que Webots los resuelve solo.

#### Backend de marcha (importante — léelo)

El controlador `darwin_agent` necesita hacer caminar al robot. Hay dos backends:

- **CPG en Python (por defecto, recomendado):** un generador de marcha
  sinusoidal en Python puro que mueve los motores de las piernas directamente.
  **No requiere compilar nada** y funciona en una instalación limpia de Webots.
  Es el que se usa automáticamente.
- **Managers oficiales (`RobotisOp2GaitManager`, C++):** son el gait manager
  nativo del DARwIn-OP, pero Webots **no distribuye un binding de Python
  precompilado** para ellos (son C++ y no hay interfaz SWIG en el repositorio
  oficial). Por eso `import managers` **suele fallar** en una instalación
  limpia. El controlador lo intenta primero y, si no está, cae al CPG en Python
  mostrando un aviso en la consola de Webots.
  - Si quisieras usar los managers C++, tendrías que compilar la librería
    `projects/robots/robotis/darwin-op/libraries/managers` de tu instalación de
    Webots y generar un wrapper de Python; **no es necesario** para este
    proyecto y añade complejidad. El CPG en Python cumple la función.

Resumen de la decisión: **no necesitas copiar ni configurar archivos extra** para
que el robot camine; el backend CPG en Python funciona out-of-the-box.

---

## Cómo ejecutar

### A. Validar el pipeline SIN Webots (modo depuración)

Esto comprueba que `train`/`evaluate`/`analyze` corren de punta a punta. **Los
resultados NO sirven para el informe** (quedan marcados como
`DEBUG_SIM_NO_WEBOTS`).

```powershell
venv\Scripts\activate
python -m experiments.run_all --profile fast --debug
python analyze.py
```

### B0. VERIFICAR LA MARCHA antes de entrenar (recomendado)

Antes de invertir noches entrenando, comprueba que el robot camina:

1. Abre `worlds/darwin_race.wbt` en Webots. **Para VER al robot, usa el modo
   Real-time** (botón ▶ normal), no el modo Fast. En modo Fast Webots muestra
   "No rendering" a propósito (desactiva el dibujado para ir más rápido); eso es
   normal y NO es un error. Para entrenar luego sí conviene Fast.
2. Con el mundo en Play, en otra terminal (venv activado):

```powershell
venv\Scripts\activate
python test_walk.py
```

Esto hace avanzar al robot ~8 s e imprime la distancia recorrida y si se cayó.
- Si avanza > 0.5 m sin caerse: listo para entrenar.
- Si no avanza o se cae: copia la salida y los avisos de la consola de Webots.

> **"No rendering" al abrir el mundo:** es el modo Fast de Webots, no un fallo
> del proyecto. El supervisor NO cambia el modo de simulación. Para ver el robot,
> la salida y la meta, usa Real-time y, si hace falta, encuadra con el ratón.

### B. Ejecutar de verdad con Webots (para el informe)

Para **entrenar/evaluar un agente** (mundo de 1 robot):

1. Abre Webots y carga `worlds/darwin_race.wbt`. Pulsa **Play** (▶). El
   controlador `supervisor_controller` abre el servidor TCP en el puerto 10001 y
   queda esperando ("escuchando en 127.0.0.1:10001" en la consola de Webots).
   Sugerencia: para entrenar, pon la simulación en modo **Fast** (sin render)
   para acelerar; para observar, usa Real-time.
2. En otra terminal (con el venv activado), lanza el entrenamiento:

```powershell
venv\Scripts\activate

REM Entrenar un agente en un escenario (ej.: PPO en el escenario A2)
python train.py --algo PPO    --exp A2 --seed 0 --timesteps 200000
python train.py --algo DQN    --exp A2 --seed 0 --timesteps 200000
python train.py --algo A2C    --exp A2 --seed 0 --timesteps 200000
python train.py --algo CUSTOM --exp A2 --seed 0 --timesteps 200000 --variant population

REM Evaluar (determinista, sin exploración) -> tasa de éxito, tiempos, seguridad
python evaluate.py --algo PPO    --exp A2 --seed 0 --episodes 10
python evaluate.py --algo CUSTOM --exp A2 --seed 0 --episodes 10
```

Cada `train` produce `results/train_<algo>_<exp>_seed<seed>.csv` (episodios de
entrenamiento, para las curvas) y un modelo en `models/`. Cada `evaluate`
produce `results/eval_<algo>_<exp>_seed<seed>.csv`.

#### Barrido completo de los 20+ experimentos

```powershell
python -m experiments.run_all --profile full --timesteps 200000 --seeds 0 1 2
```

Esto entrena y evalúa los 4 algoritmos en los 23 escenarios con 3 semillas.
**Requiere que Webots esté abierto en `darwin_race.wbt` y en Play** durante todo
el barrido. (Para una primera prueba, usa `--profile fast` sin `--debug`.)

### C. Carrera simultánea de los 4 robots

1. Abre `worlds/darwin_race_4.wbt` en Webots y pulsa **Play** (abre el puerto
   10002). Requiere que ya existan los 4 modelos entrenados de ese escenario en
   `models/`.
2. En la terminal:

```powershell
python run_race.py --exp A2 --seed 0
```

Produce `results/race_A2_seed0.csv` con el **orden de llegada** y los **tiempos**.

### D. Generar tablas y figuras

```powershell
python analyze.py
```

Genera `figures/learning_curves_<exp>.png`, `figures/success_rate.png`,
`figures/reward_boxplot.png`, `results/summary_metrics.csv` y
`results/race_summary.csv`. Separa siempre datos reales de datos de depuración.

---

## Qué pegar de vuelta para completar el informe

Tras ejecutar en tu máquina, comparte:

- La carpeta `results/` (todos los CSV).
- La carpeta `figures/` (PNG generados).
- Cualquier error de consola (de Webots o de Python) si algo falló.

Con eso se rellenan las tablas y figuras marcadas `[PENDIENTE]` en el informe.

## Perfiles de cómputo

- **fast:** 4 escenarios, pocos timesteps, 1 semilla. Para validar el pipeline.
- **full:** 23 escenarios, timesteps del informe, 3 semillas. Para resultados.

El tiempo real depende de la velocidad de Webots en tu equipo; sin GPU y con
Webots en modo Fast, empieza con `--timesteps 50000` por agente para estimar, y
sube si el tiempo lo permite.
