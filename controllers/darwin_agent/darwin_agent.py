"""Controlador del robot DARwIn-OP: ejecuta la marcha a partir de comandos.

Corre DENTRO de Webots como controlador del nodo DARwIn-OP. Recibe del
Supervisor (por el Receiver nativo, canal 0) los comandos de marcha
"x_amp a_amp" (amplitud de avance X, amplitud de giro A) que corresponden a la
acción discreta elegida por el agente, y los convierte en movimiento.

DOS BACKENDS DE MARCHA (importante para instalación limpia de Webots R2025a):

  1) Backend "managers" (C++ RobotisOp2GaitManager): es el gait manager oficial
     del DARwIn-OP. PERO Webots NO distribuye un binding de Python precompilado
     para estos managers (son C++; no hay interfaz SWIG en el repositorio). Por
     tanto `import managers` suele FALLAR en una instalación limpia salvo que el
     usuario compile la librería. Lo intentamos primero por si está disponible.

  2) Backend "cpg" (POR DEFECTO, Python puro): un generador central de patrones
     (CPG) sinusoidal que mueve directamente los motores de las piernas para
     producir una marcha estable. NO requiere compilar nada: funciona con el
     controlador Python estándar de Webots. Es el que garantiza que el proyecto
     corra "out of the box".

En ambos casos, el agente de RL solo decide avanzar/girar/parar: el patrón de
marcha (equilibrio) lo resuelve este controlador, no el RL (decisión de diseño
justificada en el informe).

Safe learning a nivel de actuador: las amplitudes se recortan a límites seguros
(límite de velocidad de marcha / giro) antes de aplicarlas.
"""

from __future__ import annotations

import math
import os
import sys

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.config import SafetyConfig  # noqa: E402


# Nombres de los 12 motores de las piernas + pelvis usados por el CPG.
# (verificados contra el PROTO Darwin-op de Webots R2025a)
LEG_MOTORS = [
    "PelvYL", "PelvYR", "PelvL", "PelvR",
    "LegUpperL", "LegUpperR", "LegLowerL", "LegLowerR",
    "AnkleL", "AnkleR", "FootL", "FootR",
]


class PythonCPGGait:
    """Generador de marcha sinusoidal (CPG) en Python puro.

    Produce una marcha funcional controlable por amplitud de avance (x_amp) y de
    giro (a_amp). Es el backend por defecto para que el proyecto funcione sin
    compilar los managers C++ del DARwIn-OP.

    Convencion de ejes de las articulaciones (verificada contra los
    minPosition/maxPosition del PROTO Darwin-op R2025a): las piernas izquierda y
    derecha son ESPEJO, por lo que la flexion ocurre con signos OPUESTOS:

        LegUpper (cadera, pitch):  izq flexiona en +,  der flexiona en -
        LegLower (rodilla):        izq flexiona en -,  der flexiona en +
        Ankle    (tobillo, pitch): izq compensa en +,  der compensa en -

    Rangos reales (rad) relevantes:
        LegUpperL [-0.50, 1.68]   LegUpperR [-1.77, 0.45]
        LegLowerL [-2.25, 0.03]   LegLowerR [-0.02, 2.25]
        AnkleL    [-1.39, 1.22]   AnkleR    [-1.24, 1.38]

    Toda posicion pedida se recorta (clamp) al rango real de cada motor leido con
    getMinPosition()/getMaxPosition(). Cada aviso de recorte se imprime UNA sola
    vez por motor para no saturar la consola.
    """

    def __init__(self, robot, timestep):
        self.robot = robot
        self.timestep = timestep
        self.phase = 0.0
        self.freq = 1.0  # Hz aprox. del ciclo de marcha
        self.motors = {}
        self.limits = {}          # name -> (min, max)
        self._warned = set()      # motores ya avisados de recorte
        for name in LEG_MOTORS:
            dev = robot.getDevice(name)
            if dev is not None:
                self.motors[name] = dev
                try:
                    lo, hi = dev.getMinPosition(), dev.getMaxPosition()
                    # Si el motor no declara limites, getMin/Max devuelven 0,0.
                    if lo == 0.0 and hi == 0.0:
                        lo, hi = -3.14, 3.14
                    self.limits[name] = (lo, hi)
                except Exception:
                    self.limits[name] = (-3.14, 3.14)

    def _set(self, name, pos):
        """Fija la posicion de un motor recortandola a su rango real (clamp)."""
        m = self.motors.get(name)
        if m is None:
            return
        lo, hi = self.limits.get(name, (-3.14, 3.14))
        # margen de seguridad para no tocar el tope exacto
        margin = 0.02
        lo_s, hi_s = lo + margin, hi - margin
        clamped = max(lo_s, min(hi_s, pos))
        if abs(clamped - pos) > 1e-6 and name not in self._warned:
            self._warned.add(name)
            print(f"[darwin_agent][CPG] '{name}' recortado: pedido {pos:.3f} "
                  f"fuera de [{lo:.3f}, {hi:.3f}] (aviso unico por motor).")
        try:
            m.setPosition(clamped)
        except Exception:
            pass

    # Postura de pie semiflexionada (crouch), con signos espejo correctos.
    STAND = {
        "LegUpperL": 0.45, "LegUpperR": -0.45,
        "LegLowerL": -0.9, "LegLowerR": 0.9,
        "AnkleL": 0.45,    "AnkleR": -0.45,
        "PelvL": 0.0,      "PelvR": 0.0,
        "PelvYL": 0.0,     "PelvYR": 0.0,
        "FootL": 0.0,      "FootR": 0.0,
    }

    def start(self):
        for name, pos in self.STAND.items():
            self._set(name, pos)

    def step(self, x_amp, a_amp):
        """Avanza un ciclo de CPG con la amplitud de avance/giro pedidos.

        x_amp in [-1, 1] escala la longitud de paso; a_amp in [-0.5, 0.5] el giro.
        Las amplitudes son pequenas y quedan dentro del rango de cada motor.
        """
        dt = self.timestep / 1000.0
        self.phase += 2 * math.pi * self.freq * dt
        s = math.sin(self.phase)       # fase pierna izquierda
        # Amplitudes conservadoras (rad) dentro de rango:
        hip_a = 0.35 * x_amp           # oscilacion de cadera (pitch)
        knee_a = 0.35 * abs(x_amp)     # flexion extra de rodilla en la fase de vuelo
        ank_a = 0.20 * x_amp           # compensacion de tobillo
        turn = 0.3 * a_amp             # giro con PelvY

        base = self.STAND
        # Cadera: espejo (+ izq, - der). Oscilan en antifase (s y -s).
        self._set("LegUpperL", base["LegUpperL"] + hip_a * s)
        self._set("LegUpperR", base["LegUpperR"] - hip_a * s)
        # Rodilla: flexiona en la fase de vuelo de cada pierna (signo espejo).
        self._set("LegLowerL", base["LegLowerL"] - knee_a * max(0.0, s))
        self._set("LegLowerR", base["LegLowerR"] + knee_a * max(0.0, -s))
        # Tobillo: compensa (espejo).
        self._set("AnkleL", base["AnkleL"] + ank_a * s)
        self._set("AnkleR", base["AnkleR"] - ank_a * s)
        # Giro: rotacion de la pelvis en yaw (mismo signo ambos lados).
        self._set("PelvYL", turn)
        self._set("PelvYR", turn)


def main() -> None:  # pragma: no cover - requiere Webots
    from controller import Robot  # type: ignore

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    safety = SafetyConfig()

    def try_enable(name):
        dev = robot.getDevice(name)
        if dev is not None:
            try:
                dev.enable(timestep)
            except Exception:
                pass
        return dev

    # Sensores requeridos por el gait manager y útiles para la observación.
    try_enable("Accelerometer")
    gyro = try_enable("Gyro")
    # Habilitar los sensores de posición de las piernas (requerido por managers).
    for name in LEG_MOTORS:
        try_enable(name + "S")

    # Receiver nativo del PROTO (canal 0).
    receiver = robot.getDevice("receiver")
    if receiver is not None:
        receiver.enable(timestep)

    # --- Selección de backend de marcha ---
    backend = "cpg"
    gait = None
    try:
        from managers import RobotisOp2GaitManager, RobotisOp2MotionManager  # type: ignore
        RobotisOp2MotionManager(robot)
        gait = RobotisOp2GaitManager(robot, "config.ini")
        gait.start()
        gait.setXAmplitude(0.0)
        gait.setAAmplitude(0.0)
        backend = "managers"
        print("[darwin_agent] backend de marcha: managers (RobotisOp2GaitManager).")
    except Exception as e:
        cpg = PythonCPGGait(robot, timestep)
        cpg.start()
        gait = cpg
        backend = "cpg"
        print(f"[darwin_agent] managers no disponible ({e}).")
        print("[darwin_agent] backend de marcha: CPG en Python puro (no requiere compilar managers).")

    x_amp, a_amp = 0.0, 0.0

    def clip(v, lim):
        return max(-lim, min(lim, v))

    while robot.step(timestep) != -1:
        if receiver is not None:
            while receiver.getQueueLength() > 0:
                try:
                    data = receiver.getString()
                except Exception:
                    data = receiver.getData().decode("utf-8", "ignore")
                receiver.nextPacket()
                try:
                    parts = data.split()
                    x_amp = clip(float(parts[0]), safety.max_forward_amplitude)
                    a_amp = clip(float(parts[1]), safety.max_turn_amplitude)
                except (ValueError, IndexError):
                    pass

        if backend == "managers":
            gait.setXAmplitude(x_amp)
            gait.setAAmplitude(a_amp)
            gait.step(timestep)
        else:
            gait.step(x_amp, a_amp)


if __name__ == "__main__":
    main()
