"""Controlador del robot DARwIn-OP: postura firme + animacion estetica (locomocion cuasi-estatica)."""

from __future__ import annotations

import math
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.config import SafetyConfig  # noqa: E402

LEG_MOTORS = [
    "PelvYL", "PelvYR", "PelvL", "PelvR",
    "LegUpperL", "LegUpperR", "LegLowerL", "LegLowerR",
    "AnkleL", "AnkleR", "FootL", "FootR",
]

class PythonCPGGait:
    """Postura firme de pie + leve animacion de piernas al avanzar (con clamp)."""

    def __init__(self, robot, timestep):
        self.robot = robot
        self.timestep = timestep
        self.phase = 0.0
        self.freq = 1.0
        self.motors = {}
        self.limits = {}
        self._warned = set()
        for name in LEG_MOTORS:
            dev = robot.getDevice(name)
            if dev is not None:
                self.motors[name] = dev
                try:
                    lo, hi = dev.getMinPosition(), dev.getMaxPosition()
                    if lo == 0.0 and hi == 0.0:
                        lo, hi = -3.14, 3.14
                    self.limits[name] = (lo, hi)
                except Exception:
                    self.limits[name] = (-3.14, 3.14)

    def _set(self, name, pos):
        m = self.motors.get(name)
        if m is None:
            return
        lo, hi = self.limits.get(name, (-3.14, 3.14))
        margin = 0.02
        lo_s, hi_s = lo + margin, hi - margin
        clamped = max(lo_s, min(hi_s, pos))
        if abs(clamped - pos) > 1e-6 and name not in self._warned:
            self._warned.add(name)
            print(f"[darwin_agent][CPG] '{name}' recortado: pedido {pos:.3f} "
                  f"fuera de [{lo:.3f}, {hi:.3f}] (aviso unico por motor).", flush=True)
        try:
            m.setPosition(clamped)
        except Exception:
            pass

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
        dt = self.timestep / 1000.0
        self.phase += 2 * math.pi * self.freq * dt
        s = math.sin(self.phase)
        anim = 0.15 * abs(x_amp)
        base = self.STAND
        self._set("LegUpperL", base["LegUpperL"] + anim * s)
        self._set("LegUpperR", base["LegUpperR"] - anim * s)
        self._set("LegLowerL", base["LegLowerL"] - 0.10 * abs(x_amp) * max(0.0, s))
        self._set("LegLowerR", base["LegLowerR"] + 0.10 * abs(x_amp) * max(0.0, -s))
        self._set("AnkleL", base["AnkleL"])
        self._set("AnkleR", base["AnkleR"])

def main():
    from controller import Robot  # type: ignore

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    safety = SafetyConfig()
    print(f"[darwin_agent] iniciado. basicTimeStep={timestep} ms", flush=True)

    def try_enable(name):
        dev = robot.getDevice(name)
        if dev is not None:
            try:
                dev.enable(timestep)
            except Exception:
                pass
        return dev

    try_enable("Accelerometer")
    try_enable("Gyro")
    for name in LEG_MOTORS:
        try_enable(name + "S")

    receiver = robot.getDevice("receiver")
    if receiver is not None:
        receiver.enable(timestep)
        print("[darwin_agent] receiver habilitado (canal del PROTO).", flush=True)
    else:
        print("[darwin_agent] AVISO: no se encontro el Receiver 'receiver'.", flush=True)

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
        print("[darwin_agent] backend de marcha: managers.", flush=True)
    except Exception as e:
        cpg = PythonCPGGait(robot, timestep)
        cpg.start()
        gait = cpg
        backend = "cpg"
        print(f"[darwin_agent] managers no disponible ({type(e).__name__}: {e}).", flush=True)
        print("[darwin_agent] locomocion CUASI-ESTATICA (postura firme + animacion).", flush=True)
        print(f"[darwin_agent] motores de pierna detectados: {sorted(cpg.motors.keys())}", flush=True)

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
