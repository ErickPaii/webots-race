"""Protocolo de comunicación entre el entorno Gymnasium y el Supervisor de Webots.

El proceso de entrenamiento (Stable-Baselines3 / agente propio) corre en un
intérprete Python normal y NO dentro de Webots. El controlador Supervisor de
Webots corre dentro del simulador. Ambos se comunican por un socket TCP en
localhost con mensajes JSON delimitados por salto de línea ("\\n").

Formato:
  - El entorno (cliente) envía peticiones:
        {"cmd": "reset", "config": {...}}            -> nueva configuración
        {"cmd": "step", "action": <int>}             -> aplicar acción
        {"cmd": "close"}                             -> terminar
  - El Supervisor (servidor) responde con el estado:
        {"obs": [...], "reward": float, "terminated": bool,
         "truncated": bool, "info": {...}}

Mantener este archivo SIN dependencias de Webots ni de gymnasium para que lo
puedan importar tanto el controlador (dentro de Webots) como el entorno.
"""

from __future__ import annotations

import json
import socket
from typing import Optional


def send_msg(sock: socket.socket, obj: dict) -> None:
    """Envía un dict como línea JSON terminada en '\\n'."""
    data = (json.dumps(obj) + "\n").encode("utf-8")
    sock.sendall(data)


class LineReader:
    """Lee mensajes JSON delimitados por '\\n' desde un socket con buffer."""

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self._buf = b""

    def read_msg(self, timeout: Optional[float] = None) -> Optional[dict]:
        if timeout is not None:
            self.sock.settimeout(timeout)
        while b"\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                return None  # conexión cerrada
            self._buf += chunk
        line, _, rest = self._buf.partition(b"\n")
        self._buf = rest
        return json.loads(line.decode("utf-8"))
