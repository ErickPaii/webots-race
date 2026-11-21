"""race_agent.py — Controlador de cada DARwIn-OP en la carrera de 4 robots.

Es idéntico en espíritu a darwin_agent.py (recibe "x_amp a_amp" por el Receiver
nativo y aplica la marcha con el backend disponible: managers C++ o CPG Python),
pero cada robot escucha en su propio canal (definido por el campo `channel` del
PROTO en el .wbt). Reutiliza la lógica de marcha de darwin_agent.
"""

from __future__ import annotations

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Reutilizamos el main del controlador individual: su lógica de marcha y de
# recepción de comandos es la misma. El canal lo fija el PROTO en el .wbt.
from controllers.darwin_agent.darwin_agent import main  # noqa: E402

if __name__ == "__main__":
    main()
