"""Windows console hygiene for the web edition's subprocess spawns.

A windowed exe (the desktop app) has no console. Every console child it
spawns — git.exe, curl.exe, taskkill.exe, cmd.exe wrappers, provider CLIs —
allocates its own console window on launch, which flashes on the desktop on
every settings save, project refresh or status check (measured: +1 visible
window per child; with CREATE_NO_WINDOW, +0). The flag suppresses that
allocation for children whose output the app pipes. The deliberate visible
terminal (spawn_terminal) keeps its window: `start` allocates one for the
script regardless of the wrapper's own visibility.
"""
from __future__ import annotations

import os
import subprocess

NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def quiet_flags(extra: int = 0) -> int:
    """Creation flags that suppress console windows, plus any extras."""
    if os.name != "nt":
        return 0
    return NO_WINDOW | extra
