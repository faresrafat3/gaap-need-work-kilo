"""
CLI Commands Package
"""

from .chat import cmd_chat, cmd_interactive
from .config import cmd_config
from .history import cmd_history
from .models import cmd_models
from .providers import cmd_providers
from .system import cmd_doctor, cmd_status, cmd_version
from .web import cmd_web

__all__ = [
    "cmd_chat",
    "cmd_interactive",
    "cmd_providers",
    "cmd_models",
    "cmd_config",
    "cmd_history",
    "cmd_status",
    "cmd_version",
    "cmd_doctor",
    "cmd_web",
]
