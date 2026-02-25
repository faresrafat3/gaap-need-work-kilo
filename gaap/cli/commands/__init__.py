"""
CLI Commands Package
"""

from .chat import cmd_chat, cmd_interactive
from .config import cmd_config
from .history import cmd_history
from .models import cmd_models
from .providers import cmd_providers
from .system import cmd_doctor, cmd_status, cmd_version

from .sovereign import cmd_research, cmd_debug, cmd_dream, cmd_audit
from .feedback import cmd_feedback
from .debt import cmd_debt

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
    "cmd_research",
    "cmd_debug",
    "cmd_dream",
    "cmd_audit",
    "cmd_feedback",
    "cmd_debt",
]
