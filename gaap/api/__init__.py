"""
GAAP REST API Package
======================

REST API endpoints for GAAP system.
"""

from gaap.api.budget import router as budget_router
from gaap.api.config import router as config_router
from gaap.api.healing import router as healing_router
from gaap.api.memory import router as memory_router
from gaap.api.providers import router as providers_router
from gaap.api.providers_status import router as providers_status_router
from gaap.api.research import router as research_router
from gaap.api.sessions import router as sessions_router
from gaap.api.system import router as system_router

__all__ = [
    "research_router",
    "config_router",
    "providers_router",
    "providers_status_router",
    "healing_router",
    "memory_router",
    "budget_router",
    "sessions_router",
    "system_router",
]
