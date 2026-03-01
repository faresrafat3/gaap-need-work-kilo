"""
Config API - Configuration Management Endpoints
================================================

Provides REST API for reading and updating GAAP configuration.

Endpoints:
- GET /api/config - Get full configuration
- PUT /api/config - Update full configuration
- GET /api/config/{module} - Get module configuration
- PUT /api/config/{module} - Update module configuration
- POST /api/config/validate - Validate configuration
- POST /api/config/reload - Reload from file
- GET /api/config/presets - List available presets
- GET /api/config/schema - Get configuration schema
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gaap.core.config import (
    GAAPConfig,
    get_config,
    get_config_manager,
)
from gaap.core.exceptions import ConfigurationError

logger = logging.getLogger("gaap.api.config")

router = APIRouter(prefix="/api/config", tags=["config"])


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""

    config: dict[str, Any]
    validate_config: bool = True


class ConfigResponse(BaseModel):
    """Configuration response."""

    success: bool
    config: dict[str, Any] | None = None
    error: str | None = None


class ModuleConfigUpdateRequest(BaseModel):
    """Request to update module configuration."""

    config: dict[str, Any]
    validate_config: bool = True


class ValidationResult(BaseModel):
    """Validation result."""

    valid: bool
    errors: list[str] = []
    warnings: list[str] = []


class PresetInfo(BaseModel):
    """Information about a preset."""

    name: str
    description: str
    modules: list[str]


def _config_to_dict(config: GAAPConfig) -> dict[str, Any]:
    """Convert GAAPConfig to dictionary."""
    return config.to_dict()


def _module_to_dict(module_config: Any) -> dict[str, Any]:
    """Convert a module config dataclass to dict."""
    if hasattr(module_config, "__dataclass_fields__"):
        return {k: v for k, v in asdict(module_config).items() if not k.startswith("_")}
    elif hasattr(module_config, "__dict__"):
        return {k: v for k, v in module_config.__dict__.items() if not k.startswith("_")}
    elif isinstance(module_config, dict):
        return module_config
    return {}


@router.get("", response_model=ConfigResponse)
async def get_full_config() -> ConfigResponse:
    """Get the complete configuration."""
    try:
        config = get_config()
        return ConfigResponse(
            success=True,
            config=_config_to_dict(config),
        )
    except Exception as e:
        logger.error(f"Failed to get config: {e}")
        return ConfigResponse(success=False, error=str(e))


@router.put("", response_model=ConfigResponse)
async def update_full_config(request: ConfigUpdateRequest) -> ConfigResponse:
    """Update the complete configuration."""
    try:
        manager = get_config_manager()

        config = manager._dict_to_config(request.config)

        if request.validate_config:
            manager._validate_config(config)

        manager._config = config

        logger.info("Configuration updated successfully")

        return ConfigResponse(
            success=True,
            config=_config_to_dict(config),
        )
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return ConfigResponse(success=False, error=str(e))


@router.get("/{module}", response_model=ConfigResponse)
async def get_module_config(module: str) -> ConfigResponse:
    """Get configuration for a specific module."""
    try:
        config = get_config()
        module_config = getattr(config, module, None)

        if module_config is None:
            raise HTTPException(status_code=404, detail=f"Module '{module}' not found")

        return ConfigResponse(success=True, config=_module_to_dict(module_config))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get module config: {e}")
        return ConfigResponse(success=False, error=str(e))


@router.put("/{module}", response_model=ConfigResponse)
async def update_module_config(
    module: str,
    request: ModuleConfigUpdateRequest,
) -> ConfigResponse:
    """Update configuration for a specific module."""
    try:
        manager = get_config_manager()
        config = manager.config

        if not hasattr(config, module):
            raise HTTPException(status_code=404, detail=f"Module '{module}' not found")

        module_config = getattr(config, module)

        for key, value in request.config.items():
            if hasattr(module_config, key):
                setattr(module_config, key, value)
            else:
                logger.warning(f"Unknown config field: {module}.{key}")

        if request.validate_config:
            manager._validate_config(config)

        logger.info(f"Module '{module}' configuration updated")

        return ConfigResponse(success=True, config=_module_to_dict(module_config))
    except HTTPException:
        raise
    except ConfigurationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update module config: {e}")
        return ConfigResponse(success=False, error=str(e))


@router.post("/validate", response_model=ValidationResult)
async def validate_config(request: ConfigUpdateRequest) -> ValidationResult:
    """Validate configuration without applying."""
    try:
        manager = get_config_manager()

        config = manager._dict_to_config(request.config)

        errors: list[str] = []
        try:
            manager._validate_config(config)
        except ConfigurationError as e:
            errors = e.details.get("errors", [str(e)])

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=[],
        )
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return ValidationResult(valid=False, errors=[str(e)])


@router.post("/reload", response_model=ConfigResponse)
async def reload_config() -> ConfigResponse:
    """Reload configuration from file."""
    try:
        manager = get_config_manager()
        manager.reload()

        logger.info("Configuration reloaded successfully")

        return ConfigResponse(
            success=True,
            config=_config_to_dict(manager.config),
        )
    except Exception as e:
        logger.error(f"Failed to reload config: {e}")
        return ConfigResponse(success=False, error=str(e))


@router.get("/presets/list", response_model=list[PresetInfo])
async def list_presets() -> list[PresetInfo]:
    """List available configuration presets."""
    presets = [
        PresetInfo(
            name="development",
            description="Development environment with verbose logging",
            modules=["system", "parser"],
        ),
        PresetInfo(
            name="production",
            description="Production environment optimized for performance",
            modules=["system", "security", "execution"],
        ),
        PresetInfo(
            name="conservative",
            description="Conservative execution with maximum retries",
            modules=["execution"],
        ),
        PresetInfo(
            name="aggressive",
            description="Aggressive execution for fast processing",
            modules=["execution"],
        ),
        PresetInfo(
            name="quality_first",
            description="Prioritize quality over cost/speed",
            modules=["parser", "resource_allocator"],
        ),
        PresetInfo(
            name="cost_optimized",
            description="Optimize for lowest cost",
            modules=["budget", "resource_allocator"],
        ),
    ]
    return presets


@router.get("/schema/all", response_model=dict[str, Any])
async def get_config_schema() -> dict[str, Any]:
    """Get the configuration schema for all modules."""
    schema = {
        "system": {
            "fields": [
                {"name": "name", "type": "text"},
                {
                    "name": "environment",
                    "type": "select",
                    "options": ["development", "staging", "production"],
                },
                {"name": "version", "type": "text"},
                {
                    "name": "log_level",
                    "type": "select",
                    "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                },
                {"name": "log_format", "type": "select", "options": ["json", "text"]},
                {"name": "metrics_enabled", "type": "boolean"},
                {"name": "health_check_enabled", "type": "boolean"},
            ]
        },
        "budget": {
            "fields": [
                {"name": "monthly_limit", "type": "number", "min": 0},
                {"name": "daily_limit", "type": "number", "min": 0},
                {"name": "per_task_limit", "type": "number", "min": 0},
                {"name": "auto_throttle_at", "type": "number", "min": 0, "max": 1},
                {"name": "hard_stop_at", "type": "number", "min": 0, "max": 1},
                {
                    "name": "cost_optimization_mode",
                    "type": "select",
                    "options": ["aggressive", "balanced", "quality_first"],
                },
            ]
        },
        "security": {
            "fields": [
                {
                    "name": "sandbox_type",
                    "type": "select",
                    "options": ["gvisor", "docker", "process"],
                },
                {"name": "blockchain_audit_enabled", "type": "boolean"},
                {"name": "encryption_enabled", "type": "boolean"},
                {"name": "seccomp_enabled", "type": "boolean"},
            ]
        },
        "execution": {
            "fields": [
                {"name": "max_parallel_tasks", "type": "number", "min": 1, "max": 100},
                {"name": "genetic_twin_enabled", "type": "boolean"},
                {"name": "self_healing_enabled", "type": "boolean"},
                {"name": "self_healing_max_retries", "type": "number", "min": 0, "max": 10},
                {"name": "checkpoint_interval_seconds", "type": "number", "min": 10, "max": 300},
            ]
        },
        "firewall": {
            "fields": [
                {
                    "name": "strictness",
                    "type": "select",
                    "options": ["low", "medium", "high", "paranoid"],
                },
                {"name": "adversarial_test_enabled", "type": "boolean"},
                {"name": "enable_behavioral_analysis", "type": "boolean"},
                {"name": "enable_semantic_analysis", "type": "boolean"},
            ]
        },
        "parser": {
            "fields": [
                {"name": "default_model", "type": "text"},
                {
                    "name": "routing_strategy",
                    "type": "select",
                    "options": ["speed_first", "cost_optimized", "quality_first"],
                },
                {"name": "extract_implicit_requirements", "type": "boolean"},
                {"name": "confidence_threshold", "type": "number", "min": 0, "max": 1},
            ]
        },
        "strategic_planner": {
            "fields": [
                {"name": "tot_depth", "type": "number", "min": 1, "max": 10},
                {"name": "branching_factor", "type": "number", "min": 2, "max": 8},
                {"name": "mad_debate_rounds", "type": "number", "min": 1, "max": 10},
                {"name": "consensus_threshold", "type": "number", "min": 0, "max": 1},
            ]
        },
        "resource_allocator": {
            "fields": [
                {"name": "tier_1_model", "type": "text"},
                {"name": "tier_2_model", "type": "text"},
                {"name": "tier_3_model", "type": "text"},
                {"name": "allow_local_fallbacks", "type": "boolean"},
                {"name": "cost_tracking_enabled", "type": "boolean"},
            ]
        },
        "tactical_decomposer": {
            "fields": [
                {"name": "max_subtasks", "type": "number", "min": 1, "max": 100},
                {"name": "max_task_size_lines", "type": "number", "min": 50, "max": 2000},
                {"name": "max_task_time_minutes", "type": "number", "min": 1, "max": 60},
                {"name": "enable_smart_batching", "type": "boolean"},
            ]
        },
    }
    return schema


def register_routes(app: Any) -> None:
    """Register config routes with FastAPI app."""
    app.include_router(router)
