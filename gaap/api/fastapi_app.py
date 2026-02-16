import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from gaap.api.models import (
    CacheStatsResponse,
    ChatRequest,
    ChatResponse,
    ConfigGetResponse,
    ConfigSetRequest,
    ExecuteRequest,
    ExecuteResponse,
    HealthResponse,
    ProviderInfo,
    StatusResponse,
)
from gaap.gaap_engine import GAAPEngine, create_engine
from gaap.providers import get_provider

logger = logging.getLogger("gaap.api.fastapi")

_engine: GAAPEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _engine
    _engine = create_engine()
    yield
    if _engine:
        _engine.shutdown()


app = FastAPI(
    title="GAAP API",
    description="Generic Adaptive AI Platform - REST API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> GAAPEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    engine = get_engine()
    return StatusResponse(
        providers={"groq": True, "gemini": True},
        memory={"working": 100, "episodic": 0, "semantic": 0},
        cache={"backend": "memory", "size": 0},
        stats={"requests": 0, "errors": 0},
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    engine = get_engine()

    try:
        result = await engine.chat(
            message=request.message,
            model=request.model,
            provider=request.provider,
        )

        return ChatResponse(
            success=True,
            message=request.message,
            response=result.get("response"),
            model=result.get("model"),
            provider=result.get("provider"),
            tokens_used=result.get("tokens_used"),
            cost_usd=result.get("cost_usd"),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            success=False,
            message=request.message,
            error=str(e),
        )


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest) -> ExecuteResponse:
    engine = get_engine()

    try:
        result = await engine.execute(task=request.task, priority=request.priority)

        return ExecuteResponse(
            success=True,
            task_id=result.get("task_id", ""),
            result=result.get("output"),
        )

    except Exception as e:
        logger.error(f"Execute error: {e}")
        return ExecuteResponse(
            success=False,
            task_id="",
            error=str(e),
        )


@app.get("/providers", response_model=list[ProviderInfo])
async def list_providers() -> list[ProviderInfo]:
    providers: list[ProviderInfo] = []
    for name in ["groq", "gemini", "g4f"]:
        try:
            provider = get_provider(name)
            providers.append(
                ProviderInfo(
                    name=name,
                    available=provider is not None,
                    models=["llama-3.3-70b-versatile", "gemini-2.0-flash"],
                )
            )
        except Exception:
            providers.append(ProviderInfo(name=name, available=False, models=[]))

    return providers


@app.get("/providers/{name}/models", response_model=list[str])
async def list_models(name: str) -> list[str]:
    models_map: dict[str, list[str]] = {
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        "gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"],
        "g4f": ["gpt-3.5-turbo", "gpt-4"],
    }

    if name not in models_map:
        raise HTTPException(status_code=404, detail=f"Provider {name} not found")

    return models_map[name]


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats() -> CacheStatsResponse:
    return CacheStatsResponse(
        backend="memory",
        stats={"hits": 0, "misses": 0, "size": 0},
    )


@app.post("/cache/clear")
async def clear_cache() -> dict[str, Any]:
    return {"success": True, "message": "Cache cleared"}


@app.get("/config", response_model=ConfigGetResponse)
async def get_config_endpoint() -> ConfigGetResponse:
    return ConfigGetResponse(config={"api_key": "***", "providers": ["groq", "gemini"]})


@app.post("/config")
async def set_config(request: ConfigSetRequest) -> dict[str, Any]:
    return {"success": True, "message": f"Config updated: {request.key} = {request.value}"}


@app.get("/history")
async def get_history(limit: int = 10) -> dict[str, Any]:
    return {"history": [], "total": 0}


@app.delete("/history/{item_id}")
async def delete_history(item_id: str) -> dict[str, Any]:
    return {"success": True, "message": f"Deleted {item_id}"}


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "message": "GAAP API v1.0.0",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/status",
            "/chat",
            "/execute",
            "/providers",
            "/cache/stats",
            "/config",
            "/history",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
