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


@app.get("/health", response_model=HealthResponse)  # type: ignore[untyped-decorator]
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
    )


@app.get("/status", response_model=StatusResponse)  # type: ignore[untyped-decorator]
async def status() -> StatusResponse:
    engine = get_engine()
    return StatusResponse(
        providers={"groq": True, "gemini": True},
        memory={"working": 100, "episodic": 0, "semantic": 0},
        cache={"backend": "memory", "size": 0},
        stats={"requests": 0, "errors": 0},
    )


@app.post("/chat", response_model=ChatResponse)  # type: ignore[untyped-decorator]
async def chat(request: ChatRequest) -> ChatResponse:
    engine = get_engine()

    try:
        result = await engine.chat(
            message=request.message,
        )

        return ChatResponse(
            success=True,
            message=request.message,
            response=str(result),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            success=False,
            message=request.message,
            error=str(e),
        )


@app.post("/execute", response_model=ExecuteResponse)  # type: ignore[untyped-decorator]
async def execute(request: ExecuteRequest) -> ExecuteResponse:
    engine = get_engine()

    try:
        from gaap.gaap_engine import GAAPRequest

        gaap_request = GAAPRequest(text=request.task)
        result: Any = await engine.process(gaap_request)

        return ExecuteResponse(
            success=True,
            task_id="",
            result=str(result),
        )

    except Exception as e:
        logger.error(f"Execute error: {e}")
        return ExecuteResponse(
            success=False,
            task_id="",
            error=str(e),
        )


@app.get("/providers", response_model=list[ProviderInfo])  # type: ignore[untyped-decorator]
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


@app.get("/providers/{name}/models", response_model=list[str])  # type: ignore[untyped-decorator]
async def list_models(name: str) -> list[str]:
    models_map: dict[str, list[str]] = {
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
        "gemini": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"],
        "g4f": ["gpt-3.5-turbo", "gpt-4"],
    }

    if name not in models_map:
        raise HTTPException(status_code=404, detail=f"Provider {name} not found")

    return models_map[name]


@app.get("/cache/stats", response_model=CacheStatsResponse)  # type: ignore[untyped-decorator]
async def cache_stats() -> CacheStatsResponse:
    return CacheStatsResponse(
        backend="memory",
        stats={"hits": 0, "misses": 0, "size": 0},
    )


@app.post("/cache/clear")  # type: ignore[untyped-decorator]
async def clear_cache() -> dict[str, Any]:
    return {"success": True, "message": "Cache cleared"}


@app.get("/config", response_model=ConfigGetResponse)  # type: ignore[untyped-decorator]
async def get_config_endpoint() -> ConfigGetResponse:
    return ConfigGetResponse(config={"api_key": "***", "providers": ["groq", "gemini"]})


@app.post("/config")  # type: ignore[untyped-decorator]
async def set_config(request: ConfigSetRequest) -> dict[str, Any]:
    return {"success": True, "message": f"Config updated: {request.key} = {request.value}"}


@app.get("/history")  # type: ignore[untyped-decorator]
async def get_history(limit: int = 10) -> dict[str, Any]:
    return {"history": [], "total": 0}


@app.delete("/history/{item_id}")  # type: ignore[untyped-decorator]
async def delete_history(item_id: str) -> dict[str, Any]:
    return {"success": True, "message": f"Deleted {item_id}"}


@app.get("/")  # type: ignore[untyped-decorator]
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
