from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Chat request model"""

    message: str
    model: str | None = None
    provider: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False


class ChatResponse(BaseModel):
    """Chat response model"""

    success: bool
    message: str
    response: str | None = None
    error: str | None = None
    model: str | None = None
    provider: str | None = None
    tokens_used: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None


class ExecuteRequest(BaseModel):
    """Execute request model"""

    task: str
    priority: str = "normal"
    category: str | None = None


class ExecuteResponse(BaseModel):
    """Execute response model"""

    success: bool
    task_id: str
    result: Any | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """Health response model"""

    status: str
    timestamp: str
    version: str = "1.0.0"


class StatusResponse(BaseModel):
    """Status response model"""

    providers: dict[str, bool]
    memory: dict[str, Any]
    cache: dict[str, Any]
    stats: dict[str, Any]


class ProviderInfo(BaseModel):
    """Provider info model"""

    name: str
    available: bool
    models: list[str]
    default_model: str | None = None


class MemorySaveRequest(BaseModel):
    """Memory save request"""

    path: str


class MemoryLoadRequest(BaseModel):
    """Memory load request"""

    path: str


class CacheStatsResponse(BaseModel):
    """Cache stats response"""

    backend: str
    stats: dict[str, Any]


class ConfigGetResponse(BaseModel):
    """Config get response"""

    config: dict[str, Any]


class ConfigSetRequest(BaseModel):
    """Config set request"""

    key: str
    value: Any
