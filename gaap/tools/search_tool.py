"""
API Search Tool - Search for API Existence and Documentation
=============================================================

Provides tools for searching API documentation, verifying endpoints,
and checking for deprecation.

Key Components:
    - APISearchTool: Search for API existence
    - APIInfo: Information about an API

Usage:
    from gaap.tools.search_tool import APISearchTool

    tool = APISearchTool()
    info = await tool.search_documentation("requests.get")
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from gaap.core.logging import get_standard_logger as get_logger

logger = get_logger("gaap.tools.search")

try:
    from gaap.tools.library_discoverer import LibraryDiscoverer, LibraryInfo

    LIBRARY_DISCOVERER_AVAILABLE = True
except ImportError:
    LIBRARY_DISCOVERER_AVAILABLE = False
    LibraryDiscoverer = None
    LibraryInfo = None


class APICategory(Enum):
    """Category of API"""

    STANDARD_LIBRARY = auto()
    POPULAR_PACKAGE = auto()
    WEB_API = auto()
    INTERNAL = auto()
    UNKNOWN = auto()


class DeprecationStatus(Enum):
    """Deprecation status of an API"""

    ACTIVE = auto()
    DEPRECATED = auto()
    REMOVED = auto()
    UNKNOWN = auto()


@dataclass
class APIInfo:
    """
    Information about an API.

    Attributes:
        name: Full API name (e.g., "requests.get")
        description: Brief description
        category: API category
        module: Module name
        function: Function/method name
        signature: Function signature if available
        parameters: List of parameter names
        return_type: Return type if known
        deprecation_status: Whether deprecated
        deprecation_message: Deprecation message if deprecated
        examples: Usage examples
        documentation_url: URL to documentation
        version_added: Version when added
        version_deprecated: Version when deprecated
        metadata: Additional metadata
    """

    name: str
    description: str = ""
    category: APICategory = APICategory.UNKNOWN
    module: str = ""
    function: str = ""
    signature: str = ""
    parameters: list[str] = field(default_factory=list)
    return_type: str = ""
    deprecation_status: DeprecationStatus = DeprecationStatus.UNKNOWN
    deprecation_message: str = ""
    examples: list[str] = field(default_factory=list)
    documentation_url: str = ""
    version_added: str = ""
    version_deprecated: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return self.category != APICategory.UNKNOWN

    @property
    def is_deprecated(self) -> bool:
        return self.deprecation_status in (
            DeprecationStatus.DEPRECATED,
            DeprecationStatus.REMOVED,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.name,
            "module": self.module,
            "function": self.function,
            "signature": self.signature,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "deprecation_status": self.deprecation_status.name,
            "is_deprecated": self.is_deprecated,
            "examples": self.examples,
            "documentation_url": self.documentation_url,
        }


@dataclass
class EndpointInfo:
    """Information about an API endpoint"""

    url: str
    method: str
    exists: bool = False
    status_code: int = 0
    description: str = ""
    parameters: list[str] = field(default_factory=list)
    response_format: str = ""
    requires_auth: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


STANDARD_LIBRARY_APIS: dict[str, dict[str, Any]] = {
    "os.path.join": {
        "description": "Join path components",
        "module": "os.path",
        "function": "join",
        "signature": "join(path, *paths)",
        "parameters": ["path", "paths"],
        "return_type": "str",
    },
    "os.path.exists": {
        "description": "Check if path exists",
        "module": "os.path",
        "function": "exists",
        "signature": "exists(path)",
        "parameters": ["path"],
        "return_type": "bool",
    },
    "os.listdir": {
        "description": "List directory contents",
        "module": "os",
        "function": "listdir",
        "signature": "listdir(path='.')",
        "parameters": ["path"],
        "return_type": "list[str]",
    },
    "json.loads": {
        "description": "Parse JSON string",
        "module": "json",
        "function": "loads",
        "signature": "loads(s, *, cls=None, object_hook=None, ...)",
        "parameters": ["s", "cls", "object_hook"],
        "return_type": "Any",
    },
    "json.dumps": {
        "description": "Serialize object to JSON",
        "module": "json",
        "function": "dumps",
        "signature": "dumps(obj, *, skipkeys=False, ensure_ascii=True, ...)",
        "parameters": ["obj", "skipkeys", "ensure_ascii"],
        "return_type": "str",
    },
    "re.match": {
        "description": "Match pattern at string start",
        "module": "re",
        "function": "match",
        "signature": "match(pattern, string, flags=0)",
        "parameters": ["pattern", "string", "flags"],
        "return_type": "Match | None",
    },
    "re.search": {
        "description": "Search pattern in string",
        "module": "re",
        "function": "search",
        "signature": "search(pattern, string, flags=0)",
        "parameters": ["pattern", "string", "flags"],
        "return_type": "Match | None",
    },
    "re.sub": {
        "description": "Replace pattern in string",
        "module": "re",
        "function": "sub",
        "signature": "sub(pattern, repl, string, count=0, flags=0)",
        "parameters": ["pattern", "repl", "string", "count", "flags"],
        "return_type": "str",
    },
    "math.sqrt": {
        "description": "Calculate square root",
        "module": "math",
        "function": "sqrt",
        "signature": "sqrt(x)",
        "parameters": ["x"],
        "return_type": "float",
    },
    "math.pow": {
        "description": "Calculate x raised to power y",
        "module": "math",
        "function": "pow",
        "signature": "pow(x, y)",
        "parameters": ["x", "y"],
        "return_type": "float",
    },
    "math.sin": {
        "description": "Calculate sine",
        "module": "math",
        "function": "sin",
        "signature": "sin(x)",
        "parameters": ["x"],
        "return_type": "float",
    },
    "math.cos": {
        "description": "Calculate cosine",
        "module": "math",
        "function": "cos",
        "signature": "cos(x)",
        "parameters": ["x"],
        "return_type": "float",
    },
    "asyncio.run": {
        "description": "Run async coroutine",
        "module": "asyncio",
        "function": "run",
        "signature": "run(main, *, debug=None)",
        "parameters": ["main", "debug"],
        "return_type": "Any",
        "version_added": "3.7",
    },
    "asyncio.gather": {
        "description": "Gather multiple coroutines",
        "module": "asyncio",
        "function": "gather",
        "signature": "gather(*coros_or_futures, return_exceptions=False)",
        "parameters": ["coros_or_futures", "return_exceptions"],
        "return_type": "list[Any]",
    },
    "functools.lru_cache": {
        "description": "LRU cache decorator",
        "module": "functools",
        "function": "lru_cache",
        "signature": "lru_cache(user_function=None, /, maxsize=128, typed=False)",
        "parameters": ["user_function", "maxsize", "typed"],
        "return_type": "Callable",
    },
    "itertools.chain": {
        "description": "Chain iterables",
        "module": "itertools",
        "function": "chain",
        "signature": "chain(*iterables)",
        "parameters": ["iterables"],
        "return_type": "Iterator",
    },
    "collections.defaultdict": {
        "description": "Dictionary with default factory",
        "module": "collections",
        "function": "defaultdict",
        "signature": "defaultdict(default_factory=None, /, ...)",
        "parameters": ["default_factory"],
        "return_type": "dict",
    },
    "collections.Counter": {
        "description": "Count hashable objects",
        "module": "collections",
        "function": "Counter",
        "signature": "Counter(iterable=None, /, **kwds)",
        "parameters": ["iterable", "kwds"],
        "return_type": "dict",
    },
}

POPULAR_PACKAGE_APIS: dict[str, dict[str, Any]] = {
    "requests.get": {
        "description": "Send HTTP GET request",
        "module": "requests",
        "function": "get",
        "signature": "get(url, params=None, **kwargs)",
        "parameters": ["url", "params"],
        "return_type": "Response",
    },
    "requests.post": {
        "description": "Send HTTP POST request",
        "module": "requests",
        "function": "post",
        "signature": "post(url, data=None, json=None, **kwargs)",
        "parameters": ["url", "data", "json"],
        "return_type": "Response",
    },
    "requests.Session": {
        "description": "HTTP session with connection pooling",
        "module": "requests",
        "function": "Session",
        "signature": "Session()",
        "parameters": [],
        "return_type": "Session",
    },
    "numpy.array": {
        "description": "Create NumPy array",
        "module": "numpy",
        "function": "array",
        "signature": "array(object, dtype=None, copy=True, ...)",
        "parameters": ["object", "dtype", "copy"],
        "return_type": "ndarray",
    },
    "numpy.zeros": {
        "description": "Create array of zeros",
        "module": "numpy",
        "function": "zeros",
        "signature": "zeros(shape, dtype=float, order='C')",
        "parameters": ["shape", "dtype", "order"],
        "return_type": "ndarray",
    },
    "numpy.mean": {
        "description": "Calculate mean",
        "module": "numpy",
        "function": "mean",
        "signature": "mean(a, axis=None, dtype=None, ...)",
        "parameters": ["a", "axis", "dtype"],
        "return_type": "float | ndarray",
    },
    "pandas.DataFrame": {
        "description": "Create pandas DataFrame",
        "module": "pandas",
        "function": "DataFrame",
        "signature": "DataFrame(data=None, index=None, columns=None, ...)",
        "parameters": ["data", "index", "columns"],
        "return_type": "DataFrame",
    },
    "pandas.read_csv": {
        "description": "Read CSV file into DataFrame",
        "module": "pandas",
        "function": "read_csv",
        "signature": "read_csv(filepath_or_buffer, sep=',', ...)",
        "parameters": ["filepath_or_buffer", "sep"],
        "return_type": "DataFrame",
    },
    "flask.Flask": {
        "description": "Create Flask application",
        "module": "flask",
        "function": "Flask",
        "signature": "Flask(import_name, static_url_path=None, ...)",
        "parameters": ["import_name", "static_url_path"],
        "return_type": "Flask",
    },
    "fastapi.FastAPI": {
        "description": "Create FastAPI application",
        "module": "fastapi",
        "function": "FastAPI",
        "signature": "FastAPI(*, debug=False, title='FastAPI', ...)",
        "parameters": ["debug", "title"],
        "return_type": "FastAPI",
    },
    "httpx.get": {
        "description": "Async HTTP GET request",
        "module": "httpx",
        "function": "get",
        "signature": "get(url, *, params=None, headers=None, ...)",
        "parameters": ["url", "params", "headers"],
        "return_type": "Response",
    },
    "aiohttp.ClientSession": {
        "description": "Async HTTP client session",
        "module": "aiohttp",
        "function": "ClientSession",
        "signature": "ClientSession(*, base_url=None, ...)",
        "parameters": ["base_url"],
        "return_type": "ClientSession",
    },
}

DEPRECATED_APIS: dict[str, dict[str, Any]] = {
    "os.system": {
        "deprecation_status": DeprecationStatus.DEPRECATED,
        "deprecation_message": "Use subprocess module instead for better security",
        "version_deprecated": "3.0",
    },
    "threading.Thread.setDaemon": {
        "deprecation_status": DeprecationStatus.DEPRECATED,
        "deprecation_message": "Use daemon property instead",
        "version_deprecated": "3.9",
    },
    "asyncio.Task.all_tasks": {
        "deprecation_status": DeprecationStatus.DEPRECATED,
        "deprecation_message": "Use asyncio.all_tasks instead",
        "version_deprecated": "3.9",
    },
    "typing.Text": {
        "deprecation_status": DeprecationStatus.DEPRECATED,
        "deprecation_message": "Use str instead",
        "version_deprecated": "3.11",
    },
}


class APISearchTool:
    """
    Search for API existence and documentation.

    Provides:
    - API documentation search
    - Endpoint verification
    - Deprecation checking
    - Usage examples

    Attributes:
        use_web_search: Whether to use web search for unknown APIs
        library_discoverer: Library discoverer for package info
    """

    def __init__(self, use_web_search: bool = False):
        self.use_web_search = use_web_search
        self._logger = logger
        self._library_discoverer: Any = None

        if LIBRARY_DISCOVERER_AVAILABLE and LibraryDiscoverer is not None:
            self._library_discoverer = LibraryDiscoverer()

        self._cache: dict[str, APIInfo] = {}

    def _parse_api_name(self, api_name: str) -> tuple[str, str]:
        """Parse API name into module and function"""
        parts = api_name.rsplit(".", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return "", parts[0]

    def _create_api_info(
        self,
        api_name: str,
        data: dict[str, Any],
        category: APICategory,
    ) -> APIInfo:
        """Create APIInfo from data dictionary"""
        module, function = self._parse_api_name(api_name)

        info = APIInfo(
            name=api_name,
            description=data.get("description", ""),
            category=category,
            module=data.get("module", module),
            function=data.get("function", function),
            signature=data.get("signature", ""),
            parameters=data.get("parameters", []),
            return_type=data.get("return_type", ""),
            examples=data.get("examples", []),
            documentation_url=data.get("documentation_url", ""),
            version_added=data.get("version_added", ""),
            version_deprecated=data.get("version_deprecated", ""),
        )

        if api_name in DEPRECATED_APIS:
            dep_data = DEPRECATED_APIS[api_name]
            info.deprecation_status = dep_data.get(
                "deprecation_status", DeprecationStatus.DEPRECATED
            )
            info.deprecation_message = dep_data.get("deprecation_message", "")
            info.version_deprecated = dep_data.get("version_deprecated", "")

        return info

    async def search_documentation(self, api_name: str) -> APIInfo:
        """
        Search for API documentation.

        Args:
            api_name: Full API name (e.g., "requests.get")

        Returns:
            APIInfo with documentation
        """
        if api_name in self._cache:
            return self._cache[api_name]

        if api_name in STANDARD_LIBRARY_APIS:
            info = self._create_api_info(
                api_name, STANDARD_LIBRARY_APIS[api_name], APICategory.STANDARD_LIBRARY
            )
            self._cache[api_name] = info
            return info

        if api_name in POPULAR_PACKAGE_APIS:
            info = self._create_api_info(
                api_name, POPULAR_PACKAGE_APIS[api_name], APICategory.POPULAR_PACKAGE
            )
            self._cache[api_name] = info
            return info

        module_name = api_name.split(".")[0] if "." in api_name else api_name

        if self._library_discoverer:
            try:
                library_info = await self._library_discoverer.search_library(module_name)
                if library_info:
                    info = APIInfo(
                        name=api_name,
                        description=library_info.description,
                        category=APICategory.POPULAR_PACKAGE,
                        module=module_name,
                        documentation_url=library_info.documentation_url,
                        metadata={
                            "library_info": (
                                library_info.__dict__ if hasattr(library_info, "__dict__") else {}
                            )
                        },
                    )
                    self._cache[api_name] = info
                    return info
            except Exception as e:
                self._logger.debug(f"Library discoverer search failed: {e}")

        return APIInfo(
            name=api_name,
            category=APICategory.UNKNOWN,
        )

    async def verify_endpoint(
        self,
        url: str,
        method: str = "GET",
        timeout: float = 5.0,
    ) -> EndpointInfo:
        """
        Verify if an endpoint exists.

        Args:
            url: Endpoint URL
            method: HTTP method
            timeout: Request timeout

        Returns:
            EndpointInfo with verification result
        """
        endpoint_info = EndpointInfo(
            url=url,
            method=method.upper(),
        )

        if method.upper() not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
            endpoint_info.description = f"Unknown HTTP method: {method}"
            return endpoint_info

        url_pattern = re.compile(
            r"^https?://"
            r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"
            r"[A-Z]{2,6}"
            r"(?::[0-9]+)?"
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )

        if not url_pattern.match(url):
            endpoint_info.description = "Invalid URL format"
            return endpoint_info

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.head(
                    url, timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    endpoint_info.exists = True
                    endpoint_info.status_code = response.status
                    endpoint_info.description = f"Endpoint responded with status {response.status}"

        except ImportError:
            endpoint_info.exists = True
            endpoint_info.description = "URL format valid (aiohttp not available for verification)"
        except asyncio.TimeoutError:
            endpoint_info.exists = False
            endpoint_info.description = f"Endpoint timed out after {timeout}s"
        except Exception as e:
            endpoint_info.exists = False
            endpoint_info.description = f"Endpoint verification failed: {str(e)}"

        return endpoint_info

    async def check_deprecation(self, api_name: str) -> tuple[bool, str]:
        """
        Check if an API is deprecated.

        Args:
            api_name: Full API name

        Returns:
            Tuple of (is_deprecated, deprecation_message)
        """
        if api_name in DEPRECATED_APIS:
            dep_data = DEPRECATED_APIS[api_name]
            return True, dep_data.get("deprecation_message", "API is deprecated")

        info = await self.search_documentation(api_name)
        return info.is_deprecated, info.deprecation_message

    async def get_api_examples(self, api_name: str) -> list[str]:
        """
        Get usage examples for an API.

        Args:
            api_name: Full API name

        Returns:
            List of usage examples
        """
        info = await self.search_documentation(api_name)

        if info.examples:
            return info.examples

        examples: list[str] = []

        if info.category == APICategory.STANDARD_LIBRARY:
            if "json" in info.module:
                if info.function == "loads":
                    examples.append('import json\nresult = json.loads(\'{"key": "value"}\')')
                elif info.function == "dumps":
                    examples.append('import json\nresult = json.dumps({"key": "value"})')
            elif "re" in info.module:
                if info.function == "match":
                    examples.append('import re\nmatch = re.match(r"\\d+", "123abc")')
                elif info.function == "search":
                    examples.append('import re\nmatch = re.search(r"\\d+", "abc123")')
            elif "os.path" in info.module:
                if info.function == "join":
                    examples.append('import os\npath = os.path.join("dir", "file.txt")')
                elif info.function == "exists":
                    examples.append('import os\nexists = os.path.exists("/path/to/file")')

        elif info.category == APICategory.POPULAR_PACKAGE:
            if "requests" in info.module:
                if info.function == "get":
                    examples.append(
                        'import requests\nresponse = requests.get("https://api.example.com")'
                    )
                elif info.function == "post":
                    examples.append(
                        'import requests\nresponse = requests.post("https://api.example.com", json={"key": "value"})'
                    )
            elif "numpy" in info.module:
                if info.function == "array":
                    examples.append("import numpy as np\narr = np.array([1, 2, 3])")
            elif "pandas" in info.module:
                if info.function == "DataFrame":
                    examples.append('import pandas as pd\ndf = pd.DataFrame({"col": [1, 2, 3]})')

        return examples

    async def search_multiple(self, api_names: list[str]) -> dict[str, APIInfo]:
        """
        Search for multiple APIs at once.

        Args:
            api_names: List of API names

        Returns:
            Dictionary mapping API names to APIInfo
        """
        results = {}
        for api_name in api_names:
            results[api_name] = await self.search_documentation(api_name)
        return results

    def add_known_api(
        self,
        api_name: str,
        description: str,
        category: APICategory = APICategory.POPULAR_PACKAGE,
        **kwargs: Any,
    ) -> None:
        """
        Add a known API to the cache.

        Args:
            api_name: Full API name
            description: API description
            category: API category
            **kwargs: Additional API info fields
        """
        info = APIInfo(
            name=api_name,
            description=description,
            category=category,
            **kwargs,
        )
        self._cache[api_name] = info


def create_api_search_tool(use_web_search: bool = False) -> APISearchTool:
    """Create an API search tool with default settings"""
    return APISearchTool(use_web_search=use_web_search)
