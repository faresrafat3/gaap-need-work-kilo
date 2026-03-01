"""
Knowledge API - Repository Learning Endpoints
==========================================

Provides endpoints for:
- Library ingestion (learn from repos)
- AST parsing (extract structure)
- Usage mining (find patterns)
- Cheat sheet generation
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gaap.knowledge import (
    KnowledgeIngestion,
)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class ParseCodeRequest(BaseModel):
    """Request for AST parsing."""

    code: str = Field(..., min_length=1)
    file_path: Optional[str] = None


class MineUsageRequest(BaseModel):
    """Request for usage mining."""

    code: str = Field(..., min_length=1)
    target_function: Optional[str] = None


class CheatSheetRequest(BaseModel):
    """Request for cheat sheet generation."""

    library_name: str
    functions: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    """Request for library ingestion."""

    repo_url: str
    library_name: str


@router.post("/parse")
async def parse_code(request: ParseCodeRequest) -> dict:
    """Parse code and extract AST structure."""
    try:
        import ast

        tree = ast.parse(request.code)

        classes = []
        functions = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(
                    {
                        "name": node.name,
                        "bases": [b.id for b in node.bases if isinstance(b, ast.Name)],
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        "docstring": ast.get_docstring(node),
                    }
                )
            elif isinstance(node, ast.FunctionDef):
                functions.append(
                    {
                        "name": node.name,
                        "params": [a.arg for a in node.args.args],
                        "docstring": ast.get_docstring(node),
                    }
                )
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                else:
                    imports.append(f"{node.module or ''}")

        return {
            "file": request.file_path,
            "classes": classes,
            "functions": functions,
            "imports": imports,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mine")
async def mine_usage(request: MineUsageRequest) -> dict:
    """Mine usage patterns from code."""
    try:
        import ast

        code = request.code
        target = request.target_function

        # Find all function calls in the code
        patterns = []

        tree = ast.parse(code)

        for node in ast.walk(tree):
            # Find function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    patterns.append(
                        {
                            "type": "function_call",
                            "example": (
                                ast.unparse(node) if hasattr(ast, "unparse") else str(node.func.id)
                            ),
                            "context": f"Calling {node.func.id}",
                        }
                    )

        # Simple pattern detection
        if target:
            # Find all usages of the target function
            for line_num, line in enumerate(code.split("\n"), 1):
                if target in line:
                    patterns.append(
                        {
                            "type": "usage",
                            "example": line.strip(),
                            "context": f"Line {line_num}",
                        }
                    )

        return {
            "target": target or "all",
            "patterns": patterns[:10],  # Limit to 10 patterns
            "score": len(patterns) / 10 if patterns else 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cheatsheet")
async def generate_cheatsheet(request: CheatSheetRequest) -> dict:
    """Generate cheat sheet for a library."""
    try:
        library = request.library_name
        functions = request.functions

        # Generate a simple cheatsheet based on library name
        # In production, this would use AI or load from knowledge base

        cheat_sheet_functions = []

        # Common Python libraries
        common_functions = {
            "os": [
                {
                    "name": "os.path.join(*paths)",
                    "signature": "join(*paths)",
                    "description": "Join path components",
                    "example": "os.path.join('dir', 'file.py')",
                },
                {
                    "name": "os.path.exists(path)",
                    "signature": "exists(path)",
                    "description": "Check if path exists",
                    "example": "os.path.exists('/tmp/file')",
                },
                {
                    "name": "os.listdir(path)",
                    "signature": "listdir(path)",
                    "description": "List directory contents",
                    "example": "os.listdir('.')",
                },
            ],
            "json": [
                {
                    "name": "json.dumps(obj)",
                    "signature": "dumps(obj)",
                    "description": "Convert to JSON string",
                    "example": "json.dumps({'a': 1})",
                },
                {
                    "name": "json.loads(s)",
                    "signature": "loads(s)",
                    "description": "Parse JSON string",
                    "example": "json.loads('{\"a\": 1}')",
                },
            ],
            "requests": [
                {
                    "name": "requests.get(url)",
                    "signature": "get(url)",
                    "description": "HTTP GET request",
                    "example": "requests.get('https://api.example.com')",
                },
                {
                    "name": "requests.post(url, json=data)",
                    "signature": "post(url, json=data)",
                    "description": "HTTP POST request",
                    "example": "requests.post(url, json={'key': 'value'})",
                },
            ],
        }

        # Get functions for the library or use provided
        if library.lower() in common_functions:
            cheat_sheet_functions = common_functions[library.lower()]
        else:
            # Generate generic response
            cheat_sheet_functions = [
                {
                    "name": f"{library}.{func}",
                    "signature": func,
                    "description": f"Function from {library}",
                    "example": f"{library}.{func}()",
                }
                for func in (functions or ["main", "init"])
            ]

        return {
            "library": library,
            "functions": cheat_sheet_functions,
            "patterns": [
                {"name": "Import", "example": f"import {library}"},
                {"name": "Usage", "example": f"{library}.function()"},
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def ingest_library(request: IngestRequest) -> dict:
    """Ingest a library from repository."""
    try:
        ingestion = KnowledgeIngestion()
        result = await ingestion.ingest_repo(request.repo_url, request.library_name)

        return {
            "library": request.library_name,
            "status": "completed",
            "files_parsed": result.files_parsed,
            "functions_found": result.functions_found,
            "classes_found": result.classes_found,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
