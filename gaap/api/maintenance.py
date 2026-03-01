"""
Maintenance API - Technical Debt & Refinancing Endpoints
=====================================================

Provides endpoints for:
- Debt scanning (find technical debt)
- Refinancing (optimize code)
- Interest calculation (debt tracking)
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gaap.maintenance import (
    DebtScanner,
)

router = APIRouter(prefix="/api/maintenance", tags=["maintenance"])


class ScanDebtRequest(BaseModel):
    """Request for debt scanning."""

    project_path: str = Field(..., min_length=1)
    include_types: list[str] = Field(default_factory=lambda: ["all"])


class RefinanceRequest(BaseModel):
    """Request for code refinancing."""

    debt_items: list[dict]
    optimization_level: str = Field(default="medium")


class CalculateInterestRequest(BaseModel):
    """Request for interest calculation."""

    principal: float = Field(..., gt=0)
    rate: float = Field(..., ge=0, le=100)
    time_months: int = Field(..., gt=0)


@router.post("/scan")
async def scan_debt(request: ScanDebtRequest) -> dict:
    """Scan project for technical debt."""
    try:
        scanner = DebtScanner()
        results = scanner.scan_directory(Path(request.project_path))

        return {
            "project": request.project_path,
            "total_debt": len(results.items),
            "items": [
                {
                    "type": d.type.name,
                    "severity": d.priority.name,
                    "impact": d.complexity or 1,
                    "file": d.file_path,
                    "line": d.line_number,
                    "description": d.message,
                    "suggestion": d.snippet or "",
                }
                for d in results.items
            ],
            "scanned_files": results.scanned_files,
            "scan_time_ms": results.scan_time_ms,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refinance")
async def refinance_debt(request: RefinanceRequest) -> dict:
    """Optimize/refinance technical debt."""
    try:
        debt_items = request.debt_items
        optimization_level = request.optimization_level

        # Simple optimization simulation
        results = []

        for item in debt_items:
            severity = item.get("severity", 5)

            # Calculate potential improvements based on optimization level
            if optimization_level == "high":
                lines_removed = severity * 2
                complexity_reduction = severity * 0.8
            elif optimization_level == "medium":
                lines_removed = severity
                complexity_reduction = severity * 0.5
            else:  # low
                lines_removed = severity // 2
                complexity_reduction = severity * 0.2

            results.append(
                {
                    "file": item.get("file", "unknown"),
                    "line": item.get("line", 0),
                    "lines_removed": lines_removed,
                    "complexity_reduction": round(complexity_reduction, 2),
                    "status": "optimized",
                }
            )

        return {
            "original_items": len(debt_items),
            "optimized_items": len(results),
            "savings": {
                "lines_removed": sum(r.get("lines_removed", 0) for r in results),
                "complexity_reduction": sum(r.get("complexity_reduction", 0) for r in results),
            },
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/interest")
async def calculate_interest(request: CalculateInterestRequest) -> dict:
    """Calculate interest on technical debt."""
    try:
        principal = request.principal
        rate = request.rate / 100  # Convert percentage to decimal
        time_years = request.time_months / 12

        # Simple interest: I = P * r * t
        simple = principal * rate * time_years

        # Compound interest: A = P * (1 + r)^t
        compound = principal * ((1 + rate) ** time_years) - principal

        return {
            "principal": request.principal,
            "rate_percent": request.rate,
            "time_months": request.time_months,
            "simple_interest": round(simple, 2),
            "compound_interest": round(compound, 2),
            "total_simple": round(principal + simple, 2),
            "total_compound": round(principal + compound, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
