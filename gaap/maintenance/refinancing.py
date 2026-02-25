"""
Refinancing Engine - Refactoring Proposals
==========================================

Generates and manages refactoring proposals for technical debt.
Uses LLM to suggest fixes and creates branches for safe refactoring.

Implements: docs/evolution_plan_2026/29_TECHNICAL_DEBT_AGENT.md

SAFETY AXIOM: Never push to main! Always work on a side branch.
"""

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from gaap.maintenance.debt_config import DebtConfig, DebtPriority, ProposalStatus
from gaap.maintenance.debt_scanner import DebtItem

logger = logging.getLogger("gaap.maintenance.refinancing")


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        ...


@dataclass
class RefactoringProposal:
    """A proposal for refactoring a debt item."""

    id: str
    debt_item: DebtItem
    proposed_fix: str
    branch_name: str
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime | None = None
    llm_generated: bool = False
    confidence: float = 0.0
    test_results: dict[str, Any] | None = None
    review_notes: str | None = None
    applied_changes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "debt_item": self.debt_item.to_dict(),
            "proposed_fix": self.proposed_fix,
            "branch_name": self.branch_name,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "llm_generated": self.llm_generated,
            "confidence": self.confidence,
            "test_results": self.test_results,
            "review_notes": self.review_notes,
            "applied_changes": self.applied_changes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RefactoringProposal":
        return cls(
            id=data["id"],
            debt_item=DebtItem.from_dict(data["debt_item"]),
            proposed_fix=data["proposed_fix"],
            branch_name=data["branch_name"],
            status=ProposalStatus[data["status"]],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at")
            else None,
            llm_generated=data.get("llm_generated", False),
            confidence=data.get("confidence", 0.0),
            test_results=data.get("test_results"),
            review_notes=data.get("review_notes"),
            applied_changes=data.get("applied_changes", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RefinancingResult:
    """Result of a refinancing operation."""

    success: bool
    proposal: RefactoringProposal | None = None
    branch_created: bool = False
    tests_passed: bool | None = None
    error: str | None = None
    output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "proposal": self.proposal.to_dict() if self.proposal else None,
            "branch_created": self.branch_created,
            "tests_passed": self.tests_passed,
            "error": self.error,
            "output": self.output,
        }


class RefinancingEngine:
    """
    Manages refactoring proposals for technical debt.

    Features:
    - LLM-powered fix suggestions
    - Automatic branch creation
    - Test execution
    - Safe refactoring workflow

    SAFETY: Never pushes to main branch!
    """

    def __init__(
        self,
        config: DebtConfig | None = None,
        llm_provider: LLMProvider | None = None,
        project_root: Path | str | None = None,
    ):
        self._config = config or DebtConfig()
        self._llm = llm_provider
        self._project_root = Path(project_root) if project_root else Path.cwd()
        self._logger = logger

        self._proposals: dict[str, RefactoringProposal] = {}
        self._git_available = self._check_git()

        self._load_proposals()

    @property
    def config(self) -> DebtConfig:
        return self._config

    async def propose(
        self,
        debt: DebtItem,
        use_llm: bool = True,
    ) -> RefactoringProposal:
        """
        Create a refactoring proposal for a debt item.

        Args:
            debt: The debt item to propose a fix for
            use_llm: Whether to use LLM for generating the fix

        Returns:
            RefactoringProposal with suggested fix
        """
        proposal_id = self._generate_proposal_id(debt)
        branch_name = self._generate_branch_name(debt)

        proposed_fix = ""
        confidence = 0.0
        llm_generated = False

        if use_llm and self._llm and self._config.llm_enabled:
            try:
                context = self._get_code_context(debt)
                proposed_fix = await self._generate_llm_fix(debt, context)
                llm_generated = True
                confidence = 0.7
            except Exception as e:
                self._logger.warning(f"LLM proposal generation failed: {e}")
                proposed_fix = self._generate_template_fix(debt)
                confidence = 0.3
        else:
            proposed_fix = self._generate_template_fix(debt)
            confidence = 0.3

        proposal = RefactoringProposal(
            id=proposal_id,
            debt_item=debt,
            proposed_fix=proposed_fix,
            branch_name=branch_name,
            status=ProposalStatus.PENDING,
            llm_generated=llm_generated,
            confidence=confidence,
        )

        self._proposals[proposal_id] = proposal
        self._save_proposals()

        self._logger.info(f"Created proposal {proposal_id} for debt {debt.id}")

        return proposal

    def create_branch(self, proposal: RefactoringProposal) -> RefinancingResult:
        """
        Create a git branch for the proposal.

        Args:
            proposal: The proposal to create a branch for

        Returns:
            RefinancingResult with branch creation status
        """
        if not self._git_available:
            return RefinancingResult(
                success=False,
                proposal=proposal,
                error="Git is not available",
            )

        if self._config.never_push_to_main:
            current_branch = self._get_current_branch()
            if current_branch == "main" or current_branch == "master":
                pass

        try:
            result = subprocess.run(
                ["git", "checkout", "-b", proposal.branch_name],
                cwd=self._project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                proposal.status = ProposalStatus.IN_PROGRESS
                proposal.updated_at = datetime.now()
                self._save_proposals()

                return RefinancingResult(
                    success=True,
                    proposal=proposal,
                    branch_created=True,
                    output=result.stdout,
                )
            else:
                return RefinancingResult(
                    success=False,
                    proposal=proposal,
                    error=result.stderr,
                )
        except Exception as e:
            return RefinancingResult(
                success=False,
                proposal=proposal,
                error=str(e),
            )

    def apply_fix(self, proposal: RefactoringProposal) -> RefinancingResult:
        """
        Apply the proposed fix to the codebase.

        Args:
            proposal: The proposal to apply

        Returns:
            RefinancingResult with application status
        """
        debt = proposal.debt_item
        file_path = self._project_root / debt.file_path

        if not file_path.exists():
            return RefinancingResult(
                success=False,
                proposal=proposal,
                error=f"File not found: {debt.file_path}",
            )

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.splitlines(keepends=True)

            if debt.line_number <= len(lines):
                old_line = lines[debt.line_number - 1]
                indent = len(old_line) - len(old_line.lstrip())

                if debt.type.name == "TODO" or debt.type.name == "FIXME":
                    new_line = " " * indent + proposal.proposed_fix + "\n"
                    lines[debt.line_number - 1] = new_line
                    proposal.applied_changes.append(
                        {
                            "type": "line_replace",
                            "line": debt.line_number,
                            "old": old_line.rstrip(),
                            "new": new_line.rstrip(),
                        }
                    )

            new_content = "".join(lines)
            file_path.write_text(new_content, encoding="utf-8")

            proposal.status = ProposalStatus.READY_FOR_REVIEW
            proposal.updated_at = datetime.now()
            self._save_proposals()

            return RefinancingResult(
                success=True,
                proposal=proposal,
            )

        except Exception as e:
            return RefinancingResult(
                success=False,
                proposal=proposal,
                error=str(e),
            )

    def run_tests(self, proposal: RefactoringProposal) -> RefinancingResult:
        """
        Run tests to verify the fix.

        Args:
            proposal: The proposal to test

        Returns:
            RefinancingResult with test results
        """
        try:
            result = subprocess.run(
                ["pytest", "tests/", "-v", "--tb=short", "-x"],
                cwd=self._project_root,
                capture_output=True,
                text=True,
                timeout=300,
            )

            tests_passed = result.returncode == 0

            proposal.test_results = {
                "passed": tests_passed,
                "output": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
                "return_code": result.returncode,
            }
            proposal.updated_at = datetime.now()
            self._save_proposals()

            return RefinancingResult(
                success=tests_passed,
                proposal=proposal,
                tests_passed=tests_passed,
                output=result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
            )

        except subprocess.TimeoutExpired:
            return RefinancingResult(
                success=False,
                proposal=proposal,
                tests_passed=False,
                error="Test execution timed out",
            )
        except Exception as e:
            return RefinancingResult(
                success=False,
                proposal=proposal,
                tests_passed=False,
                error=str(e),
            )

    def get_proposal(self, proposal_id: str) -> RefactoringProposal | None:
        """Get a proposal by ID."""
        return self._proposals.get(proposal_id)

    def list_proposals(
        self,
        status: ProposalStatus | None = None,
    ) -> list[RefactoringProposal]:
        """List all proposals, optionally filtered by status."""
        proposals = list(self._proposals.values())

        if status:
            proposals = [p for p in proposals if p.status == status]

        return sorted(proposals, key=lambda p: p.created_at, reverse=True)

    def update_status(
        self,
        proposal_id: str,
        status: ProposalStatus,
        notes: str | None = None,
    ) -> RefactoringProposal | None:
        """Update the status of a proposal."""
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            return None

        proposal.status = status
        proposal.updated_at = datetime.now()
        if notes:
            proposal.review_notes = notes

        self._save_proposals()
        return proposal

    async def _generate_llm_fix(
        self,
        debt: DebtItem,
        context: str,
    ) -> str:
        """Use LLM to generate a fix suggestion."""
        prompt = self._build_fix_prompt(debt, context)

        llm = self._llm
        if llm is not None and hasattr(llm, "generate"):
            response = await llm.generate(
                prompt,
                max_tokens=self._config.llm_max_proposal_tokens,
            )
        else:
            response = ""

        return self._extract_fix(response)

    def _build_fix_prompt(self, debt: DebtItem, context: str) -> str:
        """Build a prompt for LLM fix generation."""
        return f"""You are a code refactoring assistant. Analyze the following technical debt and propose a fix.

## Debt Information
- Type: {debt.type.name}
- File: {debt.file_path}
- Line: {debt.line_number}
- Message: {debt.message}
- Complexity: {debt.complexity or "N/A"}

## Code Context
```
{context}
```

## Task
Propose a refactoring fix for this debt. Provide:
1. A brief explanation of the issue
2. The proposed solution
3. Any potential risks or considerations

Format your response clearly with sections.
"""

    def _extract_fix(self, response: str) -> str:
        """Extract the fix from LLM response."""
        if "Proposed Solution:" in response:
            parts = response.split("Proposed Solution:")
            if len(parts) > 1:
                return parts[1].split("\n\n")[0].strip()
        return response[:500].strip()

    def _generate_template_fix(self, debt: DebtItem) -> str:
        """Generate a template fix based on debt type."""
        templates = {
            "TODO": f"# TODO addressed: {debt.message}",
            "FIXME": f"# FIXME: This needs attention - {debt.message}",
            "XXX": f"# XXX: Review this code - {debt.message}",
            "HACK": f"# HACK: Consider refactoring - {debt.message}",
            "COMPLEXITY": f"# Consider simplifying this function (complexity: {debt.complexity})",
            "LONG_FUNCTION": f"# Consider splitting this function into smaller pieces",
            "DUPLICATE": f"# Consider extracting this into a reusable function",
            "DEAD_CODE": f"# Consider removing this unused code",
        }
        return templates.get(debt.type.name, f"# Address debt: {debt.message}")

    def _get_code_context(self, debt: DebtItem) -> str:
        """Get surrounding code context for a debt item."""
        file_path = self._project_root / debt.file_path

        if not file_path.exists():
            return ""

        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
            start = max(0, debt.line_number - 5)
            end = min(len(lines), debt.end_line + 5)
            return "\n".join(f"{i + 1}: {lines[i]}" for i in range(start, end))
        except Exception:
            return ""

    def _generate_proposal_id(self, debt: DebtItem) -> str:
        """Generate a unique ID for a proposal."""
        import hashlib

        content = f"{debt.id}:{datetime.now().isoformat()}"
        return f"prop-{hashlib.md5(content.encode()).hexdigest()[:8]}"

    def _generate_branch_name(self, debt: DebtItem) -> str:
        """Generate a branch name for a proposal."""
        safe_name = debt.id.replace("/", "-").replace(" ", "-")[:30]
        return f"{self._config.branch_prefix}{safe_name}"

    def _get_current_branch(self) -> str:
        """Get the current git branch name."""
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self._project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            return ""

    def _check_git(self) -> bool:
        """Check if git is available."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self._project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _save_proposals(self) -> None:
        """Save proposals to disk."""
        storage_path = self._project_root / self._config.storage_path
        storage_path.mkdir(parents=True, exist_ok=True)

        filepath = storage_path / "proposals.json"

        try:
            data = {pid: p.to_dict() for pid, p in self._proposals.items()}
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self._logger.error(f"Failed to save proposals: {e}")

    def _load_proposals(self) -> None:
        """Load proposals from disk."""
        storage_path = self._project_root / self._config.storage_path
        filepath = storage_path / "proposals.json"

        if not filepath.exists():
            return

        try:
            with open(filepath) as f:
                data = json.load(f)

            self._proposals = {pid: RefactoringProposal.from_dict(p) for pid, p in data.items()}
            self._logger.info(f"Loaded {len(self._proposals)} proposals")
        except Exception as e:
            self._logger.error(f"Failed to load proposals: {e}")


def create_refinancing_engine(
    config: DebtConfig | None = None,
    llm_provider: LLMProvider | None = None,
    project_root: Path | str | None = None,
) -> RefinancingEngine:
    """Create a RefinancingEngine instance."""
    return RefinancingEngine(
        config=config,
        llm_provider=llm_provider,
        project_root=project_root,
    )
