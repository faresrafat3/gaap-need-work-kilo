"""
Tool-Interactive CRITIC Implementation
======================================

Critic with tool access for verification-based evaluation.
Uses tools to validate claims and gather evidence.

Key Components:
    - ToolInteractiveCritic: Critic with tool access
    - VerificationPlan: Steps to verify
    - VerificationResult: Result of verification

Usage:
    from gaap.layers.tool_critic import ToolInteractiveCritic

    critic = ToolInteractiveCritic(tools=[interpreter, api_search])
    result = await critic.evaluate_with_tools(subject)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol

from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import Message, MessageRole

if TYPE_CHECKING:
    from gaap.layers.evidence_critic import EvidenceBasedEvaluation

logger = get_logger("gaap.layers.tool_critic")


class VerificationStepType(Enum):
    """Type of verification step"""

    CODE_EXECUTION = auto()
    API_SEARCH = auto()
    ENDPOINT_CHECK = auto()
    DEPRECATION_CHECK = auto()
    SYNTAX_VALIDATION = auto()
    FUNCTION_TEST = auto()
    OUTPUT_COMPARISON = auto()
    CUSTOM = auto()


class VerificationStatus(Enum):
    """Status of verification"""

    PENDING = auto()
    RUNNING = auto()
    PASSED = auto()
    FAILED = auto()
    ERROR = auto()
    SKIPPED = auto()


@dataclass
class VerificationStep:
    """
    A single verification step.

    Attributes:
        id: Unique step identifier
        step_type: Type of verification
        description: Human-readable description
        tool_name: Name of tool to use
        tool_params: Parameters for the tool
        expected_result: Expected result for pass/fail
        actual_result: Actual result after execution
        status: Execution status
        error_message: Error if failed
        runtime_ms: Execution time
    """

    id: str
    step_type: VerificationStepType
    description: str
    tool_name: str = ""
    tool_params: dict[str, Any] = field(default_factory=dict)
    expected_result: Any = None
    actual_result: Any = None
    status: VerificationStatus = VerificationStatus.PENDING
    error_message: str = ""
    runtime_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "step_type": self.step_type.name,
            "description": self.description,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "expected_result": repr(self.expected_result),
            "actual_result": repr(self.actual_result),
            "status": self.status.name,
            "error_message": self.error_message,
            "runtime_ms": self.runtime_ms,
        }


@dataclass
class VerificationPlan:
    """
    Plan for verification with multiple steps.

    Attributes:
        plan_id: Unique plan identifier
        subject: What is being verified
        steps: List of verification steps
        created_at: Creation timestamp
        metadata: Additional metadata
    """

    plan_id: str
    subject: str
    steps: list[VerificationStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        step_type: VerificationStepType,
        description: str,
        tool_name: str = "",
        tool_params: dict[str, Any] | None = None,
        expected_result: Any = None,
    ) -> VerificationStep:
        """Add a verification step"""
        step = VerificationStep(
            id=f"{self.plan_id}_step_{len(self.steps)}",
            step_type=step_type,
            description=description,
            tool_name=tool_name,
            tool_params=tool_params or {},
            expected_result=expected_result,
        )
        self.steps.append(step)
        return step

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "subject": self.subject,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class VerificationResult:
    """
    Result of executing a verification plan.

    Attributes:
        plan_id: ID of the verification plan
        passed: Whether all verifications passed
        total_steps: Total number of steps
        passed_steps: Number of passed steps
        failed_steps: Number of failed steps
        error_steps: Number of error steps
        skipped_steps: Number of skipped steps
        total_runtime_ms: Total execution time
        evidence: Evidence collected during verification
        step_results: Detailed results for each step
        metadata: Additional metadata
    """

    plan_id: str
    passed: bool = False
    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0
    error_steps: int = 0
    skipped_steps: int = 0
    total_runtime_ms: float = 0.0
    evidence: list[str] = field(default_factory=list)
    step_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.passed_steps / self.total_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "passed": self.passed,
            "total_steps": self.total_steps,
            "passed_steps": self.passed_steps,
            "failed_steps": self.failed_steps,
            "error_steps": self.error_steps,
            "skipped_steps": self.skipped_steps,
            "success_rate": self.success_rate,
            "total_runtime_ms": self.total_runtime_ms,
            "evidence": self.evidence,
            "step_results": self.step_results,
            "metadata": self.metadata,
        }


class ToolProtocol(Protocol):
    """Protocol for tools used by the critic"""

    @property
    def name(self) -> str: ...

    async def execute(self, *args: Any, **kwargs: Any) -> Any: ...


class ToolInteractiveCritic:
    """
    Critic with tool access for verification-based evaluation.

    Uses tools to:
    - Execute code and verify behavior
    - Search API documentation
    - Check deprecation status
    - Compare outputs

    Attributes:
        tools: Dictionary of available tools
        provider: LLM provider for evaluation
        model: Model to use
        max_verification_steps: Maximum steps per verification
    """

    VERIFICATION_PROMPT = """You are a Tool-Interactive Critic. Your task is to evaluate a subject and create a verification plan.

## Subject to Evaluate
{subject}

## Available Tools
{tools_description}

## Instructions
1. Analyze the subject and identify claims that need verification
2. Create a verification plan using available tools
3. Each step should have clear expected outcomes
4. Output ONLY valid JSON

## Output Format
```json
{{
  "evaluation": {{
    "initial_score": <number 0-100>,
    "claims": ["claim 1", "claim 2"],
    "reasoning": "Initial reasoning"
  }},
  "verification_plan": {{
    "steps": [
      {{
        "type": "code_execution|api_search|endpoint_check|deprecation_check|syntax_validation|function_test",
        "description": "What to verify",
        "tool_name": "interpreter|api_search",
        "tool_params": {{}},
        "expected_result": "expected outcome"
      }}
    ]
  }}
}}
```

Remember: Be specific about what needs verification and how tools should be used."""

    def __init__(
        self,
        tools: list[Any] | None = None,
        provider: Any = None,
        model: str = "llama-3.3-70b-versatile",
        max_verification_steps: int = 10,
    ):
        self._tools: dict[str, Any] = {}
        self.provider = provider
        self.model = model
        self.max_verification_steps = max_verification_steps
        self._logger = logger

        if tools:
            for tool in tools:
                tool_name = getattr(tool, "name", tool.__class__.__name__.lower())
                self._tools[tool_name] = tool

    def register_tool(self, name: str, tool: Any) -> None:
        """Register a tool for verification"""
        self._tools[name] = tool

    def get_tool(self, name: str) -> Any | None:
        """Get a registered tool"""
        return self._tools.get(name)

    @property
    def available_tools(self) -> list[str]:
        """List of available tool names"""
        return list(self._tools.keys())

    async def generate_verification_plan(
        self,
        subject: str,
        context: dict[str, Any] | None = None,
    ) -> VerificationPlan:
        """
        Generate a verification plan for a subject.

        Args:
            subject: The subject to verify
            context: Additional context

        Returns:
            VerificationPlan with steps
        """
        plan_id = f"plan_{int(time.time() * 1000)}"
        plan = VerificationPlan(plan_id=plan_id, subject=subject)

        if not self.provider:
            return self._generate_fallback_plan(plan, subject)

        tools_desc = self._get_tools_description()

        prompt = self.VERIFICATION_PROMPT.format(
            subject=subject,
            tools_description=tools_desc,
        )

        try:
            messages = [
                Message(role=MessageRole.SYSTEM, content=prompt),
                Message(role=MessageRole.USER, content="Generate verification plan."),
            ]

            response = await self.provider.chat_completion(
                messages=messages,
                model=self.model,
                temperature=0.3,
                max_tokens=2048,
            )

            if not response.choices:
                return self._generate_fallback_plan(plan, subject)

            content = response.choices[0].message.content
            return self._parse_verification_plan(content, plan)

        except Exception as e:
            self._logger.warning(f"LLM verification plan failed: {e}")
            return self._generate_fallback_plan(plan, subject)

    def _get_tools_description(self) -> str:
        """Get description of available tools"""
        descriptions = []
        for name, tool in self._tools.items():
            desc = getattr(tool, "__doc__", f"Tool: {name}")
            descriptions.append(f"- {name}: {desc[:100]}...")
        return "\n".join(descriptions) if descriptions else "No tools available"

    def _parse_verification_plan(
        self,
        content: str,
        plan: VerificationPlan,
    ) -> VerificationPlan:
        """Parse LLM response into verification plan"""
        import re

        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return plan
            else:
                return plan

        plan.metadata["initial_score"] = data.get("evaluation", {}).get("initial_score", 50)
        plan.metadata["claims"] = data.get("evaluation", {}).get("claims", [])
        plan.metadata["reasoning"] = data.get("evaluation", {}).get("reasoning", "")

        steps_data = data.get("verification_plan", {}).get("steps", [])
        for i, step_data in enumerate(steps_data[: self.max_verification_steps]):
            step_type_str = step_data.get("type", "custom")
            try:
                step_type = VerificationStepType[step_type_str.upper()]
            except KeyError:
                step_type = VerificationStepType.CUSTOM

            plan.add_step(
                step_type=step_type,
                description=step_data.get("description", ""),
                tool_name=step_data.get("tool_name", ""),
                tool_params=step_data.get("tool_params", {}),
                expected_result=step_data.get("expected_result"),
            )

        return plan

    def _generate_fallback_plan(
        self,
        plan: VerificationPlan,
        subject: str,
    ) -> VerificationPlan:
        """Generate a fallback verification plan"""
        if "def " in subject or "async def " in subject:
            plan.add_step(
                step_type=VerificationStepType.SYNTAX_VALIDATION,
                description="Validate code syntax",
                tool_name="interpreter",
                tool_params={"code": subject},
            )
            plan.add_step(
                step_type=VerificationStepType.CODE_EXECUTION,
                description="Execute code safely",
                tool_name="interpreter",
                tool_params={"code": subject},
            )

        if "." in subject and "(" in subject:
            api_name = subject.split("(")[0].strip()
            plan.add_step(
                step_type=VerificationStepType.API_SEARCH,
                description=f"Search for API: {api_name}",
                tool_name="api_search",
                tool_params={"api_name": api_name},
            )

        return plan

    async def execute_verification(
        self,
        plan: VerificationPlan,
    ) -> VerificationResult:
        """
        Execute a verification plan.

        Args:
            plan: The verification plan to execute

        Returns:
            VerificationResult with outcomes
        """
        result = VerificationResult(plan_id=plan.plan_id)
        result.total_steps = len(plan.steps)

        for step in plan.steps:
            step.status = VerificationStatus.RUNNING
            start_time = time.time()

            try:
                tool = self.get_tool(step.tool_name)
                if not tool:
                    step.status = VerificationStatus.SKIPPED
                    step.error_message = f"Tool '{step.tool_name}' not available"
                    result.skipped_steps += 1
                    continue

                if step.step_type == VerificationStepType.CODE_EXECUTION:
                    step_result = await self._execute_code(step, tool)
                elif step.step_type == VerificationStepType.SYNTAX_VALIDATION:
                    step_result = await self._validate_syntax(step, tool)
                elif step.step_type == VerificationStepType.API_SEARCH:
                    step_result = await self._search_api(step, tool)
                elif step.step_type == VerificationStepType.ENDPOINT_CHECK:
                    step_result = await self._check_endpoint(step, tool)
                elif step.step_type == VerificationStepType.DEPRECATION_CHECK:
                    step_result = await self._check_deprecation(step, tool)
                elif step.step_type == VerificationStepType.FUNCTION_TEST:
                    step_result = await self._test_function(step, tool)
                elif step.step_type == VerificationStepType.OUTPUT_COMPARISON:
                    step_result = await self._compare_outputs(step, tool)
                else:
                    step_result = await self._execute_custom(step, tool)

                step.actual_result = step_result.get("result")
                step.status = (
                    VerificationStatus.PASSED
                    if step_result.get("success")
                    else VerificationStatus.FAILED
                )
                if step_result.get("error"):
                    step.error_message = step_result["error"]

                if step.status == VerificationStatus.PASSED:
                    result.passed_steps += 1
                    result.evidence.append(f"✓ {step.description}: {step.actual_result}")
                else:
                    result.failed_steps += 1
                    result.evidence.append(f"✗ {step.description}: {step.error_message}")

            except Exception as e:
                step.status = VerificationStatus.ERROR
                step.error_message = str(e)
                result.error_steps += 1
                result.evidence.append(f"⚠ {step.description}: Error - {e}")

            step.runtime_ms = (time.time() - start_time) * 1000
            result.step_results.append(step.to_dict())
            result.total_runtime_ms += step.runtime_ms

        result.passed = (
            result.passed_steps == result.total_steps
            and result.failed_steps == 0
            and result.error_steps == 0
        )

        return result

    async def _execute_code(self, step: VerificationStep, tool: Any) -> dict[str, Any]:
        """Execute code verification"""
        code = step.tool_params.get("code", "")
        timeout = step.tool_params.get("timeout", 5.0)

        result = await tool.execute(code, timeout=timeout)
        return {
            "success": result.success,
            "result": result.return_value,
            "error": result.error,
        }

    async def _validate_syntax(self, step: VerificationStep, tool: Any) -> dict[str, Any]:
        """Validate syntax verification"""
        code = step.tool_params.get("code", "")
        is_valid, error = tool.validate_syntax(code)
        return {
            "success": is_valid,
            "result": "valid" if is_valid else "invalid",
            "error": error,
        }

    async def _search_api(self, step: VerificationStep, tool: Any) -> dict[str, Any]:
        """API search verification"""
        api_name = step.tool_params.get("api_name", "")
        info = await tool.search_documentation(api_name)
        return {
            "success": info.exists,
            "result": info.to_dict(),
            "error": "" if info.exists else f"API '{api_name}' not found",
        }

    async def _check_endpoint(self, step: VerificationStep, tool: Any) -> dict[str, Any]:
        """Endpoint check verification"""
        url = step.tool_params.get("url", "")
        method = step.tool_params.get("method", "GET")
        info = await tool.verify_endpoint(url, method)
        return {
            "success": info.exists,
            "result": info.__dict__,
            "error": info.description,
        }

    async def _check_deprecation(self, step: VerificationStep, tool: Any) -> dict[str, Any]:
        """Deprecation check verification"""
        api_name = step.tool_params.get("api_name", "")
        is_deprecated, message = await tool.check_deprecation(api_name)
        return {
            "success": not is_deprecated,
            "result": {"deprecated": is_deprecated, "message": message},
            "error": message if is_deprecated else "",
        }

    async def _test_function(self, step: VerificationStep, tool: Any) -> dict[str, Any]:
        """Function test verification"""
        code = step.tool_params.get("code", "")
        function_name = step.tool_params.get("function_name", "")
        test_cases = step.tool_params.get("test_cases", [])

        results = await tool.test_function(code, function_name, test_cases)
        all_passed = all(r.passed for r in results)
        return {
            "success": all_passed,
            "result": [r.__dict__ for r in results],
            "error": "" if all_passed else "Some tests failed",
        }

    async def _compare_outputs(self, step: VerificationStep, tool: Any) -> dict[str, Any]:
        """Output comparison verification"""
        code1 = step.tool_params.get("code1", "")
        code2 = step.tool_params.get("code2", "")
        test_inputs = step.tool_params.get("test_inputs", [])

        comparison = await tool.compare_outputs(code1, code2, test_inputs)
        return {
            "success": comparison.get("match_rate", 0) == 1.0,
            "result": comparison,
            "error": "",
        }

    async def _execute_custom(self, step: VerificationStep, tool: Any) -> dict[str, Any]:
        """Execute custom verification"""
        params = step.tool_params
        if hasattr(tool, step.tool_params.get("method", "execute")):
            method = getattr(tool, step.tool_params.get("method", "execute"))
            result = await method(**step.tool_params.get("kwargs", {}))
            return {
                "success": True,
                "result": result,
                "error": "",
            }
        return {
            "success": False,
            "result": None,
            "error": "Custom verification method not found",
        }

    def compare_results(
        self,
        expected: Any,
        actual: Any,
        tolerance: float = 0.0,
    ) -> tuple[bool, str]:
        """
        Compare expected and actual results.

        Args:
            expected: Expected value
            actual: Actual value
            tolerance: Tolerance for numeric comparisons

        Returns:
            Tuple of (match, description)
        """
        if expected is None:
            return True, "No expected value to compare"

        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            if tolerance > 0:
                match = abs(expected - actual) <= tolerance
            else:
                match = expected == actual
            return match, f"Expected {expected}, got {actual}"

        if isinstance(expected, str) and isinstance(actual, str):
            match = expected.lower() == actual.lower()
            return match, f"Expected '{expected}', got '{actual}'"

        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            match = len(expected) == len(actual)
            if not match:
                return False, f"Length mismatch: expected {len(expected)}, got {len(actual)}"
            match = all(e == a for e, a in zip(expected, actual))
            return match, f"List comparison: {match}"

        if isinstance(expected, dict) and isinstance(actual, dict):
            match = expected == actual
            return match, f"Dict comparison: {match}"

        match = expected == actual
        return match, f"Comparison: {match}"

    async def evaluate_with_tools(
        self,
        subject: str,
        context: dict[str, Any] | None = None,
    ) -> "EvidenceBasedEvaluation":
        """
        Main evaluation method with tool verification.

        Args:
            subject: The subject to evaluate
            context: Additional context

        Returns:
            EvidenceBasedEvaluation with tool-verified evidence
        """
        from gaap.layers.evidence_critic import EvidenceBasedEvaluation, EvidenceStrength

        start_time = time.time()

        plan = await self.generate_verification_plan(subject, context)

        result = await self.execute_verification(plan)

        evidence = result.evidence.copy()
        reasoning = f"Verification plan executed with {result.passed_steps}/{result.total_steps} steps passed."

        if result.passed:
            score = 0.9
            evidence_strength = EvidenceStrength.STRONG
        elif result.success_rate >= 0.7:
            score = 0.7
            evidence_strength = EvidenceStrength.MODERATE
        elif result.success_rate >= 0.5:
            score = 0.5
            evidence_strength = EvidenceStrength.WEAK
        else:
            score = 0.3
            evidence_strength = EvidenceStrength.NONE

        evidence.append(f"Total verification time: {result.total_runtime_ms:.0f}ms")

        suggestions = []
        for step_result in result.step_results:
            if step_result.get("status") == "FAILED":
                suggestions.append(f"Fix: {step_result.get('description', 'Unknown step')}")
            elif step_result.get("status") == "ERROR":
                suggestions.append(
                    f"Review error: {step_result.get('error_message', 'Unknown error')}"
                )

        elapsed_ms = (time.time() - start_time) * 1000

        return EvidenceBasedEvaluation(
            critic="tool_interactive",
            score=score,
            evidence=evidence,
            reasoning=reasoning,
            confidence=result.success_rate,
            suggestions=suggestions,
            evidence_strength=evidence_strength,
            metadata={
                "plan_id": plan.plan_id,
                "total_steps": result.total_steps,
                "passed_steps": result.passed_steps,
                "runtime_ms": elapsed_ms,
            },
        )


def create_tool_interactive_critic(
    tools: list[Any] | None = None,
    provider: Any = None,
) -> ToolInteractiveCritic:
    """Create a tool-interactive critic with default tools"""
    critic = ToolInteractiveCritic(tools=tools, provider=provider)

    if tools is None:
        try:
            from gaap.tools.interpreter_tool import InterpreterTool

            interpreter = InterpreterTool()
            critic.register_tool("interpreter", interpreter)
        except ImportError:
            pass

        try:
            from gaap.tools.search_tool import APISearchTool

            api_search = APISearchTool()
            critic.register_tool("api_search", api_search)
        except ImportError:
            pass

    return critic
