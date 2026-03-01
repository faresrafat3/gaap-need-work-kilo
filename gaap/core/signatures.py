"""
Declarative Signatures - DSPy-Inspired Module System

Moves from static prompts to declarative modules with:
- Structured input/output schemas
- Automatic prompt optimization
- Few-shot example selection
- Type-safe execution

Inspired by DSPy: https://github.com/stanfordnlp/dspy
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger("gaap.core.signatures")


class FieldType(Enum):
    """Field types for signatures"""

    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    BOOLEAN = auto()
    LIST = auto()
    DICT = auto()
    OBJECT = auto()


@dataclass
class SignatureField:
    """
    A field in a signature (input or output).

    Attributes:
        name: Field name
        field_type: Type of the field
        description: Human-readable description
        required: Whether the field is required
        default: Default value if optional
        constraints: Validation constraints
        examples: Example values for few-shot prompting
    """

    name: str
    field_type: FieldType = FieldType.STRING
    description: str = ""
    required: bool = True
    default: Any = None
    constraints: dict[str, Any] = field(default_factory=dict)
    examples: list[Any] = field(default_factory=list)

    def validate(self, value: Any) -> bool:
        """Validate a value against this field"""
        if value is None:
            return not self.required

        type_check = {
            FieldType.STRING: lambda v: isinstance(v, str),
            FieldType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            FieldType.FLOAT: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            FieldType.BOOLEAN: lambda v: isinstance(v, bool),
            FieldType.LIST: lambda v: isinstance(v, list),
            FieldType.DICT: lambda v: isinstance(v, dict),
            FieldType.OBJECT: lambda v: True,
        }

        if not type_check.get(self.field_type, lambda v: False)(value):
            return False

        if self.constraints:
            if "min_length" in self.constraints and hasattr(value, "__len__"):
                if len(value) < self.constraints["min_length"]:
                    return False
            if "max_length" in self.constraints and hasattr(value, "__len__"):
                if len(value) > self.constraints["max_length"]:
                    return False
            if "min_value" in self.constraints and isinstance(value, (int, float)):
                if value < self.constraints["min_value"]:
                    return False
            if "max_value" in self.constraints and isinstance(value, (int, float)):
                if value > self.constraints["max_value"]:
                    return False
            if "pattern" in self.constraints and isinstance(value, str):
                import re

                if not re.match(self.constraints["pattern"], value):
                    return False

        return True

    def to_prompt_fragment(self) -> str:
        """Convert field to prompt fragment"""
        type_name = self.field_type.name.lower()
        req_marker = " (required)" if self.required else " (optional)"
        desc = f": {self.description}" if self.description else ""
        return f"{self.name} [{type_name}]{req_marker}{desc}"


@dataclass
class Signature:
    """
    Input/output schema for a declarative module.

    Inspired by DSPy's Signature concept - defines what a module
    expects as input and produces as output.

    Attributes:
        name: Signature name
        description: What this signature represents
        inputs: List of input fields
        outputs: List of output fields
        instructions: Additional instructions for execution
        metadata: Additional metadata
    """

    name: str
    description: str = ""
    inputs: list[SignatureField] = field(default_factory=list)
    outputs: list[SignatureField] = field(default_factory=list)
    instructions: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_input(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate input data against input fields.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        for field in self.inputs:
            value = data.get(field.name)
            if value is None and field.required:
                errors.append(f"Missing required field: {field.name}")
            elif value is not None and not field.validate(value):
                errors.append(f"Invalid value for field: {field.name}")

        return len(errors) == 0, errors

    def validate_output(self, data: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate output data against output fields.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        for field in self.outputs:
            value = data.get(field.name)
            if value is None and field.required:
                errors.append(f"Missing required output field: {field.name}")
            elif value is not None and not field.validate(value):
                errors.append(f"Invalid value for output field: {field.name}")

        return len(errors) == 0, errors

    def to_prompt(self) -> str:
        """
        Convert signature to a prompt template.

        Creates a structured prompt that describes the task,
        inputs expected, and outputs to produce.
        """
        lines = [f"Task: {self.name}"]

        if self.description:
            lines.append(f"\nDescription: {self.description}")

        if self.inputs:
            lines.append("\nInputs:")
            for inp in self.inputs:
                lines.append(f"  - {inp.to_prompt_fragment()}")

        if self.outputs:
            lines.append("\nOutputs:")
            for out in self.outputs:
                lines.append(f"  - {out.to_prompt_fragment()}")

        if self.instructions:
            lines.append(f"\nInstructions: {self.instructions}")

        return "\n".join(lines)

    def get_input_schema(self) -> dict[str, Any]:
        """Get JSON schema for inputs"""
        return {
            "type": "object",
            "properties": {f.name: self._field_to_schema(f) for f in self.inputs},
            "required": [f.name for f in self.inputs if f.required],
        }

    def get_output_schema(self) -> dict[str, Any]:
        """Get JSON schema for outputs"""
        return {
            "type": "object",
            "properties": {f.name: self._field_to_schema(f) for f in self.outputs},
            "required": [f.name for f in self.outputs if f.required],
        }

    def _field_to_schema(self, field: SignatureField) -> dict[str, Any]:
        """Convert field to JSON schema"""
        type_map = {
            FieldType.STRING: "string",
            FieldType.INTEGER: "integer",
            FieldType.FLOAT: "number",
            FieldType.BOOLEAN: "boolean",
            FieldType.LIST: "array",
            FieldType.DICT: "object",
            FieldType.OBJECT: "object",
        }

        schema: dict[str, Any] = {"type": type_map.get(field.field_type, "string")}

        if field.description:
            schema["description"] = field.description

        if field.default is not None:
            schema["default"] = field.default

        if field.constraints:
            if "min_length" in field.constraints:
                schema["minLength"] = field.constraints["min_length"]
            if "max_length" in field.constraints:
                schema["maxLength"] = field.constraints["max_length"]
            if "min_value" in field.constraints:
                schema["minimum"] = field.constraints["min_value"]
            if "max_value" in field.constraints:
                schema["maximum"] = field.constraints["max_value"]
            if "pattern" in field.constraints:
                schema["pattern"] = field.constraints["pattern"]

        return schema

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Signature:
        """Create signature from dictionary"""
        inputs = [
            SignatureField(
                name=f.get("name", ""),
                field_type=FieldType[f.get("type", "STRING").upper()],
                description=f.get("description", ""),
                required=f.get("required", True),
                default=f.get("default"),
                constraints=f.get("constraints", {}),
                examples=f.get("examples", []),
            )
            for f in data.get("inputs", [])
        ]

        outputs = [
            SignatureField(
                name=f.get("name", ""),
                field_type=FieldType[f.get("type", "STRING").upper()],
                description=f.get("description", ""),
                required=f.get("required", True),
                default=f.get("default"),
                constraints=f.get("constraints", {}),
                examples=f.get("examples", []),
            )
            for f in data.get("outputs", [])
        ]

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            inputs=inputs,
            outputs=outputs,
            instructions=data.get("instructions", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Example:
    """
    A labeled example for a signature.

    Contains input/output pairs that demonstrate
    correct behavior for a module.
    """

    inputs: dict[str, Any]
    outputs: dict[str, Any]
    quality_score: float = 1.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_few_shot(self) -> str:
        """Convert to few-shot prompt format"""
        lines = []

        lines.append("Input:")
        for key, value in self.inputs.items():
            lines.append(f"  {key}: {value}")

        lines.append("Output:")
        for key, value in self.outputs.items():
            lines.append(f"  {key}: {value}")

        return "\n".join(lines)


@dataclass
class OptimizationResult:
    """Result of prompt optimization"""

    optimized_prompt: str
    examples_used: list[Example]
    improvement_score: float
    iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Teleprompter:
    """
    Auto-optimizes prompts from episodic memory.

    Inspired by DSPy's teleprompter concept - automatically
    selects and optimizes prompts based on successful examples.

    Features:
    - Retrieves best examples from memory
    - Builds optimized prompts
    - Tracks optimization performance
    - Adapts to different signatures
    """

    def __init__(
        self,
        memory: Any = None,
        max_examples: int = 5,
        min_quality_score: float = 0.7,
    ) -> None:
        self._memory = memory
        self._max_examples = max_examples
        self._min_quality_score = min_quality_score
        self._example_cache: dict[str, list[Example]] = {}
        self._optimization_history: list[OptimizationResult] = []
        self._logger = logging.getLogger("gaap.core.signatures.teleprompter")

    def index_example(self, signature: Signature, example: Example) -> None:
        """Index an example for a signature"""
        sig_name = signature.name
        if sig_name not in self._example_cache:
            self._example_cache[sig_name] = []

        self._example_cache[sig_name].append(example)

        self._example_cache[sig_name].sort(key=lambda e: e.quality_score, reverse=True)

        if len(self._example_cache[sig_name]) > 100:
            self._example_cache[sig_name] = self._example_cache[sig_name][:100]

    def get_best_examples(
        self,
        signature: Signature,
        k: int = 3,
        filter_fn: Callable[[Example], bool] | None = None,
    ) -> list[Example]:
        """
        Get best examples for a signature.

        Args:
            signature: The signature to get examples for
            k: Number of examples to return
            filter_fn: Optional filter function

        Returns:
            List of best examples
        """
        examples = self._example_cache.get(signature.name, [])

        if filter_fn:
            examples = [e for e in examples if filter_fn(e)]

        high_quality = [e for e in examples if e.quality_score >= self._min_quality_score]

        return high_quality[:k]

    def optimize(
        self,
        signature: Signature,
        examples: list[Example] | None = None,
        context: str | None = None,
    ) -> OptimizationResult:
        """
        Optimize a prompt for a signature.

        Creates an optimized prompt that includes:
        - Clear task description
        - Input/output specifications
        - Best available examples
        - Contextual instructions

        Args:
            signature: The signature to optimize for
            examples: Optional examples to use (defaults to cached)
            context: Optional additional context

        Returns:
            OptimizationResult with optimized prompt
        """
        if examples is None:
            examples = self.get_best_examples(signature, self._max_examples)

        prompt_parts = [signature.to_prompt()]

        if context:
            prompt_parts.insert(0, f"Context: {context}")

        if examples:
            prompt_parts.append("\n\nExamples:")
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(example.to_few_shot())

        optimized_prompt = "\n".join(prompt_parts)

        improvement_score = 0.0
        if examples:
            avg_quality = sum(e.quality_score for e in examples) / len(examples)
            example_boost = min(len(examples) * 0.1, 0.5)
            improvement_score = avg_quality * 0.5 + example_boost

        result = OptimizationResult(
            optimized_prompt=optimized_prompt,
            examples_used=examples,
            improvement_score=improvement_score,
            iterations=1,
            metadata={
                "signature_name": signature.name,
                "num_examples": len(examples),
                "optimization_time": datetime.now().isoformat(),
            },
        )

        self._optimization_history.append(result)

        return result

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get statistics about optimizations"""
        if not self._optimization_history:
            return {
                "total_optimizations": 0,
                "avg_improvement": 0.0,
                "signatures_optimized": [],
            }

        return {
            "total_optimizations": len(self._optimization_history),
            "avg_improvement": sum(r.improvement_score for r in self._optimization_history)
            / len(self._optimization_history),
            "signatures_optimized": list(
                set(r.metadata.get("signature_name") for r in self._optimization_history)
            ),
            "total_examples_indexed": sum(
                len(examples) for examples in self._example_cache.values()
            ),
        }


@dataclass
class ExecutionTrace:
    """Trace of a module execution"""

    module_name: str
    signature_name: str
    inputs: dict[str, Any]
    outputs: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: str | None = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class Module:
    """
    Declarative task module.

    Inspired by DSPy's Module concept - a reusable,
    composable unit that:
    - Has a defined signature (inputs/outputs)
    - Can be executed with type checking
    - Integrates with teleprompter for optimization
    - Tracks execution traces

    Usage:
        # Define a signature
        sig = Signature(
            name="code_review",
            description="Review code for issues",
            inputs=[
                SignatureField(name="code", field_type=FieldType.STRING, description="Code to review"),
            ],
            outputs=[
                SignatureField(name="issues", field_type=FieldType.LIST, description="List of issues found"),
                SignatureField(name="score", field_type=FieldType.FLOAT, description="Quality score"),
            ],
        )

        # Create module
        module = Module(signature=sig, executor=my_llm_provider)

        # Execute
        result = await module.execute({"code": "def foo(): pass"})
    """

    def __init__(
        self,
        signature: Signature,
        executor: Callable | None = None,
        teleprompter: Teleprompter | None = None,
        name: str | None = None,
    ) -> None:
        self.signature = signature
        self._executor = executor
        self._teleprompter = teleprompter
        self.name = name or signature.name
        self._traces: list[ExecutionTrace] = []
        self._optimized_prompt: str | None = None
        self._logger = logging.getLogger(f"gaap.core.signatures.module.{self.name}")

    @property
    def prompt(self) -> str:
        """Get the current prompt (optimized if available)"""
        if self._optimized_prompt:
            return self._optimized_prompt
        return self.signature.to_prompt()

    def optimize_prompt(self, examples: list[Example] | None = None) -> str:
        """Optimize the prompt using teleprompter"""
        if self._teleprompter:
            result = self._teleprompter.optimize(self.signature, examples)
            self._optimized_prompt = result.optimized_prompt
            return self._optimized_prompt
        return self.signature.to_prompt()

    async def execute(
        self,
        inputs: dict[str, Any],
        validate: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute the module with given inputs.

        Args:
            inputs: Input data matching signature inputs
            validate: Whether to validate inputs/outputs
            **kwargs: Additional arguments for executor

        Returns:
            Output data matching signature outputs

        Raises:
            ValueError: If validation fails
            RuntimeError: If no executor is configured
        """
        start_time = datetime.now()
        trace = ExecutionTrace(
            module_name=self.name,
            signature_name=self.signature.name,
            inputs=inputs,
        )

        try:
            if validate:
                is_valid, errors = self.signature.validate_input(inputs)
                if not is_valid:
                    raise ValueError(f"Input validation failed: {errors}")

            if self._executor is None:
                raise RuntimeError("No executor configured for module")

            if callable(self._executor):
                if hasattr(self._executor, "__call__"):
                    import asyncio

                    if asyncio.iscoroutinefunction(self._executor):
                        outputs = await self._executor(inputs, prompt=self.prompt, **kwargs)
                    else:
                        outputs = self._executor(inputs, prompt=self.prompt, **kwargs)
                else:
                    outputs = self._executor(inputs, prompt=self.prompt, **kwargs)
            else:
                outputs = {}

            if validate:
                is_valid, errors = self.signature.validate_output(outputs)
                if not is_valid:
                    self._logger.warning(f"Output validation issues: {errors}")

            trace.outputs = outputs
            trace.success = True

            return outputs

        except Exception as e:
            trace.error = str(e)
            trace.success = False
            self._logger.error(f"Module execution failed: {e}")
            raise

        finally:
            trace.latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._traces.append(trace)

    def get_traces(self, limit: int = 100) -> list[ExecutionTrace]:
        """Get recent execution traces"""
        return self._traces[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get module statistics"""
        if not self._traces:
            return {
                "name": self.name,
                "signature": self.signature.name,
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
            }

        successful = sum(1 for t in self._traces if t.success)
        total = len(self._traces)
        avg_latency = sum(t.latency_ms for t in self._traces) / total

        return {
            "name": self.name,
            "signature": self.signature.name,
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "avg_latency_ms": avg_latency,
            "has_optimized_prompt": self._optimized_prompt is not None,
        }

    def __repr__(self) -> str:
        return f"Module(name={self.name}, signature={self.signature.name})"


class ModuleRegistry:
    """
    Registry for reusable modules.

    Allows registration and lookup of modules by name,
    enabling composition and reuse across the system.
    """

    _instance: ModuleRegistry | None = None

    def __init__(self) -> None:
        self._modules: dict[str, Module] = {}
        self._signatures: dict[str, Signature] = {}
        self._logger = logging.getLogger("gaap.core.signatures.registry")

    @classmethod
    def get_instance(cls) -> ModuleRegistry:
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_module(self, module: Module) -> None:
        """Register a module"""
        self._modules[module.name] = module
        self._signatures[module.signature.name] = module.signature
        self._logger.debug(f"Registered module: {module.name}")

    def register_signature(self, signature: Signature) -> None:
        """Register a signature without a module"""
        self._signatures[signature.name] = signature

    def get_module(self, name: str) -> Module | None:
        """Get a module by name"""
        return self._modules.get(name)

    def get_signature(self, name: str) -> Signature | None:
        """Get a signature by name"""
        return self._signatures.get(name)

    def list_modules(self) -> list[str]:
        """List all registered modules"""
        return list(self._modules.keys())

    def list_signatures(self) -> list[str]:
        """List all registered signatures"""
        return list(self._signatures.keys())

    def clear(self) -> None:
        """Clear all registrations"""
        self._modules.clear()
        self._signatures.clear()


def get_registry() -> ModuleRegistry:
    """Get the module registry singleton"""
    return ModuleRegistry.get_instance()
