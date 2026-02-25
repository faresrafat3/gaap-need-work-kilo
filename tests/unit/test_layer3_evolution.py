"""
Tests for Layer 3 Evolution - Zero-Trust Execution
===================================================

Tests:
- Layer3Config and presets
- StructuredToolCall and schemas
- NativeFunctionCaller
- ActiveLessonInjector
- CodeAuditor
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from gaap.layers.layer3_config import (
    Layer3Config,
    AuditConfig,
    ResourceLimits,
    LessonInjectionConfig,
    create_layer3_config,
)
from gaap.layers.execution_schema import (
    StructuredToolCall,
    StructuredOutput,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolCallStatus,
    FinishReason,
    ExecutionPlan,
    ExecutionStep,
    StructuredToolRegistry,
)
from gaap.layers.native_function_caller import (
    NativeFunctionCaller,
    create_native_caller,
)
from gaap.layers.active_lesson_injector import (
    ActiveLessonInjector,
    Lesson,
    InjectionResult,
    create_lesson_injector,
)
from gaap.layers.code_auditor import (
    CodeAuditor,
    AuditIssue,
    AuditResult,
    IssueSeverity,
    IssueType,
    create_code_auditor,
)


class TestLayer3Config:
    """Tests for Layer3Config."""

    def test_default_config(self):
        config = Layer3Config()

        assert config.execution_mode == "auto"
        assert config.sandbox_mode == "auto"
        assert config.network_enabled is False
        assert config.audit.enabled is True
        assert config.lesson_injection.enabled is True
        assert config.quality_threshold == 70.0

    def test_secure_preset(self):
        config = Layer3Config.secure()

        assert config.execution_mode == "native"
        assert config.sandbox_mode == "docker"
        assert config.network_enabled is False
        assert config.audit.fail_on_warnings is True
        assert "socket" in config.audit.banned_imports
        assert "subprocess" in config.audit.banned_imports

    def test_fast_preset(self):
        config = Layer3Config.fast()

        assert config.audit.enabled is False
        assert config.lesson_injection.enabled is False
        assert config.enable_twin is False

    def test_balanced_preset(self):
        config = Layer3Config.balanced()

        assert config.execution_mode == "auto"
        assert config.audit.enabled is True
        assert config.lesson_injection.enabled is True
        assert config.enable_twin is True

    def test_development_preset(self):
        config = Layer3Config.development()

        assert config.sandbox_mode == "disabled"
        assert config.network_enabled is True
        assert config.enable_sop is False

    def test_config_validation(self):
        with pytest.raises(ValueError):
            Layer3Config(quality_threshold=150.0)

        with pytest.raises(ValueError):
            Layer3Config(max_parallel=0)

    def test_to_dict_and_from_dict(self):
        config = Layer3Config(
            execution_mode="native",
            quality_threshold=80.0,
        )

        d = config.to_dict()
        restored = Layer3Config.from_dict(d)

        assert restored.execution_mode == "native"
        assert restored.quality_threshold == 80.0

    def test_factory_function(self):
        config = create_layer3_config("secure", max_parallel=5)

        assert config.execution_mode == "native"
        assert config.max_parallel == 5

    def test_resource_limits(self):
        limits = ResourceLimits(
            max_cpu_percent=30.0,
            max_memory_mb=256,
        )

        assert limits.max_cpu_percent == 30.0
        assert limits.to_dict()["max_cpu_percent"] == 30.0

    def test_audit_config(self):
        audit = AuditConfig(
            enabled=True,
            fail_on_errors=True,
            banned_imports=["os", "sys"],
        )

        assert len(audit.banned_imports) == 2
        assert "os" in audit.banned_imports

    def test_lesson_injection_config(self):
        lesson_config = LessonInjectionConfig(
            enabled=True,
            max_lessons=10,
            relevance_threshold=0.5,
        )

        assert lesson_config.max_lessons == 10
        assert lesson_config.relevance_threshold == 0.5


class TestStructuredToolCall:
    """Tests for StructuredToolCall."""

    def test_tool_call_creation(self):
        call = StructuredToolCall(
            call_id="call_123",
            tool_name="read_file",
            arguments={"path": "/test.txt"},
        )

        assert call.call_id == "call_123"
        assert call.tool_name == "read_file"
        assert call.status == ToolCallStatus.PENDING

    def test_tool_call_openai_format(self):
        call = StructuredToolCall(
            call_id="call_123",
            tool_name="test_tool",
            arguments={"arg1": "value1"},
        )

        formatted = call.to_openai_format()

        assert formatted["id"] == "call_123"
        assert formatted["type"] == "function"
        assert formatted["function"]["name"] == "test_tool"

    def test_tool_call_from_openai(self):
        data = {
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "write_file",
                "arguments": '{"path": "/test.txt", "content": "hello"}',
            },
        }

        call = StructuredToolCall.from_openai(data)

        assert call.call_id == "call_abc"
        assert call.tool_name == "write_file"
        assert call.arguments["path"] == "/test.txt"


class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_tool_definition_creation(self):
        tool = ToolDefinition(
            name="read_file",
            description="Read a file from disk",
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="File path to read",
                    required=True,
                ),
            ],
        )

        assert tool.name == "read_file"
        assert len(tool.parameters) == 1

    def test_tool_openai_schema(self):
        tool = ToolDefinition(
            name="test",
            description="Test tool",
            parameters=[
                ToolParameter(name="arg1", type="string", description="Arg 1"),
            ],
        )

        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test"
        assert "arg1" in schema["function"]["parameters"]["properties"]


class TestStructuredOutput:
    """Tests for StructuredOutput."""

    def test_output_creation(self):
        output = StructuredOutput(
            content="Hello world",
            tool_calls=[],
            finish_reason=FinishReason.STOP,
        )

        assert output.content == "Hello world"
        assert output.has_tool_calls() is False

    def test_output_with_tool_calls(self):
        call = StructuredToolCall(
            call_id="call_1",
            tool_name="test",
            arguments={},
        )

        output = StructuredOutput(
            content="",
            tool_calls=[call],
            finish_reason=FinishReason.TOOL_CALL,
        )

        assert output.has_tool_calls() is True
        assert len(output.get_tool_call_ids()) == 1


class TestToolResult:
    """Tests for ToolResult."""

    def test_successful_result(self):
        result = ToolResult(
            call_id="call_1",
            tool_name="read_file",
            output="File contents here",
            success=True,
        )

        assert result.success is True
        assert result.output == "File contents here"

    def test_failed_result(self):
        result = ToolResult(
            call_id="call_1",
            tool_name="read_file",
            output="",
            success=False,
            error="File not found",
        )

        assert result.success is False
        assert "Error" in result.to_openai_format()["content"]

    def test_anthropic_format(self):
        result = ToolResult(
            call_id="call_1",
            tool_name="test",
            output="success",
            success=True,
        )

        formatted = result.to_anthropic_format()

        assert formatted["type"] == "tool_result"
        assert formatted["tool_use_id"] == "call_1"


class TestExecutionPlan:
    """Tests for ExecutionPlan."""

    def test_plan_creation(self):
        plan = ExecutionPlan(
            plan_id="plan_1",
            task_id="task_1",
            steps=[
                ExecutionStep(
                    step_id="step_1",
                    description="First step",
                    action="llm_call",
                ),
                ExecutionStep(
                    step_id="step_2",
                    description="Second step",
                    action="tool_call",
                    dependencies=["step_1"],
                ),
            ],
        )

        assert len(plan.steps) == 2

    def test_get_ready_steps(self):
        plan = ExecutionPlan(
            plan_id="plan_1",
            task_id="task_1",
            steps=[
                ExecutionStep(
                    step_id="step_1",
                    description="First",
                    action="llm_call",
                ),
                ExecutionStep(
                    step_id="step_2",
                    description="Second",
                    action="tool_call",
                    dependencies=["step_1"],
                ),
            ],
        )

        ready = plan.get_ready_steps(set())
        assert len(ready) == 1
        assert ready[0].step_id == "step_1"

        ready = plan.get_ready_steps({"step_1"})
        assert len(ready) == 1
        assert ready[0].step_id == "step_2"


class TestNativeFunctionCaller:
    """Tests for NativeFunctionCaller."""

    def test_caller_creation(self):
        caller = NativeFunctionCaller()

        assert caller._config is not None
        assert caller._native_calls == 0

    def test_register_tool(self):
        caller = NativeFunctionCaller()

        tool = ToolDefinition(
            name="test_tool",
            description="Test",
        )

        caller.register_tool(tool)

        assert "test_tool" in caller._registry.tools

    def test_get_tools_schema(self):
        caller = NativeFunctionCaller()

        caller.register_tool(
            ToolDefinition(
                name="read_file",
                description="Read file",
            )
        )

        schemas = caller.get_tools_schema("openai")

        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "read_file"

    def test_parse_legacy_tool_calls(self):
        caller = NativeFunctionCaller()

        content = "Let me read the file. CALL: read_file(path='/test.txt')"
        calls = caller._parse_legacy_tool_calls(content)

        assert len(calls) == 1
        assert calls[0].tool_name == "read_file"
        assert calls[0].arguments.get("path") == "/test.txt"

    def test_caller_stats(self):
        caller = NativeFunctionCaller()

        stats = caller.get_stats()

        assert "total_calls" in stats
        assert "registered_tools" in stats


class TestActiveLessonInjector:
    """Tests for ActiveLessonInjector."""

    def test_injector_creation(self):
        injector = ActiveLessonInjector()

        assert injector._config is not None
        assert injector._injections == 0

    def test_get_default_lessons(self):
        injector = ActiveLessonInjector()

        from gaap.layers.layer2_tactical import AtomicTask, TaskCategory

        task = AtomicTask(
            id="task_1",
            name="Test Task",
            description="Test",
            category=TaskCategory.API,
        )

        lessons = injector._get_default_lessons(task)

        assert len(lessons) > 0
        assert all(isinstance(l, Lesson) for l in lessons)

    def test_format_lessons(self):
        injector = ActiveLessonInjector()

        lessons = [
            Lesson(
                lesson_id="l1",
                content="Always validate input",
                category="security",
                context="security review",
            ),
            Lesson(
                lesson_id="l2",
                content="Handle exceptions properly",
                category="code",
                context="code review",
            ),
        ]

        formatted = injector._format_lessons(lessons)

        assert "Always validate input" in formatted
        assert "Lessons from Past Executions" in formatted

    def test_injector_stats(self):
        injector = ActiveLessonInjector()

        stats = injector.get_stats()

        assert "total_injections" in stats
        assert "lessons_retrieved" in stats


class TestLesson:
    """Tests for Lesson."""

    def test_lesson_creation(self):
        lesson = Lesson(
            lesson_id="lesson_1",
            content="Never use eval()",
            category="security",
            context="Code review",
        )

        assert lesson.lesson_id == "lesson_1"
        assert lesson.success is False

    def test_lesson_to_dict(self):
        lesson = Lesson(
            lesson_id="l1",
            content="Test lesson",
            category="code",
            context="test",
        )

        d = lesson.to_dict()

        assert d["lesson_id"] == "l1"
        assert d["content"] == "Test lesson"


class TestCodeAuditor:
    """Tests for CodeAuditor."""

    def test_auditor_creation(self):
        auditor = CodeAuditor()

        assert auditor._audit_config is not None
        assert auditor._audits_run == 0

    @pytest.mark.asyncio
    async def test_check_syntax(self):
        auditor = CodeAuditor()

        code = "def broken(\n"  # Syntax error
        issues = auditor._check_syntax(code)

        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.SYNTAX

    @pytest.mark.asyncio
    async def test_check_valid_syntax(self):
        auditor = CodeAuditor()

        code = "def hello():\n    print('hello')\n"
        issues = auditor._check_syntax(code)

        assert len(issues) == 0

    @pytest.mark.asyncio
    async def test_check_banned_imports(self):
        auditor = CodeAuditor()

        code = "import socket\nimport subprocess\n"
        issues = auditor._check_banned_imports(code)

        assert len(issues) == 2

    @pytest.mark.asyncio
    async def test_check_banned_functions(self):
        auditor = CodeAuditor()

        code = "result = eval('1+1')\nexec('print(1)')\n"
        issues = auditor._check_banned_functions(code)

        assert len(issues) >= 2

    @pytest.mark.asyncio
    async def test_audit_passes(self):
        auditor = CodeAuditor()

        code = """
def hello(name: str) -> str:
    return f"Hello, {name}!"
"""
        result = await auditor.audit(code)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_audit_fails_banned_import(self):
        auditor = CodeAuditor()

        code = """
import socket

def connect():
    s = socket.socket()
    return s
"""
        result = await auditor.audit(code)

        assert result.passed is False
        assert any(i.issue_type == IssueType.IMPORT for i in result.issues)

    def test_quick_audit(self):
        auditor = CodeAuditor()

        good_code = "x = 1 + 1\nprint(x)"
        bad_code = "import subprocess\nsubprocess.run(['ls'])"

        assert auditor.quick_audit(good_code) is True
        assert auditor.quick_audit(bad_code) is False

    def test_auditor_stats(self):
        auditor = CodeAuditor()

        stats = auditor.get_stats()

        assert "audits_run" in stats
        assert "issues_found" in stats


class TestAuditIssue:
    """Tests for AuditIssue."""

    def test_issue_creation(self):
        issue = AuditIssue(
            issue_type=IssueType.SECURITY,
            severity=IssueSeverity.ERROR,
            message="Banned import: socket",
            line=1,
            code="IMPORT001",
        )

        assert issue.issue_type == IssueType.SECURITY
        assert issue.severity == IssueSeverity.ERROR

    def test_issue_to_dict(self):
        issue = AuditIssue(
            issue_type=IssueType.LINT,
            severity=IssueSeverity.WARNING,
            message="Unused variable",
            line=10,
        )

        d = issue.to_dict()

        assert d["issue_type"] == "LINT"
        assert d["severity"] == "WARNING"


class TestAuditResult:
    """Tests for AuditResult."""

    def test_result_passed(self):
        result = AuditResult(
            passed=True,
            issues=[],
        )

        assert result.passed is True
        assert result.error_count == 0

    def test_result_with_issues(self):
        result = AuditResult(
            passed=False,
            issues=[
                AuditIssue(
                    issue_type=IssueType.SECURITY,
                    severity=IssueSeverity.ERROR,
                    message="Error 1",
                ),
                AuditIssue(
                    issue_type=IssueType.LINT,
                    severity=IssueSeverity.WARNING,
                    message="Warning 1",
                ),
            ],
            error_count=1,
            warning_count=1,
        )

        assert result.passed is False
        assert len(result.get_errors()) == 1
        assert len(result.get_warnings()) == 1


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_native_caller(self):
        caller = create_native_caller()

        assert isinstance(caller, NativeFunctionCaller)

    def test_create_lesson_injector(self):
        injector = create_lesson_injector()

        assert isinstance(injector, ActiveLessonInjector)

    def test_create_code_auditor(self):
        auditor = create_code_auditor()

        assert isinstance(auditor, CodeAuditor)

    def test_create_layer3_config(self):
        config = create_layer3_config("secure")

        assert config.execution_mode == "native"
