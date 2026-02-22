import traceback
from datetime import datetime
from typing import Any

# =============================================================================
# Base Exception
# =============================================================================


class GAAPException(Exception):
    """
    الاستثناء الأساسي لجميع أخطاء GAAP

    يوفر:
    - رمز خطأ فريد
    - سياق مفصل
    - اقتراحات للإصلاح
    - تتبع كامل
    """

    error_code: str = "GAAP_000"
    error_category: str = "general"
    severity: str = "error"  # debug, info, warning, error, critical

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        suggestions: list[str] | None = None,
        cause: Exception | None = None,
        recoverable: bool = True,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.suggestions = suggestions or []
        self.cause = cause
        self.recoverable = recoverable
        self.context = context or {}
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc() if cause else None

    def to_dict(self) -> dict[str, Any]:
        """تحويل الاستثناء إلى قاموس للتسجيل والاستجابة"""
        return {
            "error_code": self.error_code,
            "error_category": self.error_category,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "suggestions": self.suggestions,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback,
        }

    def __str__(self) -> str:
        parts = [f"[{self.error_code}] {self.message}"]
        if self.details:
            parts.append(f"Details: {self.details}")
        if self.suggestions:
            parts.append(f"Suggestions: {', '.join(self.suggestions)}")
        return " | ".join(parts)


# =============================================================================
# Configuration Exceptions
# =============================================================================


class ConfigurationError(GAAPException):
    """خطأ في تكوين النظام"""

    error_code = "GAAP_CFG_001"
    error_category = "configuration"
    severity = "error"


class InvalidConfigValueError(ConfigurationError):
    """قيمة تكوين غير صالحة"""

    error_code = "GAAP_CFG_002"

    def __init__(self, key: str, value: Any, expected_type: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Invalid configuration value for '{key}'",
            details={"key": key, "value": str(value), "expected_type": expected_type},
            suggestions=[f"Provide a valid {expected_type} value for '{key}'"],
            **kwargs,
        )


class MissingConfigError(ConfigurationError):
    """تكوين مفقود"""

    error_code = "GAAP_CFG_003"

    def __init__(self, key: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Missing required configuration: '{key}'",
            details={"missing_key": key},
            suggestions=[f"Add '{key}' to your configuration file"],
            **kwargs,
        )


class ConfigLoadError(ConfigurationError):
    """فشل تحميل التكوين"""

    error_code = "GAAP_CFG_004"

    def __init__(self, config_path: str, reason: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Failed to load configuration from '{config_path}'",
            details={"path": config_path, "reason": reason},
            suggestions=[
                "Check if the file exists",
                "Verify the file format (YAML/JSON)",
                "Check file permissions",
            ],
            **kwargs,
        )


# =============================================================================
# Provider Exceptions
# =============================================================================


class ProviderError(GAAPException):
    """خطأ في المزود"""

    error_code = "GAAP_PRV_001"
    error_category = "provider"
    severity = "error"


class ProviderNotFoundError(ProviderError):
    """المزود غير موجود"""

    error_code = "GAAP_PRV_002"

    def __init__(self, provider_name: str, available_providers: list[str], **kwargs: Any) -> None:
        super().__init__(
            message=f"Provider '{provider_name}' not found",
            details={"requested": provider_name, "available": available_providers},
            suggestions=[f"Use one of: {', '.join(available_providers)}"],
            **kwargs,
        )


class ProviderNotAvailableError(ProviderError):
    """المزود غير متاح"""

    error_code = "GAAP_PRV_003"
    recoverable = True

    def __init__(self, provider_name: str, reason: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Provider '{provider_name}' is not available",
            details={"provider": provider_name, "reason": reason},
            suggestions=[
                "Try again later",
                "Check your internet connection",
                "Use a different provider",
            ],
            **kwargs,
        )


class ProviderRateLimitError(ProviderError):
    """تجاوز حد الطلبات"""

    error_code = "GAAP_PRV_004"
    recoverable = True

    def __init__(self, provider_name: str, retry_after: int | None = None, **kwargs: Any) -> None:
        details: dict[str, Any] = {"provider": provider_name}
        if retry_after:
            details["retry_after_seconds"] = retry_after

        suggestions = ["Wait before retrying", "Use a different provider"]
        if retry_after:
            suggestions.insert(0, f"Retry after {retry_after} seconds")

        super().__init__(
            message=f"Rate limit exceeded for provider '{provider_name}'",
            details=details,
            suggestions=suggestions,
            **kwargs,
        )


class ProviderAuthenticationError(ProviderError):
    """خطأ في المصادقة"""

    error_code = "GAAP_PRV_005"
    severity = "critical"

    def __init__(self, provider_name: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Authentication failed for provider '{provider_name}'",
            details={"provider": provider_name},
            suggestions=[
                "Check your API key",
                "Verify the key has the required permissions",
                "Check if the key has expired",
            ],
            recoverable=False,
            **kwargs,
        )


class ProviderTimeoutError(ProviderError):
    """انتهت مهلة المزود"""

    error_code = "GAAP_PRV_006"
    recoverable = True

    def __init__(self, provider_name: str, timeout_seconds: float, **kwargs: Any) -> None:
        super().__init__(
            message=f"Provider '{provider_name}' timed out after {timeout_seconds}s",
            details={"provider": provider_name, "timeout": timeout_seconds},
            suggestions=[
                "Increase the timeout value",
                "Use a faster model",
                "Reduce the complexity of your request",
            ],
            **kwargs,
        )


class ModelNotFoundError(ProviderError):
    """النموذج غير موجود"""

    error_code = "GAAP_PRV_007"

    def __init__(
        self, model_name: str, provider_name: str, available_models: list[str], **kwargs: Any
    ) -> None:
        super().__init__(
            message=f"Model '{model_name}' not found in provider '{provider_name}'",
            details={
                "model": model_name,
                "provider": provider_name,
                "available": available_models[:10],  # أول 10 فقط
            },
            suggestions=[f"Use one of: {', '.join(available_models[:5])}"],
            **kwargs,
        )


class ProviderResponseError(ProviderError):
    """خطأ في استجابة المزود"""

    error_code = "GAAP_PRV_008"

    def __init__(
        self, provider_name: str, status_code: int, response_body: str, **kwargs: Any
    ) -> None:
        super().__init__(
            message=f"Provider '{provider_name}' returned error {status_code}",
            details={
                "provider": provider_name,
                "status_code": status_code,
                "response": response_body[:500],  # أول 500 حرف
            },
            suggestions=[
                "Check the provider's status page",
                "Try with different parameters",
                "Contact support if the error persists",
            ],
            **kwargs,
        )


# =============================================================================
# Routing Exceptions
# =============================================================================


class RoutingError(GAAPException):
    """خطأ في التوجيه"""

    error_code = "GAAP_ROT_001"
    error_category = "routing"
    severity = "error"


class NoAvailableProviderError(RoutingError):
    """لا يوجد مزود متاح"""

    error_code = "GAAP_ROT_002"
    severity = "critical"

    def __init__(self, requirements: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(
            message="No available provider meets the requirements",
            details={"requirements": requirements},
            suggestions=[
                "Relax your requirements",
                "Add more providers to your configuration",
                "Check provider availability",
            ],
            recoverable=False,
            **kwargs,
        )


class BudgetExceededError(RoutingError):
    """تجاوز الميزانية"""

    error_code = "GAAP_ROT_003"
    severity = "critical"

    def __init__(self, budget: float, required: float, **kwargs: Any) -> None:
        super().__init__(
            message=f"Budget exceeded: ${required:.2f} required, ${budget:.2f} available",
            details={"budget": budget, "required": required},
            suggestions=["Increase your budget", "Use cheaper models", "Reduce task complexity"],
            recoverable=False,
            **kwargs,
        )


class RoutingConflictError(RoutingError):
    """تعارض في التوجيه"""

    error_code = "GAAP_ROT_004"

    def __init__(self, conflicts: list[str], **kwargs: Any) -> None:
        super().__init__(
            message="Routing conflicts detected",
            details={"conflicts": conflicts},
            suggestions=["Review and resolve conflicting requirements"],
            **kwargs,
        )


# =============================================================================
# Task Exceptions
# =============================================================================


class TaskError(GAAPException):
    """خطأ في المهمة"""

    error_code = "GAAP_TSK_001"
    error_category = "task"
    severity = "error"


class TaskValidationError(TaskError):
    """خطأ في التحقق من المهمة"""

    error_code = "GAAP_TSK_002"

    def __init__(self, task_id: str, validation_errors: list[str], **kwargs: Any) -> None:
        super().__init__(
            message=f"Task '{task_id}' validation failed",
            details={"task_id": task_id, "errors": validation_errors},
            suggestions=["Fix the validation errors and retry"],
            **kwargs,
        )


class TaskDependencyError(TaskError):
    """خطأ في تبعيات المهمة"""

    error_code = "GAAP_TSK_003"

    def __init__(self, task_id: str, missing_dependencies: list[str], **kwargs: Any) -> None:
        super().__init__(
            message=f"Task '{task_id}' has missing dependencies",
            details={"task_id": task_id, "missing": missing_dependencies},
            suggestions=[
                "Resolve the missing dependencies first",
                "Check for circular dependencies",
            ],
            **kwargs,
        )


class CircularDependencyError(TaskError):
    """تبعيات دائرية"""

    error_code = "GAAP_TSK_004"
    severity = "critical"

    def __init__(self, cycle: list[str], **kwargs: Any) -> None:
        super().__init__(
            message="Circular dependency detected",
            details={"cycle": " -> ".join(cycle)},
            suggestions=["Break the circular dependency by restructuring tasks"],
            recoverable=False,
            **kwargs,
        )


class TaskTimeoutError(TaskError):
    """انتهت مهلة المهمة"""

    error_code = "GAAP_TSK_005"
    recoverable = True

    def __init__(self, task_id: str, timeout_seconds: float, **kwargs: Any) -> None:
        super().__init__(
            message=f"Task '{task_id}' timed out after {timeout_seconds}s",
            details={"task_id": task_id, "timeout": timeout_seconds},
            suggestions=["Increase task timeout", "Simplify the task", "Use a faster model"],
            **kwargs,
        )


class TaskExecutionError(TaskError):
    """خطأ في تنفيذ المهمة"""

    error_code = "GAAP_TSK_006"

    def __init__(
        self, task_id: str, error_message: str, healing_level: str | None = None, **kwargs: Any
    ) -> None:
        suggestions = ["Retry the task"]
        if healing_level:
            suggestions.append(f"Self-healing attempted at level: {healing_level}")

        super().__init__(
            message=f"Task '{task_id}' execution failed: {error_message}",
            details={"task_id": task_id, "healing_level": healing_level},
            suggestions=suggestions,
            **kwargs,
        )


class MaxRetriesExceededError(TaskError):
    """تجاوز الحد الأقصى لإعادة المحاولة"""

    error_code = "GAAP_TSK_007"
    severity = "critical"

    def __init__(self, task_id: str, max_retries: int, last_error: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Task '{task_id}' exceeded max retries ({max_retries})",
            details={"task_id": task_id, "max_retries": max_retries, "last_error": last_error},
            suggestions=[
                "Review the task requirements",
                "Escalate to human intervention",
                "Try a different approach",
            ],
            recoverable=False,
            **kwargs,
        )


# =============================================================================
# Security Exceptions
# =============================================================================


class SecurityError(GAAPException):
    """خطأ أمني"""

    error_code = "GAAP_SEC_001"
    error_category = "security"
    severity = "critical"


class PromptInjectionError(SecurityError):
    """محاولة حقن تعليمات"""

    error_code = "GAAP_SEC_002"

    def __init__(self, detected_patterns: list[str], risk_score: float, **kwargs: Any) -> None:
        super().__init__(
            message="Potential prompt injection detected",
            details={"patterns": detected_patterns, "risk_score": risk_score},
            suggestions=[
                "Review the input for malicious content",
                "Sanitize the input before processing",
            ],
            recoverable=False,
            **kwargs,
        )


class CapabilityError(SecurityError):
    """خطأ في الصلاحيات"""

    error_code = "GAAP_SEC_003"

    def __init__(self, agent_id: str, resource: str, action: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Agent '{agent_id}' lacks capability for '{action}' on '{resource}'",
            details={"agent": agent_id, "resource": resource, "action": action},
            suggestions=["Request appropriate capability token", "Check agent permissions"],
            recoverable=False,
            **kwargs,
        )


class TokenExpiredError(SecurityError):
    """انتهت صلاحية التوكن"""

    error_code = "GAAP_SEC_004"
    recoverable = True

    def __init__(self, token_id: str, expired_at: datetime, **kwargs: Any) -> None:
        super().__init__(
            message=f"Capability token '{token_id}' expired at {expired_at}",
            details={"token_id": token_id, "expired_at": expired_at.isoformat()},
            suggestions=["Request a new capability token"],
            **kwargs,
        )


class SecurityScanError(SecurityError):
    """خطأ في الفحص الأمني"""

    error_code = "GAAP_SEC_005"

    def __init__(self, scan_type: str, findings: list[str], **kwargs: Any) -> None:
        super().__init__(
            message=f"Security scan '{scan_type}' found issues",
            details={"scan_type": scan_type, "findings": findings},
            suggestions=["Review and fix the security findings", "Consult security documentation"],
            **kwargs,
        )


class SandboxEscapeError(SecurityError):
    """محاولة الهروب من الـ Sandbox"""

    error_code = "GAAP_SEC_006"
    severity = "critical"

    def __init__(self, attempt_details: str, **kwargs: Any) -> None:
        super().__init__(
            message="Sandbox escape attempt detected",
            details={"attempt": attempt_details},
            suggestions=["Review the code for malicious patterns", "Report to security team"],
            recoverable=False,
            **kwargs,
        )


# =============================================================================
# Context Exceptions
# =============================================================================


class ContextError(GAAPException):
    """خطأ في السياق"""

    error_code = "GAAP_CTX_001"
    error_category = "context"
    severity = "error"


class ContextOverflowError(ContextError):
    """تجاوز سعة السياق"""

    error_code = "GAAP_CTX_002"

    def __init__(self, required: int, available: int, **kwargs: Any) -> None:
        super().__init__(
            message=f"Context overflow: {required} tokens required, {available} available",
            details={"required": required, "available": available},
            suggestions=[
                "Use hierarchical context loading",
                "Reduce context scope",
                "Use a model with larger context window",
            ],
            **kwargs,
        )


class ContextLoadError(ContextError):
    """خطأ في تحميل السياق"""

    error_code = "GAAP_CTX_003"

    def __init__(self, source: str, reason: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Failed to load context from '{source}'",
            details={"source": source, "reason": reason},
            suggestions=[
                "Check if the source exists",
                "Verify file permissions",
                "Check data format",
            ],
            **kwargs,
        )


class MemoryAccessError(ContextError):
    """خطأ في الوصول للذاكرة"""

    error_code = "GAAP_CTX_004"

    def __init__(self, memory_type: str, operation: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Memory access error: {operation} on {memory_type}",
            details={"memory_type": memory_type, "operation": operation},
            suggestions=["Check memory initialization", "Verify data integrity"],
            **kwargs,
        )


# =============================================================================
# MAD Exceptions
# =============================================================================


class MADError(GAAPException):
    """خطأ في نظام MAD"""

    error_code = "GAAP_MAD_001"
    error_category = "mad"
    severity = "error"


class ConsensusNotReachedError(MADError):
    """لم يتم الوصول لإجماع"""

    error_code = "GAAP_MAD_002"

    def __init__(self, rounds: int, scores: dict[str, float], **kwargs: Any) -> None:
        super().__init__(
            message=f"Consensus not reached after {rounds} rounds",
            details={"rounds": rounds, "scores": scores},
            suggestions=[
                "Review critic evaluations",
                "Adjust consensus threshold",
                "Add more debate rounds",
            ],
            recoverable=True,
            **kwargs,
        )


class CriticError(MADError):
    """خطأ في الناقد"""

    error_code = "GAAP_MAD_003"

    def __init__(self, critic_type: str, error: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Critic '{critic_type}' failed: {error}",
            details={"critic_type": critic_type, "error": error},
            suggestions=["Check critic configuration", "Review input quality"],
            **kwargs,
        )


# =============================================================================
# Healing Exceptions
# =============================================================================


class HealingError(GAAPException):
    """خطأ في التعافي الذاتي"""

    error_code = "GAAP_HLH_001"
    error_category = "healing"
    severity = "error"


class HealingFailedError(HealingError):
    """فشل التعافي الذاتي"""

    error_code = "GAAP_HLH_002"
    severity = "critical"

    def __init__(self, level: str, attempts: int, last_error: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Self-healing failed at level {level} after {attempts} attempts",
            details={"healing_level": level, "attempts": attempts, "last_error": last_error},
            suggestions=[
                "Escalate to human intervention",
                "Review task requirements",
                "Check system health",
            ],
            recoverable=False,
            **kwargs,
        )


class HumanEscalationError(HealingError):
    """خطأ في التصعيد البشري"""

    error_code = "GAAP_HLH_003"
    severity = "critical"

    def __init__(self, task_id: str, reason: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Task '{task_id}' requires human intervention: {reason}",
            details={"task_id": task_id, "reason": reason},
            suggestions=[
                "Review task details",
                "Provide manual resolution",
                "Update system knowledge",
            ],
            recoverable=False,
            **kwargs,
        )


# =============================================================================
# Plugin Exceptions
# =============================================================================


class PluginError(GAAPException):
    """خطأ في الإضافة"""

    error_code = "GAAP_PLG_001"
    error_category = "plugin"
    severity = "error"


class PluginLoadError(PluginError):
    """خطأ في تحميل الإضافة"""

    error_code = "GAAP_PLG_002"

    def __init__(self, plugin_name: str, reason: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Failed to load plugin '{plugin_name}'",
            details={"plugin": plugin_name, "reason": reason},
            suggestions=[
                "Check plugin dependencies",
                "Verify plugin compatibility",
                "Check plugin path",
            ],
            **kwargs,
        )


class PluginExecutionError(PluginError):
    """خطأ في تنفيذ الإضافة"""

    error_code = "GAAP_PLG_003"

    def __init__(self, plugin_name: str, error: str, **kwargs: Any) -> None:
        super().__init__(
            message=f"Plugin '{plugin_name}' execution failed: {error}",
            details={"plugin": plugin_name, "error": error},
            suggestions=["Check plugin logs", "Verify plugin inputs"],
            **kwargs,
        )


# =============================================================================
# Axiom Exceptions
# =============================================================================


class AxiomError(GAAPException):
    """خطأ في البديهيات"""

    error_code = "GAAP_AXM_001"
    error_category = "axiom"
    severity = "error"


class AxiomViolationError(AxiomError):
    """انتهاك بديهية"""

    error_code = "GAAP_AXM_002"

    def __init__(
        self,
        axiom_name: str,
        violation_details: str,
        task_id: str | None = None,
        severity_level: str = "medium",
        **kwargs: Any,
    ) -> None:
        severity = "warning" if severity_level == "low" else "error"
        super().__init__(
            message=f"Axiom '{axiom_name}' violated: {violation_details}",
            details={
                "axiom": axiom_name,
                "task_id": task_id,
                "severity_level": severity_level,
            },
            suggestions=[
                f"Fix the violation of axiom '{axiom_name}'",
                "Review the code against project constraints",
            ],
            **kwargs,
        )
        self.severity = severity


class SyntaxAxiomError(AxiomViolationError):
    """خطأ في بديهية الصيغة"""

    error_code = "GAAP_AXM_003"

    def __init__(self, syntax_error: str, code_snippet: str, **kwargs: Any) -> None:
        super().__init__(
            axiom_name="syntax",
            violation_details=f"Code does not parse: {syntax_error}",
            severity_level="low",
            **kwargs,
        )
        self.details["code_snippet"] = code_snippet[:200]


class DependencyAxiomError(AxiomViolationError):
    """خطأ في بديهية التبعيات"""

    error_code = "GAAP_AXM_004"

    def __init__(self, package_name: str, **kwargs: Any) -> None:
        super().__init__(
            axiom_name="dependency",
            violation_details=f"New package '{package_name}' added without L1 approval",
            severity_level="medium",
            **kwargs,
        )
        self.details["package"] = package_name


class InterfaceAxiomError(AxiomViolationError):
    """خطأ في بديهية الواجهة"""

    error_code = "GAAP_AXM_005"
    severity = "critical"

    def __init__(self, file_path: str, change_type: str, **kwargs: Any) -> None:
        super().__init__(
            axiom_name="interface",
            violation_details=f"Sensitive file '{file_path}' modified: {change_type}",
            severity_level="high",
            **kwargs,
        )
        self.details["file_path"] = file_path
        self.details["change_type"] = change_type


# =============================================================================
# Utility Functions
# =============================================================================


def wrap_exception(
    original: Exception,
    gaap_exception_class: type[GAAPException],
    message: str | None = None,
    **kwargs: Any,
) -> GAAPException:
    """
    تغليف استثناء عادي في استثناء GAAP

    Args:
        original: الاستثناء الأصلي
        gaap_exception_class: صنف استثناء GAAP
        message: رسالة مخصصة (اختياري)
        **kwargs: معاملات إضافية لاستثناء GAAP

    Returns:
        استثناء GAAP مُغلف
    """
    return gaap_exception_class(message=message or str(original), cause=original, **kwargs)


def is_recoverable(error: Exception) -> bool:
    """التحقق مما إذا كان الخطأ قابلاً للاستعادة"""
    if isinstance(error, GAAPException):
        return error.recoverable
    # الأخطاء العادية تعتبر قابلة للاستعادة افتراضياً
    return True


def get_error_severity(error: Exception) -> str:
    """الحصول على شدة الخطأ"""
    if isinstance(error, GAAPException):
        return error.severity
    return "error"
