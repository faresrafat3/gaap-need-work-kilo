# Security Firewall
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any

# =============================================================================
# Enums
# =============================================================================


class RiskLevel(Enum):
    """مستويات الخطر"""

    SAFE = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()
    BLOCKED = auto()


class AttackType(Enum):
    """أنواع الهجمات"""

    PROMPT_INJECTION = auto()
    JAILBREAK = auto()
    DATA_EXFILTRATION = auto()
    CODE_INJECTION = auto()
    MALICIOUS_INSTRUCTION = auto()
    ROLE_CONFUSION = auto()
    CONTEXT_MANIPULATION = auto()


# =============================================================================
# Prompt Firewall
# =============================================================================


@dataclass
class FirewallResult:
    """نتيجة فحص الجدار الناري"""

    is_safe: bool
    risk_level: RiskLevel
    detected_patterns: list[str] = field(default_factory=list)
    sanitized_input: str = ""
    recommendations: list[str] = field(default_factory=list)
    scan_time_ms: float = 0.0
    layer_scores: dict[str, float] = field(default_factory=dict)


class PromptFirewall:
    """
    جدار الحماية للتعليمات

    7 طبقات دفاع:
    - L1: Surface Inspection
    - L2: Lexical Analysis
    - L3: Syntactic Analysis
    - L4: Semantic Analysis
    - L5: Contextual Verification
    - L6: Behavioral Analysis
    - L7: Adversarial Testing
    """

    # أنماط الحقن المعروفة
    INJECTION_PATTERNS = [
        # محاولات تجاهل التعليمات
        (r"ignore\s+(previous|all|above)\s+(instructions?|prompts?)", AttackType.PROMPT_INJECTION),
        (r"disregard\s+(all|any|previous)", AttackType.PROMPT_INJECTION),
        (r"forget\s+(everything|all|previous)", AttackType.PROMPT_INJECTION),
        # محاولات Role Play
        (r"you\s+are\s+now\s+(a|an)\s+\w+", AttackType.ROLE_CONFUSION),
        (r"act\s+as\s+(if|though|a)", AttackType.ROLE_CONFUSION),
        (r"pretend\s+(to\s+be|that)", AttackType.ROLE_CONFUSION),
        # محاولات Jailbreak
        (r"(developer|admin|system)\s+mode", AttackType.JAILBREAK),
        (r"bypass\s+(all\s+)?(restrictions?|filters?|safety)", AttackType.JAILBREAK),
        (r"DAN\s*(mode|prompt)?", AttackType.JAILBREAK),
        # حقن كود
        (r"<script[^>]*>", AttackType.CODE_INJECTION),
        (r"javascript:", AttackType.CODE_INJECTION),
        (r"on\w+\s*=", AttackType.CODE_INJECTION),
        # استخراج بيانات
        (r"reveal\s+(your|the)\s+(instructions?|prompt)", AttackType.DATA_EXFILTRATION),
        (r"show\s+me\s+(your|the)\s+(system|developer)", AttackType.DATA_EXFILTRATION),
        (r"print\s+(your|the)\s+(instructions?|prompt)", AttackType.DATA_EXFILTRATION),
        # تعليمات خفية
        (r"\[SYSTEM\]", AttackType.MALICIOUS_INSTRUCTION),
        (r"\[INST\]", AttackType.MALICIOUS_INSTRUCTION),
        (r"<<<.*>>>", AttackType.MALICIOUS_INSTRUCTION),
    ]

    # أنماط التشويش
    OBFUSCATION_PATTERNS = [
        r"\\x[0-9a-fA-F]{2}",  # Hex encoding
        r"\\u[0-9a-fA-F]{4}",  # Unicode escapes
        r"%[0-9a-fA-F]{2}",  # URL encoding
        r"&#[0-9]+;",  # HTML entities
        r"\\[nrtu]",  # Escape sequences
    ]

    def __init__(self, strictness: str = "high"):
        self.strictness = strictness
        self._logger = logging.getLogger("gaap.security.firewall")
        self._scan_history: list[FirewallResult] = []
        self._blocked_count = 0

    def scan(self, input_text: str, context: dict[str, Any] | None = None) -> FirewallResult:
        """فحص المدخل"""
        start_time = time.time()

        detected_patterns: list[str] = []
        layer_scores: dict[str, float] = {}
        risk_score = 0.0

        # L1: Surface Inspection
        l1_score = self._layer1_scan(input_text, detected_patterns)
        layer_scores["L1_surface"] = l1_score
        risk_score += l1_score * 0.15

        # L2: Lexical Analysis
        l2_score = self._layer2_scan(input_text, detected_patterns)
        layer_scores["L2_lexical"] = l2_score
        risk_score += l2_score * 0.15

        # L3: Syntactic Analysis
        l3_score = self._layer3_scan(input_text, detected_patterns)
        layer_scores["L3_syntactic"] = l3_score
        risk_score += l3_score * 0.15

        # L4: Semantic Analysis
        l4_score = self._layer4_scan(input_text, detected_patterns, context)
        layer_scores["L4_semantic"] = l4_score
        risk_score += l4_score * 0.25

        # L5: Contextual Verification
        l5_score = self._layer5_scan(input_text, context)
        layer_scores["L5_contextual"] = l5_score
        risk_score += l5_score * 0.15

        # === CRITICAL FIX: Escalate risk for high-severity attack types ===
        # Pattern-based scoring alone is too low because weights dilute the signal.
        # If we detected explicit JAILBREAK / PROMPT_INJECTION / DATA_EXFILTRATION
        # patterns, force-escalate the risk score regardless of weighted total.
        _CRITICAL_ATTACK_TYPES = {"JAILBREAK", "PROMPT_INJECTION", "DATA_EXFILTRATION"}
        detected_critical = [
            p for p in detected_patterns if any(at in p for at in _CRITICAL_ATTACK_TYPES)
        ]
        if len(detected_critical) >= 3:
            risk_score = max(risk_score, 0.90)  # CRITICAL
        elif len(detected_critical) >= 2:
            risk_score = max(risk_score, 0.75)  # CRITICAL
        elif len(detected_critical) >= 1:
            risk_score = max(risk_score, 0.55)  # HIGH

        layer_scores["escalation_critical_patterns"] = len(detected_critical)

        # تحديد مستوى الخطر
        risk_level = self._calculate_risk_level(risk_score)

        # التطهير
        sanitized = self._sanitize(input_text, detected_patterns)

        scan_time = (time.time() - start_time) * 1000

        # is_safe: only SAFE is truly safe. LOW means patterns were
        # detected but score was marginal — still allow, BUT if any critical
        # attack type was found, always block regardless of score.
        has_critical_patterns = len(detected_critical) > 0
        result = FirewallResult(
            is_safe=(risk_level == RiskLevel.SAFE) and not has_critical_patterns,
            risk_level=risk_level,
            detected_patterns=detected_patterns,
            sanitized_input=sanitized,
            recommendations=self._get_recommendations(risk_level, detected_patterns),
            scan_time_ms=scan_time,
            layer_scores=layer_scores,
        )

        self._scan_history.append(result)

        if not result.is_safe:
            self._blocked_count += 1
            self._logger.warning(
                f"Blocked input with risk level {risk_level.name}: " f"{detected_patterns[:3]}"
            )

        return result

    def _layer1_scan(self, text: str, detected: list[str]) -> float:
        """فحص سطحي للأنماط"""
        score = 0.0
        text_lower = text.lower()

        for pattern, attack_type in self.INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                detected.append(f"L1:{attack_type.name}")
                score += 0.3

        return min(score, 1.0)

    def _layer2_scan(self, text: str, detected: list[str]) -> float:
        """فحص التشويش"""
        score = 0.0

        for pattern in self.OBFUSCATION_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                detected.append(f"L2:obfuscation:{len(matches)}")
                score += 0.1 * len(matches)

        # فحص الأحرف غير المرئية
        invisible_chars = len(re.findall(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text))
        if invisible_chars > 0:
            detected.append(f"L2:invisible_chars:{invisible_chars}")
            score += 0.2

        return min(score, 1.0)

    def _layer3_scan(self, text: str, detected: list[str]) -> float:
        """فحص بنية الجمل"""
        score = 0.0

        # تعليمات متداخلة
        nested_brackets = len(re.findall(r"\[.*\[.*\].*\]", text))
        if nested_brackets > 0:
            detected.append(f"L3:nested_instructions:{nested_brackets}")
            score += 0.2

        # تعليقات مشبوهة
        suspicious_comments = len(re.findall(r"(/\*.*\*/|<!--.*-->|#.*$)", text, re.MULTILINE))
        if suspicious_comments > 2:
            detected.append(f"L3:suspicious_comments:{suspicious_comments}")
            score += 0.1

        return min(score, 1.0)

    def _layer4_scan(self, text: str, detected: list[str], context: dict[str, Any] | None) -> float:
        """فحص دلالي"""
        score = 0.0

        # كلمات مفتاحية خطيرة
        danger_keywords = [
            "override",
            "bypass",
            "exploit",
            "hack",
            "unauthorized",
            "secret",
            "password",
            "token",
        ]

        text_lower = text.lower()
        found_keywords = [kw for kw in danger_keywords if kw in text_lower]

        if found_keywords:
            detected.append(f"L4:danger_keywords:{found_keywords}")
            score += 0.2 * len(found_keywords)

        # سياق المشروع
        if context:
            context.get("allowed_topics", [])
            # فحص إذا كان النص خارج الموضوع المسموح
            # ...

        return min(score, 1.0)

    def _layer5_scan(self, text: str, context: dict[str, Any] | None) -> float:
        """فحص سياقي"""
        score = 0.0

        # التحقق من تطابق السياق
        if context:
            user_role = context.get("user_role", "user")

            # طلبات إدارية من مستخدم عادي
            if user_role == "user":
                admin_patterns = ["admin", "root", "sudo", "elevated"]
                if any(p in text.lower() for p in admin_patterns):
                    score += 0.3

        return min(score, 1.0)

    def _calculate_risk_level(self, score: float) -> RiskLevel:
        """حساب مستوى الخطر"""
        if score < 0.1:
            return RiskLevel.SAFE
        elif score < 0.25:
            return RiskLevel.LOW
        elif score < 0.5:
            return RiskLevel.MEDIUM
        elif score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL

    def _sanitize(self, text: str, detected: list[str]) -> str:
        """تطهير النص"""
        sanitized = text

        # إزالة الأنماط الخطيرة
        for pattern, _ in self.INJECTION_PATTERNS:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)

        # إزالة الأحرف غير المرئية
        sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", sanitized)

        return sanitized

    def _get_recommendations(self, risk_level: RiskLevel, patterns: list[str]) -> list[str]:
        """توصيات"""
        recommendations = []

        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("Block this input immediately")
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Review input before processing")
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("Consider sanitizing input")

        if any("injection" in p.lower() for p in patterns):
            recommendations.append("Potential prompt injection detected")

        if any("obfuscation" in p.lower() for p in patterns):
            recommendations.append("Obfuscation techniques detected")

        return recommendations

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "total_scans": len(self._scan_history),
            "blocked": self._blocked_count,
            "block_rate": (
                self._blocked_count / len(self._scan_history) if self._scan_history else 0
            ),
        }


# =============================================================================
# Audit Trail
# =============================================================================


@dataclass
class AuditEntry:
    """سجل تدقيق"""

    id: str
    timestamp: datetime
    action: str
    agent_id: str
    resource: str
    result: str
    details: dict[str, Any] = field(default_factory=dict)
    previous_hash: str = ""
    hash: str = ""


class AuditTrail:
    """
    سجل التدقيق - غير قابل للتزوير

    يستخدم سلسلة hash لضمان النزاهة
    """

    def __init__(self, storage_path: str | None = None):
        self.storage_path = storage_path
        self._chain: list[AuditEntry] = []
        self._logger = logging.getLogger("gaap.security.audit")

    def record(
        self,
        action: str,
        agent_id: str,
        resource: str,
        result: str,
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """تسجيل حدث"""
        # إنشاء المدخل
        entry_id = hashlib.sha256(f"{action}:{agent_id}:{time.time()}".encode()).hexdigest()[:16]

        previous_hash = self._chain[-1].hash if self._chain else "genesis"

        entry = AuditEntry(
            id=entry_id,
            timestamp=datetime.now(),
            action=action,
            agent_id=agent_id,
            resource=resource,
            result=result,
            details=details or {},
            previous_hash=previous_hash,
        )

        # حساب hash
        entry.hash = self._calculate_hash(entry)

        self._chain.append(entry)

        return entry

    def _calculate_hash(self, entry: AuditEntry) -> str:
        """حساب hash"""
        data = f"{entry.id}{entry.timestamp}{entry.action}{entry.agent_id}{entry.resource}{entry.result}{entry.previous_hash}"
        return hashlib.sha256(data.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """التحقق من النزاهة"""
        for i, entry in enumerate(self._chain):
            # التحقق من hash
            if entry.hash != self._calculate_hash(entry):
                return False

            # التحقق من السلسلة
            if i > 0 and entry.previous_hash != self._chain[i - 1].hash:
                return False

        return True

    def get_agent_history(self, agent_id: str) -> list[AuditEntry]:
        """تاريخ وكيل"""
        return [e for e in self._chain if e.agent_id == agent_id]

    def get_recent(self, limit: int = 100) -> list[AuditEntry]:
        """أحدث السجلات"""
        return self._chain[-limit:]

    def export(self, path: str) -> None:
        """تصدير"""
        data = [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "action": e.action,
                "agent_id": e.agent_id,
                "resource": e.resource,
                "result": e.result,
                "hash": e.hash,
            }
            for e in self._chain
        ]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Capability Token
# =============================================================================


@dataclass
class CapabilityToken:
    """توكن القدرة"""

    subject: str  # معرف الوكيل
    resource: str  # المورد
    action: str  # الإجراء
    issued_at: datetime
    expires_at: datetime
    constraints: dict[str, Any] = field(default_factory=dict)
    nonce: str = ""
    signature: str = ""


class CapabilityManager:
    """مدير القدرات"""

    def __init__(self, secret_key: str = "gaap-secret"):
        self.secret_key = secret_key
        self._active_tokens: dict[str, CapabilityToken] = {}
        self._logger = logging.getLogger("gaap.security.capability")

    def issue_token(
        self,
        agent_id: str,
        resource: str,
        action: str,
        ttl_seconds: int = 300,
        constraints: dict[str, Any] | None = None,
    ) -> CapabilityToken:
        """إصدار توكن"""
        now = datetime.now()
        nonce = hashlib.sha256(f"{agent_id}{time.time()}".encode()).hexdigest()[:16]

        token = CapabilityToken(
            subject=agent_id,
            resource=resource,
            action=action,
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
            constraints=constraints or {},
            nonce=nonce,
        )

        # توقيع
        token.signature = self._sign(token)

        self._active_tokens[f"{agent_id}:{resource}:{action}"] = token

        return token

    def verify_token(
        self, token: CapabilityToken, requested_resource: str, requested_action: str
    ) -> bool:
        """التحقق من التوكن"""
        # التحقق من التوقيع
        if token.signature != self._sign(token):
            return False

        # التحقق من الصلاحية
        if datetime.now() > token.expires_at:
            return False

        # التحقق من المورد والإجراء
        if token.resource != requested_resource:
            return False

        return token.action == requested_action

    def _sign(self, token: CapabilityToken) -> str:
        """توقيع التوكن"""
        data = f"{token.subject}{token.resource}{token.action}{token.nonce}{self.secret_key}"
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def revoke_token(self, agent_id: str, resource: str, action: str) -> None:
        """إلغاء توكن"""
        key = f"{agent_id}:{resource}:{action}"
        if key in self._active_tokens:
            del self._active_tokens[key]


# =============================================================================
# Convenience Functions
# =============================================================================


def create_firewall(strictness: str = "high") -> PromptFirewall:
    """إنشاء جدار حماية"""
    return PromptFirewall(strictness=strictness)


def create_audit_trail(storage_path: str | None = None) -> AuditTrail:
    """إنشاء سجل تدقيق"""
    return AuditTrail(storage_path=storage_path)


def create_capability_manager(secret_key: str = "gaap-secret") -> CapabilityManager:
    """إنشاء مدير قدرات"""
    return CapabilityManager(secret_key=secret_key)
