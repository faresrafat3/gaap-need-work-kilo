"""
Data Loss Prevention (DLP) Scanner
New Module - Evolution 2026
Implements: docs/evolution_plan_2026/39_SECURITY_AUDIT_SPEC.md
"""
import re
import math
import logging
from typing import Dict, List, Set

logger = logging.getLogger("gaap.security.dlp")

class DLPScanner:
    """
    Advanced scanner to prevent leakage of PII and Secrets.
    Uses regex + Shannon Entropy analysis.
    """
    
    # Known patterns for sensitive data
    PATTERNS = {
        "API_KEY": r"(?:[a-zA-Z0-9]{32,}|sk-[a-zA-Z0-9]{32,}|AIza[a-zA-Z0-9_-]{35})",
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        "PRIVATE_KEY": r"-----BEGIN [A-Z ]+ PRIVATE KEY-----",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b"
    }

    def __init__(self, entropy_threshold: float = 3.8):
        """
        threshold 3.8 is generally enough to catch base64/hex keys 
        while avoiding most natural language.
        """
        self.entropy_threshold = entropy_threshold

    def _calculate_entropy(self, text: str) -> float:
        """Calculates Shannon Entropy of a string."""
        if not text:
            return 0.0
        
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
        entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
        return entropy

    def scan_and_redact(self, text: str) -> str:
        """
        Scans outbound text and redacts sensitive information.
        """
        redacted_text = text
        
        # 1. Regex Redaction
        for label, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, redacted_text)
            for match in matches:
                val = match.group()
                # Double check entropy for generic API_KEY pattern to avoid false positives
                if label == "API_KEY" and self._calculate_entropy(val) < self.entropy_threshold:
                    continue
                redacted_text = redacted_text.replace(val, f"[REDACTED_{label}]")

        # 2. Heuristic Entropy Scan (Catch unknown secrets)
        # Scan words longer than 20 chars
        words = re.findall(r'\b[A-Za-z0-9+/=_-]{20,}\b', redacted_text)
        for word in words:
            if self._calculate_entropy(word) > self.entropy_threshold:
                # Potential token/key
                redacted_text = redacted_text.replace(word, "[REDACTED_SECRET_TOKEN]")
                logger.warning(f"DLP: High-entropy string detected and redacted.")

        return redacted_text

    def audit_leaks(self, text: str) -> List[Dict[str, str]]:
        """
        Returns a list of detected potential leaks without redacting them.
        Used for the Internal Auditor role.
        """
        leaks = []
        for label, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            for m in matches:
                leaks.append({"type": label, "value": m})
        return leaks
