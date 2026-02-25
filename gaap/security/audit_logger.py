"""
Audit Logger - Persistent Audit Trail
Implements: docs/evolution_plan_2026/39_SECURITY_AUDIT_SPEC.md

Features:
- Disk persistence (.gaap/audit/log.jsonl)
- Forward-secure logging with hash chain
- Automatic rotation (every N entries)
- Tamper detection
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("gaap.security.audit_logger")


@dataclass
class AuditLogEntry:
    id: str
    timestamp: str
    action: str
    agent_id: str
    resource: str
    result: str
    details: dict[str, Any] = field(default_factory=dict)
    previous_hash: str = ""
    entry_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "action": self.action,
            "agent_id": self.agent_id,
            "resource": self.resource,
            "result": self.result,
            "details": self.details,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditLogEntry:
        return cls(
            id=data.get("id", ""),
            timestamp=data.get("timestamp", ""),
            action=data.get("action", ""),
            agent_id=data.get("agent_id", ""),
            resource=data.get("resource", ""),
            result=data.get("result", ""),
            details=data.get("details", {}),
            previous_hash=data.get("previous_hash", ""),
            entry_hash=data.get("entry_hash", ""),
        )


@dataclass
class AuditLoggerConfig:
    log_dir: Path | str = ".gaap/audit"
    log_file: str = "log.jsonl"
    state_file: str = "state.json"
    rotation_size: int = 100
    max_backups: int = 10
    auto_flush: bool = True
    flush_interval: int = 10

    @classmethod
    def default(cls) -> AuditLoggerConfig:
        return cls()

    @classmethod
    def strict(cls) -> AuditLoggerConfig:
        return cls(rotation_size=50, flush_interval=5)

    @classmethod
    def development(cls) -> AuditLoggerConfig:
        return cls(rotation_size=200, max_backups=3)


class AuditLogger:
    """
    Persistent audit logger with hash chain verification.

    Features:
    - JSONL format (one entry per line)
    - Hash chain for tamper detection
    - Automatic rotation
    - Thread-safe operations

    Usage:
        logger = AuditLogger()
        entry = logger.log("api_call", "agent_1", "/users", "success")
        print(entry.entry_hash)
    """

    def __init__(self, config: AuditLoggerConfig | None = None) -> None:
        self.config = config or AuditLoggerConfig.default()
        self.log_dir = Path(self.config.log_dir)
        self.log_path = self.log_dir / self.config.log_file
        self.state_path = self.log_dir / self.config.state_file

        self._lock = threading.Lock()
        self._buffer: list[AuditLogEntry] = []
        self._entries_since_flush = 0
        self._last_hash = self._load_last_hash()
        self._entry_count = 0

        self._ensure_dir()
        self._load_entry_count()

        self._logger = logger

    def _ensure_dir(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _load_last_hash(self) -> str:
        if self.state_path.exists():
            try:
                with open(self.state_path, encoding="utf-8") as f:
                    state = json.load(f)
                    return state.get("last_hash", "genesis")  # type: ignore[no-any-return]
            except Exception:
                pass
        return "genesis"

    def _load_entry_count(self) -> None:
        if self.log_path.exists():
            try:
                with open(self.log_path, encoding="utf-8") as f:
                    self._entry_count = sum(1 for _ in f)
            except Exception:
                self._entry_count = 0

    def _save_state(self, last_hash: str, entry_count: int) -> None:
        state = {
            "last_hash": last_hash,
            "entry_count": entry_count,
            "updated_at": datetime.now().isoformat(),
        }
        tmp_path = self.state_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(self.state_path)

    def _calculate_hash(self, entry: AuditLogEntry) -> str:
        data = (
            f"{entry.id}{entry.timestamp}{entry.action}"
            f"{entry.agent_id}{entry.resource}{entry.result}"
            f"{entry.previous_hash}"
        )
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    def _generate_id(self) -> str:
        import uuid

        return uuid.uuid4().hex[:12]

    def log(
        self,
        action: str,
        agent_id: str,
        resource: str,
        result: str,
        details: dict[str, Any] | None = None,
    ) -> AuditLogEntry:
        entry = AuditLogEntry(
            id=self._generate_id(),
            timestamp=datetime.now().isoformat(),
            action=action,
            agent_id=agent_id,
            resource=resource,
            result=result,
            details=details or {},
            previous_hash=self._last_hash,
        )

        entry.entry_hash = self._calculate_hash(entry)

        with self._lock:
            self._buffer.append(entry)
            self._last_hash = entry.entry_hash
            self._entry_count += 1
            self._entries_since_flush += 1

            if self.config.auto_flush and self._entries_since_flush >= self.config.flush_interval:
                self._flush_buffer()
                self._save_state(self._last_hash, self._entry_count)

                if self._entry_count >= self.config.rotation_size:
                    self._rotate_log()

        self._logger.debug(f"Audit logged: {action} by {agent_id}")
        return entry

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return

        with open(self.log_path, "a", encoding="utf-8") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        self._buffer.clear()
        self._entries_since_flush = 0

    def _rotate_log(self) -> None:
        if not self.log_path.exists():
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"log_{timestamp}.jsonl"
        backup_path = self.log_dir / backup_name

        self._flush_buffer()

        self.log_path.rename(backup_path)
        self._logger.info(f"Rotated audit log to {backup_name}")

        self._cleanup_old_backups()

        self._entry_count = 0

    def _cleanup_old_backups(self) -> None:
        backups = sorted(
            self.log_dir.glob("log_*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old_backup in backups[self.config.max_backups :]:
            try:
                old_backup.unlink()
                self._logger.debug(f"Removed old backup: {old_backup.name}")
            except Exception:
                pass

    def verify_integrity(self) -> tuple[bool, list[str]]:
        errors: list[str] = []

        if not self.log_path.exists():
            return True, []

        previous_hash = "genesis"
        line_num = 0

        try:
            with open(self.log_path, encoding="utf-8") as f:
                for line in f:
                    line_num += 1

                    try:
                        data = json.loads(line.strip())
                        entry = AuditLogEntry.from_dict(data)
                    except json.JSONDecodeError as e:
                        errors.append(f"Line {line_num}: Invalid JSON - {e}")
                        continue

                    if entry.previous_hash != previous_hash:
                        errors.append(
                            f"Line {line_num}: Hash chain broken. "
                            f"Expected {previous_hash[:8]}, got {entry.previous_hash[:8]}"
                        )

                    expected_hash = self._calculate_hash(entry)
                    if entry.entry_hash != expected_hash:
                        errors.append(
                            f"Line {line_num}: Entry hash mismatch. "
                            f"Entry may have been tampered with."
                        )

                    previous_hash = entry.entry_hash
        except Exception as e:
            errors.append(f"Error reading log file: {e}")
            return False, errors

        is_valid = len(errors) == 0
        return is_valid, errors

    def query(
        self,
        action: str | None = None,
        agent_id: str | None = None,
        resource: str | None = None,
        result: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditLogEntry]:
        entries: list[AuditLogEntry] = []

        if not self.log_path.exists():
            return entries

        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    entry = AuditLogEntry.from_dict(data)

                    if action and entry.action != action:
                        continue
                    if agent_id and entry.agent_id != agent_id:
                        continue
                    if resource and entry.resource != resource:
                        continue
                    if result and entry.result != result:
                        continue

                    if start_time or end_time:
                        try:
                            entry_time = datetime.fromisoformat(entry.timestamp)
                            if start_time and entry_time < start_time:
                                continue
                            if end_time and entry_time > end_time:
                                continue
                        except ValueError:
                            continue

                    entries.append(entry)

                    if len(entries) >= limit:
                        break
                except json.JSONDecodeError:
                    continue

        return entries

    def get_stats(self) -> dict[str, Any]:
        return {
            "entry_count": self._entry_count,
            "buffer_size": len(self._buffer),
            "last_hash": self._last_hash[:16] if self._last_hash else None,
            "log_file": str(self.log_path),
            "rotation_size": self.config.rotation_size,
        }

    def flush(self) -> None:
        with self._lock:
            self._flush_buffer()
            self._save_state(self._last_hash, self._entry_count)

    def export(self, output_path: Path | str, format: str = "jsonl") -> int:
        output = Path(output_path)
        count = 0

        with open(output, "w", encoding="utf-8") as out_f:
            if format == "json":
                out_f.write("[\n")
                first = True

            if self.log_path.exists():
                with open(self.log_path, encoding="utf-8") as in_f:
                    for line in in_f:
                        if format == "json":
                            if not first:
                                out_f.write(",\n")
                            first = False
                            out_f.write("  " + line.strip())
                        else:
                            out_f.write(line)
                        count += 1

            if format == "json":
                out_f.write("\n]")

        return count


def create_audit_logger(
    log_dir: str = ".gaap/audit",
    rotation_size: int = 100,
) -> AuditLogger:
    config = AuditLoggerConfig(
        log_dir=log_dir,
        rotation_size=rotation_size,
    )
    return AuditLogger(config)
