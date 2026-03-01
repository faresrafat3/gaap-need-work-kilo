"""
Atomic Write Operations
=======================

Implements atomic file writes to prevent data corruption.

Pattern:
1. Write to temporary file {name}.tmp
2. Flush and sync to disk
3. Atomic rename to final path

This guarantees:
- File is either old version or complete new version
- Never corrupted partial writes
"""

import logging
import os
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Iterator, TextIO, TypeVar

from pydantic import BaseModel

logger = logging.getLogger("gaap.storage.atomic")

T = TypeVar("T", bound=BaseModel)


def atomic_write(
    path: Path | str,
    content: str | bytes,
    encoding: str = "utf-8",
    sync: bool = True,
) -> bool:
    """
    Write content to file atomically.

    Args:
        path: Target file path
        content: Content to write (string or bytes)
        encoding: Encoding for string content
        sync: Whether to sync to disk (fsync)

    Returns:
        True if successful
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, bytes):
            with open(os.fspath(tmp_path), "wb") as f:
                f.write(content)
                if sync:
                    f.flush()
                    os.fsync(f.fileno())
        else:
            with open(os.fspath(tmp_path), "w", encoding=encoding) as f:
                f.write(content)
                if sync:
                    f.flush()
                    os.fsync(f.fileno())

        if sync and path.exists():
            path_dir = path.parent
            try:
                fd = os.open(str(path_dir), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except (OSError, AttributeError):
                pass

        tmp_path.replace(path)

        return True

    except Exception as e:
        logger.error(f"Atomic write failed for {path}: {e}")

        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

        return False


@contextmanager
def atomic_writer(
    path: Path | str,
    mode: str = "w",
    encoding: str = "utf-8",
    sync: bool = True,
) -> Iterator[TextIO | BinaryIO]:
    """
    Context manager for atomic file writing.

    Args:
        path: Target file path
        mode: Write mode ('w' or 'wb')
        encoding: Encoding for text mode
        sync: Whether to sync to disk

    Yields:
        File handle for writing
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    f: TextIO | BinaryIO
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        if "b" in mode:
            f = open(os.fspath(tmp_path), "wb")
        else:
            f = open(os.fspath(tmp_path), "w", encoding=encoding)

        yield f

        if sync:
            f.flush()
            os.fsync(f.fileno())

        f.close()

        if sync and path.exists():
            path_dir = path.parent
            try:
                fd = os.open(str(path_dir), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except (OSError, AttributeError):
                pass

        tmp_path.replace(path)

    except Exception as e:
        logger.error(f"Atomic writer failed for {path}: {e}")

        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

        raise


class AtomicWriter:
    """
    Class-based atomic writer with backup support.

    Features:
    - Atomic writes
    - Automatic backups
    - Pydantic model validation
    - Rollback on failure
    """

    def __init__(
        self,
        base_dir: Path | str,
        backup_dir: Path | str | None = None,
        max_backups: int = 5,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.backup_dir = Path(backup_dir) if backup_dir else self.base_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.max_backups = max_backups

        self._logger = logger

    def write_json(
        self,
        name: str,
        data: dict[str, Any] | list[Any],
        backup: bool = False,
    ) -> bool:
        """Write JSON data atomically."""
        import json

        path = self.base_dir / f"{name}.json"

        if backup and path.exists():
            self._create_backup(path)

        content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
        return atomic_write(path, content)

    def write_model(
        self,
        name: str,
        model: BaseModel,
        backup: bool = False,
    ) -> bool:
        """Write Pydantic model atomically."""
        path = self.base_dir / f"{name}.json"

        if backup and path.exists():
            self._create_backup(path)

        content = model.model_dump_json(indent=2)
        return atomic_write(path, content)

    def read_json(self, name: str) -> dict[str, Any] | list[Any] | None:
        """Read JSON data."""
        import json

        path = self.base_dir / f"{name}.json"

        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)  # type: ignore[no-any-return]
        except Exception as e:
            self._logger.error(f"Failed to read {path}: {e}")
            return None

    def read_model(self, name: str, model_class: type[T]) -> T | None:
        """Read and validate Pydantic model."""
        import json

        path = self.base_dir / f"{name}.json"

        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            return model_class.model_validate(data)  # type: ignore[no-any-return]

        except Exception as e:
            self._logger.error(f"Failed to read/validate {path}: {e}")
            return None

    def _create_backup(self, path: Path) -> Path | None:
        """Create timestamped backup of file."""
        if not path.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{timestamp}_{path.name}"
        backup_path = self.backup_dir / backup_name

        try:
            shutil.copy2(path, backup_path)

            self._cleanup_old_backups(path.name)

            return backup_path

        except Exception as e:
            self._logger.error(f"Failed to create backup: {e}")
            return None

    def _cleanup_old_backups(self, original_name: str) -> None:
        """Remove old backups beyond max_backups."""
        backups = sorted(
            self.backup_dir.glob(f"*_{original_name}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old_backup in backups[self.max_backups :]:
            try:
                old_backup.unlink()
            except Exception:
                pass

    def delete(self, name: str) -> bool:
        """Delete a file."""
        path = self.base_dir / f"{name}.json"

        if not path.exists():
            return False

        try:
            self._create_backup(path)
            path.unlink()
            return True
        except Exception as e:
            self._logger.error(f"Failed to delete {path}: {e}")
            return False

    def exists(self, name: str) -> bool:
        """Check if file exists."""
        return (self.base_dir / f"{name}.json").exists()
