"""
Memory Guard — OOM Kill Prevention
====================================

Problem: Python tests grew to 11.4GB RSS → Linux OOM killer killed the process
         → VS Code crashed because python was in VS Code's cgroup.

Solution: Hard memory limit + monitoring + early abort.

Usage:
    from gaap.core.memory_guard import MemoryGuard, set_memory_limit

    # Set hard limit (rlimit) — kernel kills process cleanly if exceeded
    set_memory_limit(max_gb=6.0)

    # Or use the guard for periodic monitoring + early warning
    guard = MemoryGuard(max_rss_mb=4096, warn_rss_mb=2048)
    guard.check()  # raises MemoryError if over limit
"""

import gc
import logging
import resource

logger = logging.getLogger("gaap.memory_guard")


def get_rss_mb() -> float:
    """Get current RSS (Resident Set Size) in MB."""
    # /proc/self/status is more accurate than resource.getrusage on Linux
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0  # kB → MB
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    # Fallback to getrusage (returns max RSS, not current)
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def set_memory_limit(max_gb: float = 6.0) -> None:
    """
    Set a hard RSS limit via RLIMIT_AS (address space).

    If the python process tries to allocate beyond this, malloc() returns NULL
    and Python raises MemoryError — which is clean and recoverable,
    unlike the OOM killer which sends SIGKILL (instant death, no cleanup).

    Args:
        max_gb: Maximum virtual memory in GB. Default 6GB leaves room for
                VS Code (~3GB) + system (~2GB) on a 16GB machine.
    """
    max_bytes = int(max_gb * 1024 * 1024 * 1024)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        logger.info(f"Memory limit set: {max_gb:.1f}GB (was soft={soft}, hard={hard})")
    except (OSError, ValueError) as e:
        logger.warning(f"Could not set memory limit: {e}")


class MemoryGuard:
    """
    Periodic memory monitor with early abort.

    Call guard.check() between operations (LLM calls, task processing).
    If RSS exceeds the limit, it:
      1. Runs gc.collect() aggressively
      2. Rechecks
      3. Raises MemoryError if still over (allows clean shutdown)
    """

    def __init__(
        self,
        max_rss_mb: float = 4096,
        warn_rss_mb: float = 2048,
        gc_threshold_mb: float = 1024,
    ):
        self.max_rss_mb = max_rss_mb
        self.warn_rss_mb = warn_rss_mb
        self.gc_threshold_mb = gc_threshold_mb
        self._warned = False
        self._checks = 0
        self._gc_runs = 0

    def check(self, context: str = "") -> float:
        """
        Check current memory. Returns RSS in MB.

        Raises MemoryError if over hard limit after GC.
        """
        self._checks += 1
        rss = get_rss_mb()

        # Auto-GC when passing threshold
        if rss > self.gc_threshold_mb:
            gc.collect()
            self._gc_runs += 1
            rss = get_rss_mb()  # recheck after GC

        # Warning zone
        if rss > self.warn_rss_mb and not self._warned:
            self._warned = True
            logger.warning(
                f"Memory warning: {rss:.0f}MB (limit={self.max_rss_mb:.0f}MB) {context}"
            )

        # Hard limit
        if rss > self.max_rss_mb:
            # Last-resort aggressive GC
            gc.collect(generation=2)
            rss = get_rss_mb()

            if rss > self.max_rss_mb:
                msg = (
                    f"Memory limit exceeded: {rss:.0f}MB > {self.max_rss_mb:.0f}MB. "
                    f"Aborting to prevent OOM kill. {context}"
                )
                logger.critical(msg)
                raise MemoryError(msg)

        return rss

    def get_stats(self) -> dict:
        return {
            "current_rss_mb": round(get_rss_mb(), 1),
            "max_rss_mb": self.max_rss_mb,
            "checks": self._checks,
            "gc_runs": self._gc_runs,
        }


# Module-level singleton
_guard: MemoryGuard | None = None


def get_guard(
    max_rss_mb: float = 4096,
    warn_rss_mb: float = 2048,
) -> MemoryGuard:
    """Get or create the global MemoryGuard singleton."""
    global _guard
    if _guard is None:
        _guard = MemoryGuard(max_rss_mb=max_rss_mb, warn_rss_mb=warn_rss_mb)
    return _guard
