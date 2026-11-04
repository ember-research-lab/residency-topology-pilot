"""cap - fail-closed memory-cap detection for model jobs (CLAUDE.md rule 1).

A model job must run inside `systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0` (cgroup v2: memory.max <= 16 GiB and
memory.swap.max == 0) or, as a fallback, under an address-space rlimit (RLIMIT_AS <= 20 GiB). `require_cap()` raises CapError otherwise.
There is no override."""
from __future__ import annotations

import pathlib
import resource

GIB = 2**30


class CapError(RuntimeError):
    """No enforced memory cap detected from inside the process."""


def detect() -> str | None:
    """Return 'cgroup' or 'rlimit' if a cap is live, else None."""
    try:
        line = [l for l in open("/proc/self/cgroup") if l.startswith("0::")][0].strip()[3:]
        d = pathlib.Path("/sys/fs/cgroup") / line.lstrip("/")
        mx, sw = (d / "memory.max").read_text().strip(), (d / "memory.swap.max").read_text().strip()
        if mx != "max" and int(mx) <= 16 * GIB and sw == "0":
            return "cgroup"
    except (OSError, IndexError, ValueError):
        pass
    soft, _ = resource.getrlimit(resource.RLIMIT_AS)
    if soft != resource.RLIM_INFINITY and soft <= 20 * GIB:
        return "rlimit"
    return None


def require_cap() -> str:
    m = detect()
    if m is None:
        raise CapError("no memory cap (cgroup MemoryMax<=16G with swap 0, or RLIMIT_AS<=20G): run under "
                       "`systemd-run --user --scope --quiet -p MemoryMax=16G -p MemorySwapMax=0 -- ...`")
    return m
