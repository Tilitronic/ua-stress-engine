#!/usr/bin/env python3
"""
Git LFS Storage Monitor

Checks current remote LFS usage AND objects about to be pushed.
Blocks the push if the combined total would exceed the free-tier limit.

Wired into .git/hooks/pre-push — runs automatically on every `git push`.
Can also be called manually:
    python scripts/lfs_storage_monitor.py [--limit-mb N]
"""

import re
import subprocess
import sys
from argparse import ArgumentParser


FREE_TIER_MB = 1024  # GitHub free plan


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_size_mb(size_str: str) -> float:
    """Parse a size string like '259.0 MB' or '512 KB' into MB."""
    m = re.search(r'([\d.]+)\s*(MB|KB|GB|B)', size_str, re.I)
    if not m:
        return 0.0
    value, unit = float(m.group(1)), m.group(2).upper()
    return {'GB': value * 1024, 'MB': value, 'KB': value / 1024, 'B': value / 1024 / 1024}[unit]


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


# ── size queries ─────────────────────────────────────────────────────────────

def remote_lfs_usage_mb() -> tuple[float, list[tuple[str, float]]]:
    """
    Return (total_mb, [(path, size_mb), ...]) for LFS objects already on remote.
    Uses `git lfs ls-files --all --size` — lists every OID ever tracked.
    """
    out = _run(['git', 'lfs', 'ls-files', '--all', '--size'])
    files: list[tuple[str, float]] = []
    for line in out.splitlines():
        # format: "abc123def * path/to/file (12.3 MB)"
        m = re.match(r'\w+\s+[*-]\s+(.+?)\s+\((.+?)\)', line)
        if m:
            files.append((m.group(1), _parse_size_mb(m.group(2))))
    return sum(s for _, s in files), files


def pending_lfs_objects_mb() -> tuple[float, list[tuple[str, float]]]:
    """
    Return (total_mb, [(path, size_mb), ...]) for LFS objects staged locally
    but not yet pushed to the remote (i.e. present in local .git/lfs/objects
    but not confirmed on the server).

    Strategy: compare `git lfs ls-files --size` (current HEAD + index)
    against `git lfs ls-files --all --size` — the difference is what's local-only.
    """
    # All known OIDs (remote + local)
    all_out  = _run(['git', 'lfs', 'ls-files', '--all',  '--size'])
    # OIDs in current HEAD / index
    head_out = _run(['git', 'lfs', 'ls-files',           '--size'])

    def parse(text: str) -> dict[str, tuple[str, float]]:
        result: dict[str, tuple[str, float]] = {}
        for line in text.splitlines():
            m = re.match(r'(\w+)\s+[*-]\s+(.+?)\s+\((.+?)\)', line)
            if m:
                oid, path, size = m.group(1), m.group(2), m.group(3)
                result[oid] = (path, _parse_size_mb(size))
        return result

    all_oids  = parse(all_out)
    head_oids = parse(head_out)

    # OIDs in HEAD but not yet confirmed as pushed
    # git lfs status shows files not yet uploaded
    status_out = _run(['git', 'lfs', 'status'])
    pending: list[tuple[str, float]] = []
    for oid, (path, size_mb) in head_oids.items():
        if oid not in all_oids:
            pending.append((path, size_mb))

    # Supplement with `git lfs status` output for staged changes
    for line in status_out.splitlines():
        # Lines look like:   "\tpath/to/file (Git: abc123 -> LFS: def456)"
        m = re.match(r'\s+(.+?)\s+\(', line)
        if m:
            path = m.group(1)
            # look up size from head_oids by path
            for oid, (p, s) in head_oids.items():
                if p == path and (path, s) not in pending:
                    pending.append((path, s))
                    break

    return sum(s for _, s in pending), pending


# ── main check ───────────────────────────────────────────────────────────────

def check(limit_mb: float = FREE_TIER_MB) -> int:
    """
    Print LFS storage status and return exit code.
    Returns 1 (blocks push) if this push would exceed the limit.
    """
    remote_mb, remote_files = remote_lfs_usage_mb()
    pending_mb, pending_files = pending_lfs_objects_mb()
    after_mb = remote_mb + pending_mb
    pct = after_mb / limit_mb * 100

    print("\n📊 Git LFS Storage Check")
    print(f"   Remote (already pushed) : {remote_mb:>8.1f} MB")

    if pending_files:
        print(f"   Pending (this push)     : {pending_mb:>8.1f} MB")
        for path, size in pending_files:
            flag = "  ⚠️ " if size > 200 else "    "
            print(f"   {flag}  {size:>7.1f} MB  {path}")
        print(f"   ─────────────────────────────────────")
        print(f"   After push              : {after_mb:>8.1f} MB / {limit_mb:.0f} MB  ({pct:.1f}%)")
    else:
        print(f"   ─────────────────────────────────────")
        print(f"   Total                   : {remote_mb:>8.1f} MB / {limit_mb:.0f} MB  ({pct:.1f}%)")

    if after_mb > limit_mb:
        over = after_mb - limit_mb
        print(f"\n   ❌  PUSH BLOCKED — would exceed limit by {over:.1f} MB")
        print(f"   💡  Options:")
        print(f"       • Run: python scripts/lfs_cleanup.py --dry-run")
        print(f"       • Add large source files to .gitignore instead of LFS")
        print(f"       • Upgrade GitHub LFS storage pack ($5/month per 50 GB)")
        print()
        return 1
    elif pct > 90:
        print(f"\n   ⚠️   {pct:.0f}% used — approaching limit, cleanup soon")
        print(f"   💡  Run: python scripts/lfs_cleanup.py --dry-run")
    elif pct > 75:
        print(f"\n   ⚡  {pct:.0f}% used")
    else:
        print(f"\n   ✅  Storage healthy")

    print()
    return 0


def main() -> None:
    parser = ArgumentParser(description="Check Git LFS storage before pushing")
    parser.add_argument('--limit-mb', type=float, default=FREE_TIER_MB,
                        help=f'Storage limit in MB (default: {FREE_TIER_MB})')
    # When called as a hook, git passes remote name + URL as positional args — ignore them
    parser.add_argument('remote', nargs='?', help=argparse.SUPPRESS if False else None)
    parser.add_argument('url',    nargs='?', help=argparse.SUPPRESS if False else None)
    args = parser.parse_args()
    sys.exit(check(limit_mb=args.limit_mb))


if __name__ == '__main__':
    import argparse  # noqa: F811 — re-import to satisfy the ArgumentParser alias above
    main()
