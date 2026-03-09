#!/usr/bin/env python3
"""
Git LFS Storage Cleanup Script

Automatically removes old LFS versions to stay within storage limits.
Policy: Keep only the N most recent versions of each LFS file.

Requires git-filter-repo (pip install git-filter-repo).

Usage:
    python scripts/lfs_cleanup.py [--dry-run] [--keep-versions N]
"""

import subprocess
import re
import sys
import shutil
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser


def run_git_command(cmd: list[str], check: bool = True) -> str:
    """Run a git command and return output."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check
    )
    return result.stdout.strip()


def get_all_lfs_files() -> dict[str, list[tuple[str, str, str]]]:
    """
    Get all LFS files with their OIDs and commits.

    Returns a dict:  path -> list of (oid, size, commit_date, commit_hash)
    sorted newest-first.  The entry for the *checked-out* (HEAD) blob is
    placed first regardless of date so it is always treated as "keep".
    """
    output = run_git_command(['git', 'lfs', 'ls-files', '--all', '--size'])

    files: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    # Track which OIDs are present locally (marked with '*')
    present_oids: set[str] = set()

    for line in output.splitlines():
        if not line.strip():
            continue
        # "oid [*|-] path (size)"
        m = re.match(r'(\w+)\s+([*-])\s+(.+?)\s+\((.+?)\)', line)
        if not m:
            continue
        oid, marker, path, size = m.group(1), m.group(2), m.group(3), m.group(4)

        if marker == '*':
            present_oids.add(oid)

        # Get the most-recent commit that introduced this OID for this path
        try:
            commit_info = run_git_command([
                'git', 'log', '--all', '--pretty=format:%H|%ai',
                '-S', oid, '--', path
            ])
            if commit_info:
                # -S can return multiple commits (added + removed); take newest
                first_line = commit_info.splitlines()[0]
                commit_hash, commit_date = first_line.split('|', 1)
                files[path].append((oid, size, commit_date, commit_hash))
        except subprocess.CalledProcessError:
            files[path].append((oid, size, '9999-12-31', 'HEAD'))

    # Sort: present (checked-out) blobs first, then by date descending
    for path in files:
        files[path].sort(
            key=lambda x: (0 if x[0] in present_oids else 1, x[2]),
            reverse=False,
        )
        # Within present blobs keep newest-date first
        present = [(o, s, d, c) for o, s, d, c in files[path] if o in present_oids]
        absent  = [(o, s, d, c) for o, s, d, c in files[path] if o not in present_oids]
        present.sort(key=lambda x: x[2], reverse=True)
        absent.sort(key=lambda x: x[2], reverse=True)
        files[path] = present + absent

    return files


def get_current_lfs_usage() -> dict[str, int]:
    """Get current LFS storage usage in MB."""
    output = run_git_command(['git', 'lfs', 'ls-files', '--all', '--size'])
    
    total_mb = 0
    file_count = 0
    
    for line in output.splitlines():
        if not line.strip():
            continue
        
        match = re.search(r'\((\d+(?:\.\d+)?)\s*MB\)', line)
        if match:
            total_mb += float(match.group(1))
            file_count += 1
        
        match = re.search(r'\((\d+(?:\.\d+)?)\s*KB\)', line)
        if match:
            total_mb += float(match.group(1)) / 1024
            file_count += 1
    
    return {'total_mb': total_mb, 'file_count': file_count}


def check_filter_repo_available() -> bool:
    """Check if git filter-repo is available."""
    result = subprocess.run(
        ['git', 'filter-repo', '--version'],
        capture_output=True, text=True
    )
    return result.returncode == 0


def get_affected_paths(versions_to_remove: list[tuple]) -> list[str]:
    """Return unique file paths that have old versions to remove."""
    seen = set()
    paths = []
    for path, _oid, _size, _commit in versions_to_remove:
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def cleanup_old_versions(keep_versions: int = 3, dry_run: bool = False):
    """
    Remove old LFS versions, keeping only N most recent versions.

    Strategy:
      1. Identify files that have more than `keep_versions` LFS objects.
      2. For each such file, use `git filter-repo --path <file> --force`
         to keep the file in history but collapse its old blobs:
         we rewrite history so every commit that previously referenced an
         old (to-be-removed) blob instead carries the *newest* blob for
         that file.  We achieve this via `--blob-callback` in filter-repo.
      3. Force-push the rewritten history.
      4. Run `git lfs prune` so the old local + remote objects are garbage
         collected.

    Args:
        keep_versions: Number of versions to keep per file (default: 3)
        dry_run: If True, only show what would be deleted
    """
    print("[*] Analyzing LFS files...")
    print(f"[*] Policy: keep {keep_versions} most recent version(s) per file\n")

    # ------------------------------------------------------------------
    # Guard: git filter-repo must be installed
    # ------------------------------------------------------------------
    if not check_filter_repo_available():
        print("[ERROR] git filter-repo is not installed.")
        print("        Install it with: pip install git-filter-repo")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Current usage
    # ------------------------------------------------------------------
    usage = get_current_lfs_usage()
    print(f"[LFS] Current local LFS objects : {usage['total_mb']:.1f} MB  ({usage['file_count']} objects)")
    print(f"[LFS] GitHub free tier limit    : 1024 MB")
    print(f"[LFS] Headroom                  : {1024 - usage['total_mb']:.1f} MB\n")

    # ------------------------------------------------------------------
    # Collect all LFS file versions
    # ------------------------------------------------------------------
    lfs_files = get_all_lfs_files()

    if not lfs_files:
        print("[OK] No LFS files found.")
        return

    print(f"[*] Found {len(lfs_files)} unique LFS file(s):\n")

    total_to_remove_mb = 0.0
    versions_to_remove: list[tuple[str, str, str, str]] = []  # (path, oid, size, commit)

    for path, versions in lfs_files.items():
        print(f"  {path}")
        print(f"    versions tracked: {len(versions)}")

        if len(versions) <= keep_versions:
            print(f"    [OK] all {len(versions)} version(s) within policy\n")
            continue

        to_keep   = versions[:keep_versions]
        to_remove = versions[keep_versions:]

        print(f"    [KEEP]   {len(to_keep)} version(s):")
        for oid, size, date, commit in to_keep:
            print(f"      + {oid[:12]}  ({size})  {date[:10]}  {commit[:8]}")

        print(f"    [REMOVE] {len(to_remove)} old version(s):")
        for oid, size, date, commit in to_remove:
            print(f"      - {oid[:12]}  ({size})  {date[:10]}  {commit[:8]}")
            versions_to_remove.append((path, oid, size, commit))

            m = re.match(r'(\d+(?:\.\d+)?)\s*MB', size)
            if m:
                total_to_remove_mb += float(m.group(1))
        print()

    if not versions_to_remove:
        print("[OK] No old versions to remove — storage policy already satisfied!")
        return

    estimated_after = usage['total_mb'] - total_to_remove_mb
    print("[*] Summary")
    print(f"    Versions to remove : {len(versions_to_remove)}")
    print(f"    Space to free      : ~{total_to_remove_mb:.1f} MB")
    print(f"    Estimated after    : ~{estimated_after:.1f} MB")
    print()

    # ------------------------------------------------------------------
    # Dry-run exit
    # ------------------------------------------------------------------
    if dry_run:
        print("[DRY-RUN] No changes made.")
        print()
        print("  Steps that would run:")
        for path in get_affected_paths(versions_to_remove):
            print(f"    git filter-repo --path \"{path}\" --force  (rewrites history)")
        print("    git push --force origin HEAD")
        print("    git lfs prune --verify-remote")
        return

    # ------------------------------------------------------------------
    # Confirmation
    # ------------------------------------------------------------------
    print("[!] WARNING: This rewrites git history.")
    print("    - Commit SHAs will change for every affected commit.")
    print("    - Anyone else with a clone must re-clone or hard-reset.")
    print("    - A force-push to origin is required.")
    print()
    response = input("    Type 'yes' to proceed: ").strip().lower()
    if response != 'yes':
        print("[CANCELLED]")
        return

    # ------------------------------------------------------------------
    # Build the set of (path -> newest_oid) so we can replace old blobs
    # ------------------------------------------------------------------
    # newest_oid_for[path] = the OID we want every old commit to point at
    newest_oid_for: dict[str, str] = {}
    for path, versions in lfs_files.items():
        if len(versions) > keep_versions:
            newest_oid_for[path] = versions[0][0]  # already sorted newest-first

    affected_paths = get_affected_paths(versions_to_remove)

    # ------------------------------------------------------------------
    # Step 1 — rewrite history with git filter-repo
    #
    # We use a --blob-callback Python snippet that replaces the LFS pointer
    # content of old blobs with the newest pointer for that file.
    # Because filter-repo works on blob content (not tree entries), we need
    # to match by the oid embedded in the LFS pointer text.
    # ------------------------------------------------------------------
    print("\n[1/4] Building blob-replacement map ...")

    # old_oid -> new_oid  (12-char prefixes suffice for the pointer text,
    # but we use full OIDs for correctness)
    old_to_new: dict[str, str] = {}
    for path, oid, size, commit in versions_to_remove:
        if path in newest_oid_for:
            old_to_new[oid] = newest_oid_for[path]

    if not old_to_new:
        print("[!] Nothing to map — skipping history rewrite.")
    else:
        # Write a small helper script that filter-repo will exec per blob
        callback_lines = [
            "import re",
            "old_to_new = {",
        ]
        for old, new in old_to_new.items():
            callback_lines.append(f"    {old!r}: {new!r},")
        callback_lines += [
            "}",
            "def process_blob(blob):",
            "    try:",
            "        text = blob.data.decode('utf-8')",
            "    except Exception:",
            "        return",
            "    m = re.search(r'oid sha256:([0-9a-f]{64})', text)",
            "    if not m:",
            "        return",
            "    found_oid = m.group(1)",
            "    if found_oid in old_to_new:",
            "        new_oid = old_to_new[found_oid]",
            "        blob.data = re.sub(",
            "            b'oid sha256:[0-9a-f]{64}',",
            "            ('oid sha256:' + new_oid).encode(),",
            "            blob.data,",
            "        )",
        ]
        callback_script = "\n".join(callback_lines)

        # Temp file so we don't have to worry about shell quoting
        callback_path = Path(".git") / "_lfs_cleanup_callback.py"
        callback_path.write_text(callback_script, encoding="utf-8")

        print("[2/4] Rewriting history (git filter-repo) ...")
        try:
            result = subprocess.run(
                [
                    'git', 'filter-repo',
                    '--blob-callback', f'exec(open("{callback_path.as_posix()}").read()); process_blob(blob)',
                    '--force',
                ],
                capture_output=False,   # let output flow to console
                text=True,
            )
            if result.returncode != 0:
                print(f"[ERROR] git filter-repo exited with code {result.returncode}")
                sys.exit(result.returncode)
        finally:
            callback_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Step 2 — restore the remote origin (filter-repo removes it as a
    #          safety measure when it rewrites history)
    # ------------------------------------------------------------------
    print("\n[3/4] Checking remote ...")
    remotes_out = run_git_command(['git', 'remote', '-v'], check=False)
    if 'origin' not in remotes_out:
        # filter-repo wiped it; try to restore from ORIG_HEAD context
        print("[!] Remote 'origin' was removed by filter-repo.")
        origin_url = input("    Enter the GitHub remote URL to re-add: ").strip()
        if origin_url:
            subprocess.run(['git', 'remote', 'add', 'origin', origin_url], check=True)
            print(f"    [OK] origin -> {origin_url}")
        else:
            print("[!] No URL provided. You must manually run:")
            print("      git remote add origin <url>")
            print("      git push --force origin HEAD")
            print("      git lfs prune --verify-remote")
            return

    # ------------------------------------------------------------------
    # Step 3 — force push
    # ------------------------------------------------------------------
    print("\n[4/4] Force-pushing rewritten history ...")
    # Determine current branch name
    branch = run_git_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], check=False) or 'main'
    push_result = subprocess.run(
        ['git', 'push', '--force', 'origin', branch],
        capture_output=False, text=True
    )
    if push_result.returncode != 0:
        print(f"[ERROR] Force-push failed (exit {push_result.returncode}).")
        print("        The history has been rewritten locally.")
        print("        Fix the remote issue, then run manually:")
        print(f"          git push --force origin {branch}")
        print("          git lfs prune --verify-remote")
        sys.exit(push_result.returncode)

    # ------------------------------------------------------------------
    # Step 4 — prune old LFS objects
    # ------------------------------------------------------------------
    print("\n[*] Pruning old LFS objects ...")
    subprocess.run(['git', 'lfs', 'prune', '--verify-remote'], capture_output=False)

    print()
    print("[OK] Cleanup complete!")
    print(f"     Freed approximately {total_to_remove_mb:.1f} MB of LFS storage.")


def main():
    parser = ArgumentParser(description="Clean up old Git LFS versions")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be removed without making changes'
    )
    parser.add_argument(
        '--keep-versions',
        type=int,
        default=3,
        help='Number of versions to keep per file (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Change to repo root
    repo_root = Path(__file__).parent.parent
    import os
    os.chdir(repo_root)
    
    try:
        cleanup_old_versions(
            keep_versions=args.keep_versions,
            dry_run=args.dry_run
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Git command failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
