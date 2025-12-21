#!/usr/bin/env python3
"""
Git LFS Storage Cleanup Script

Automatically removes old LFS versions to stay within storage limits.
Policy: Keep only the 3 most recent versions of each LFS file.

Usage:
    python scripts/lfs_cleanup.py [--dry-run] [--keep-versions N]
"""

import subprocess
import re
import sys
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser


def run_git_command(cmd: list[str]) -> str:
    """Run a git command and return output."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()


def get_all_lfs_files() -> dict[str, list[tuple[str, str, str]]]:
    """
    Get all LFS files with their OIDs and commits.
    
    Returns:
        Dict mapping file paths to list of (oid, size, commit_hash) tuples
    """
    # Get all LFS files with sizes
    output = run_git_command(['git', 'lfs', 'ls-files', '--all', '--size'])
    
    files = defaultdict(list)
    
    for line in output.splitlines():
        if not line.strip():
            continue
            
        # Parse: "oid * path (size)"
        match = re.match(r'(\w+)\s+[*-]\s+(.+?)\s+\((.+?)\)', line)
        if match:
            oid = match.group(1)
            path = match.group(2)
            size = match.group(3)
            
            # Get commit info for this OID
            try:
                commit_info = run_git_command([
                    'git', 'log', '--all', '--pretty=format:%H|%ai',
                    '-S', oid, '--', path
                ])
                if commit_info:
                    commit_hash, commit_date = commit_info.split('|', 1)
                    files[path].append((oid, size, commit_date, commit_hash))
            except subprocess.CalledProcessError:
                # If we can't find commit, assume it's the current version
                files[path].append((oid, size, '9999-12-31', 'HEAD'))
    
    # Sort by date (newest first)
    for path in files:
        files[path].sort(key=lambda x: x[2], reverse=True)
    
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


def cleanup_old_versions(keep_versions: int = 3, dry_run: bool = False):
    """
    Remove old LFS versions, keeping only N most recent versions.
    
    Args:
        keep_versions: Number of versions to keep per file (default: 3)
        dry_run: If True, only show what would be deleted
    """
    print(f"🔍 Analyzing LFS files...")
    print(f"📋 Policy: Keep {keep_versions} most recent version(s) per file\n")
    
    # Get current usage
    usage = get_current_lfs_usage()
    print(f"📊 Current LFS usage: {usage['total_mb']:.2f} MB ({usage['file_count']} objects)")
    print(f"💾 GitHub free tier limit: 1024 MB")
    print(f"✅ Available space: {1024 - usage['total_mb']:.2f} MB\n")
    
    # Get all LFS files with their versions
    lfs_files = get_all_lfs_files()
    
    if not lfs_files:
        print("✨ No LFS files found")
        return
    
    print(f"📁 Found {len(lfs_files)} unique LFS file(s):\n")
    
    total_to_remove_mb = 0
    versions_to_remove = []
    
    for path, versions in lfs_files.items():
        print(f"  {path}")
        print(f"    Total versions: {len(versions)}")
        
        if len(versions) <= keep_versions:
            print(f"    ✓ Keeping all {len(versions)} version(s) (within policy)\n")
            continue
        
        # Keep the N most recent, mark rest for deletion
        to_keep = versions[:keep_versions]
        to_remove = versions[keep_versions:]
        
        print(f"    ✓ Keeping {len(to_keep)} version(s):")
        for oid, size, date, commit in to_keep:
            print(f"      - {oid[:10]}... ({size}) from {date[:10]}")
        
        print(f"    ✗ Removing {len(to_remove)} old version(s):")
        for oid, size, date, commit in to_remove:
            print(f"      - {oid[:10]}... ({size}) from {date[:10]}")
            versions_to_remove.append((path, oid, size, commit))
            
            # Calculate size to remove
            match = re.match(r'(\d+(?:\.\d+)?)\s*MB', size)
            if match:
                total_to_remove_mb += float(match.group(1))
        
        print()
    
    if not versions_to_remove:
        print("✨ No old versions to remove. Storage policy already satisfied!")
        return
    
    print(f"📊 Summary:")
    print(f"  - Versions to remove: {len(versions_to_remove)}")
    print(f"  - Space to free: ~{total_to_remove_mb:.2f} MB")
    print(f"  - New usage: ~{usage['total_mb'] - total_to_remove_mb:.2f} MB")
    print()
    
    if dry_run:
        print("🔍 DRY RUN - No changes made")
        print("\n💡 To actually remove old versions, you would need to:")
        print("   1. Rewrite git history to remove commits with old versions")
        print("   2. Force push to remote")
        print("   3. Run: git lfs prune --verify-remote")
        print("\n⚠️  This is a destructive operation requiring careful consideration.")
        return
    
    print("⚠️  WARNING: This requires rewriting git history!")
    print("   - All team members will need to re-clone")
    print("   - Old commit hashes will change")
    print("   - This cannot be easily undone")
    print()
    
    response = input("Continue? (type 'yes' to proceed): ")
    if response.lower() != 'yes':
        print("❌ Cancelled")
        return
    
    print("\n🚀 Not implemented yet - requires careful git history rewriting")
    print("   For now, manually identify old commits and use:")
    print("   git filter-branch or git filter-repo to remove them")


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
        print(f"❌ Git command failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
