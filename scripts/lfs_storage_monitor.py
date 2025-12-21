#!/usr/bin/env python3
"""
Git LFS Storage Monitor (Pre-Push Hook)

Warns when LFS storage is approaching the 1GB limit.
Install: Copy to .git/hooks/pre-push and make executable
"""

import subprocess
import re
import sys


def get_lfs_usage() -> float:
    """Get current LFS storage usage in MB."""
    try:
        result = subprocess.run(
            ['git', 'lfs', 'ls-files', '--all', '--size'],
            capture_output=True,
            text=True,
            check=True
        )
        
        total_mb = 0
        for line in result.stdout.splitlines():
            match = re.search(r'\((\d+(?:\.\d+)?)\s*MB\)', line)
            if match:
                total_mb += float(match.group(1))
            
            match = re.search(r'\((\d+(?:\.\d+)?)\s*KB\)', line)
            if match:
                total_mb += float(match.group(1)) / 1024
        
        return total_mb
    except Exception:
        return 0


def main():
    usage_mb = get_lfs_usage()
    limit_mb = 1024
    percent_used = (usage_mb / limit_mb) * 100
    
    print(f"\n📊 Git LFS Storage Check:")
    print(f"   Used: {usage_mb:.1f} MB / {limit_mb} MB ({percent_used:.1f}%)")
    
    if usage_mb > limit_mb:
        print(f"   ❌ ERROR: Exceeds GitHub's free tier limit!")
        print(f"   💡 Run: python scripts/lfs_cleanup.py --dry-run")
        sys.exit(1)
    elif percent_used > 90:
        print(f"   ⚠️  WARNING: Approaching limit! Consider cleanup.")
        print(f"   💡 Run: python scripts/lfs_cleanup.py --dry-run")
    elif percent_used > 75:
        print(f"   ⚡ Using 75%+ of storage")
    else:
        print(f"   ✅ Storage healthy")
    
    print()


if __name__ == '__main__':
    main()
