#!/usr/bin/env python3
"""
Download large data files from remote repositories to raw_data folder.
This script manages the ua_word_stress_dictionary.txt and stress.trie files
which are too large to store in Git.

Run this before using the stress parsing module:
    python download_data_files.py
"""

import os
import sys
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


# File configurations: (target_path, url, description)
DATA_FILES = [
    (
        "raw_data/stress.trie",
        "https://raw.githubusercontent.com/lang-uk/ukrainian-word-stress/master/ukrainian_word_stress/data/stress.trie",
        "Stress trie database",
    ),
    (
        "raw_data/ua_word_stress_dictionary.txt",
        "https://raw.githubusercontent.com/lang-uk/ukrainian-word-stress-dictionary/master/stress.txt",
        "Ukrainian word stress dictionary",
    ),
]


def get_file_hash(file_path: str) -> Optional[str]:
    """Calculate SHA256 hash of a file."""
    if not os.path.exists(file_path):
        return None
    
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None


def download_file(url: str, target_path: str, description: str) -> bool:
    """Download a file from URL with progress reporting."""
    try:
        # Create directory if it doesn't exist
        target_dir = os.path.dirname(target_path)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"\n📥 Downloading {description}...")
        print(f"   From: {url}")
        print(f"   To:   {target_path}")
        
        def progress_hook(block_num, block_size, total_size):
            """Display download progress."""
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, int(100.0 * downloaded / total_size))
                bar_length = 40
                filled = int(bar_length * percent // 100)
                bar = "█" * filled + "░" * (bar_length - filled)
                print(f"\r   [{bar}] {percent}% ({downloaded}/{total_size} bytes)", end="")
        
        urllib.request.urlretrieve(url, target_path, progress_hook)
        print("\n   ✓ Download complete!")
        return True
        
    except urllib.error.URLError as e:
        print(f"   ✗ Download failed: {e}")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def check_and_download_files() -> bool:
    """Check if files exist and download if needed."""
    all_success = True
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 70)
    print("Ukrainian Word Stress Data Files Manager")
    print("=" * 70)
    
    for target_path, url, description in DATA_FILES:
        full_path = os.path.join(root_dir, target_path)
        
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            file_hash = get_file_hash(full_path)
            print(f"\n✓ {description}")
            print(f"  Path: {target_path}")
            print(f"  Size: {file_size / (1024*1024):.2f} MB")
            if file_hash:
                print(f"  Hash: {file_hash[:16]}...")
        else:
            print(f"\n⚠ {description} not found")
            print(f"  Downloading to: {target_path}")
            if not download_file(url, full_path, description):
                all_success = False
    
    print("\n" + "=" * 70)
    if all_success:
        print("✓ All data files ready!")
    else:
        print("✗ Some files failed to download. Please check your internet connection.")
    print("=" * 70)
    
    return all_success


if __name__ == "__main__":
    success = check_and_download_files()
    sys.exit(0 if success else 1)
