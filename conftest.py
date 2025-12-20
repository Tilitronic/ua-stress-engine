"""
Pytest configuration file for proper Unicode/UTF-8 handling
"""

import sys
import io
import os

# Force UTF-8 encoding globally
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure stdout and stderr for UTF-8
if sys.platform == 'win32':
    # Windows-specific UTF-8 handling
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    
    # Fallback for older Python versions
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configure pytest's capture to use UTF-8
import pytest

def pytest_configure(config):
    """Configure pytest with UTF-8 encoding"""
    # This runs before all tests
    if hasattr(config, '_ensure_manager'):
        try:
            # Force UTF-8 in pytest's output
            config.option.verbose = True
        except Exception:
            pass
