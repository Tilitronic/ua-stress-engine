import os
import hashlib
import urllib.request
import urllib.error
import logging

logger = logging.getLogger(__name__)

DB_REMOTE_URL = "https://raw.githubusercontent.com/lang-uk/ukrainian-word-stress-dictionary/master/stress.txt"

def get_file_hash(file_path: str) -> str:
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
        logger.warning(f"Error calculating hash for {file_path}: {e}")
        return None

def download_file(url: str, target_path: str, description: str) -> bool:
    """Download a file from URL with progress reporting."""
    try:
        target_dir = os.path.dirname(target_path)
        os.makedirs(target_dir, exist_ok=True)
        logger.info(f"Downloading {description} from {url} to {target_path}")
        def progress_hook(block_num, block_size, total_size):
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
        logger.error(f"Download failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return False

def ensure_latest_db_file(local_path: str, url: str = DB_REMOTE_URL, description: str = "Ukrainian word stress dictionary"):
    """Check if local DB file is up-to-date with remote, download if missing or outdated."""
    logger.info("Checking local stress dictionary file...")
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path)
        file_hash = get_file_hash(local_path)
        logger.info(f"  Local file: {local_path} ({file_size / (1024*1024):.2f} MB, hash: {file_hash[:16]}...)")
        # Check remote file size
        try:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=5) as response:
                remote_size = response.headers.get('Content-Length')
                if remote_size and int(remote_size) == file_size:
                    logger.info("  ✓ Local file matches remote size. No download needed.")
                    return
                else:
                    logger.info("  ⚠ Remote file size differs. Downloading latest version...")
        except Exception as e:
            logger.warning(f"  Could not check remote file size: {e}. Proceeding with local file.")
            return
    else:
        logger.info(f"  Local file not found: {local_path}")
        logger.info("  Downloading latest version...")
    # Download or replace file
    download_file(url, local_path, description)
