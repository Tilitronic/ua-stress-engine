import logging
import os
import hashlib
import msgpack

CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# --- Streaming Msgpack (Packer) ---
def save_to_cache_streaming(obj, key, prefix=None):
    """Save a large dict to msgpack file using streaming (Packer)."""
    path = cache_path_for_key(key, prefix)
    logging.info(f"[CACHE] Saving cache (streaming) to {path}")
    with open(path, 'wb') as f:
        packer = msgpack.Packer(use_bin_type=True)
        for k, v in obj.items():
            f.write(packer.pack((k, v)))
    return path

def load_from_cache_streaming(key, prefix=None):
    """Load a large dict from streaming msgpack file (Packer)."""
    path = cache_path_for_key(key, prefix)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        logging.info(f"[CACHE] No cache found at {path}")
        return None
    result = {}
    try:
        with open(path, 'rb') as f:
            logging.info(f"[CACHE] Loading cache (streaming) from {path}")
            unpacker = msgpack.Unpacker(f, raw=False)
            for k, v in unpacker:
                result[k] = v
        return result
    except Exception as e:
        logging.warning(f"[CACHE] Failed to load cache from {path}: {e}")
        return None

# --- Line-delimited Msgpack ---
def save_to_cache_lines(obj, key, prefix=None):
    """Save a large dict to msgpack file, one line per (k,v) pair."""
    path = cache_path_for_key(key, prefix)
    logging.info(f"[CACHE] Saving cache (lines) to {path}")
    with open(path, 'wb') as f:
        for k, v in obj.items():
            packed = msgpack.packb((k, v), use_bin_type=True)
            f.write(packed + b'\n')
    return path

def load_from_cache_lines(key, prefix=None):
    """Load a large dict from line-delimited msgpack file."""
    path = cache_path_for_key(key, prefix)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        logging.info(f"[CACHE] No cache found at {path}")
        return None
    result = {}
    try:
        with open(path, 'rb') as f:
            logging.info(f"[CACHE] Loading cache (lines) from {path}")
            for line in f:
                if line.strip():
                    k, v = msgpack.unpackb(line.strip(), raw=False)
                    result[k] = v
        return result
    except Exception as e:
        logging.warning(f"[CACHE] Failed to load cache from {path}: {e}")
        return None


def compute_file_hash(file_path):
    """Compute SHA256 hash of a file's contents."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def compute_parser_hash(parser_path, db_path):
    """Hash parser file and DB file together for cache key."""
    parser_hash = compute_file_hash(parser_path)
    db_hash = compute_file_hash(db_path)
    return hashlib.sha256((parser_hash + db_hash).encode('utf-8')).hexdigest()

def cache_path_for_key(key, prefix=None):
    if prefix:
        filename = f"{prefix}_{key}.msgpack"
    else:
        filename = f"{key}.msgpack"
    return os.path.join(CACHE_DIR, filename)

def save_to_cache(obj, key, prefix=None):
    path = cache_path_for_key(key, prefix)
    logging.info(f"[CACHE] Saving cache (classic) to {path}")
    with open(path, 'wb') as f:
        msgpack.pack(obj, f, use_bin_type=True)
    return path

def load_from_cache(key, prefix=None):
    path = cache_path_for_key(key, prefix)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        logging.info(f"[CACHE] No cache found at {path}")
        return None
    try:
        with open(path, 'rb') as f:
            logging.info(f"[CACHE] Loading cache (classic) from {path}")
            return msgpack.unpackb(f.read(), raw=False)
    except Exception as e:
        logging.warning(f"[CACHE] Failed to load cache from {path}: {e}")
        return None

def from_serializable(data, model_cls=None):
    """
    Recursively reconstructs Pydantic models from dicts/lists loaded from cache.
    If model_cls is provided, applies it to the top-level dict values (for dict-of-models pattern).
    """
    if model_cls is not None and isinstance(data, dict):
        return {k: model_cls.model_validate(v) for k, v in data.items()}
    elif isinstance(data, dict):
        return {k: from_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [from_serializable(v) for v in data]
    else:
        return data
def to_serializable(obj):
    """Recursively convert Pydantic models and nested structures to dicts/lists for msgpack serialization."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj