from pathlib import Path

import lmdb
import msgpack
from typing import Dict, Any, Optional, List
import logging
import shutil, tempfile

class LMDBExportConfig:
    def __init__(self, db_path, map_size: Optional[int] = None, overwrite: bool = False):
        # Accept str or Path, always store as Path
        self.db_path = Path(db_path)
        self.map_size = map_size
        self.overwrite = overwrite

class LMDBExporter:
    def __init__(self, config: LMDBExportConfig, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger("LMDBExporter")

    def estimate_map_size(self, data: Dict[str, Any], sample_size: int = 1000) -> int:
        total_size = 0
        count = 0
        for k, v in data.items():
            if count >= sample_size:
                break
            total_size += len(k.encode('utf-8'))
            total_size += len(msgpack.packb(v, use_bin_type=True))
            count += 1
        avg_size = total_size // max(count, 1)
        estimated = int(avg_size * len(data) * 1.25)
        return max(estimated, 100 * 1024 * 1024)  # At least 100MB

    def export_streaming(self, data_iter, total=None) -> None:
        """
        Export entries to LMDB in a streaming/chunked fashion with manual transaction batching.
        data_iter: iterable of (key, value)
        total: total number of entries (for progress bar)
        """
        from tqdm import tqdm
        lmdb_data_file = self.config.db_path / "data.mdb"
        if self.config.overwrite and lmdb_data_file.exists():
            self.logger.info(f"Removing existing LMDB file at {lmdb_data_file}")
            lmdb_data_file.unlink()
        self.config.db_path.mkdir(parents=True, exist_ok=True)
        if self.config.map_size is None:
            self.config.map_size = 10 * 1024 * 1024 * 1024  # 10GB default, adjust as needed
        env = lmdb.open(str(self.config.db_path), map_size=self.config.map_size, writemap=True, map_async=True, sync=True, metasync=True)
        count = 0
        batch_size = 10000
        txn = env.begin(write=True)
        try:
            for key, value in tqdm(data_iter, total=total, desc="LMDB Export", ncols=80):
                packed = msgpack.packb(value, use_bin_type=True)
                txn.put(key.encode('utf-8'), packed)
                count += 1
                if count % batch_size == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            txn.commit()
        finally:
            env.close()
        self.logger.info(f"Stream-exported {count} entries to LMDB at {self.config.db_path}")
        print(f"[LMDB] Stream-exported {count} entries to LMDB at {self.config.db_path}")
        # Compaction step (unchanged)
        import os
        self.logger.info("[LMDB] Starting compaction to minimal size...")
        print(f"[LMDB] Starting compaction to minimal size at {self.config.db_path} ...")
        compact_dir = Path(tempfile.mkdtemp(prefix="lmdb_compact_"))
        env = lmdb.open(str(self.config.db_path), readonly=True)
        before_size = 0
        for fname in ["data.mdb", "lock.mdb"]:
            f = self.config.db_path / fname
            if f.exists():
                before_size += f.stat().st_size
        self.logger.info(f"[LMDB] Before compaction: data.mdb + lock.mdb size = {before_size/1024/1024:.2f} MB")
        print(f"[LMDB] Before compaction: data.mdb + lock.mdb size = {before_size/1024/1024:.2f} MB")
        env.copy(str(compact_dir), compact=True)
        env.close()
        for fname in ["data.mdb", "lock.mdb"]:
            orig = self.config.db_path / fname
            compacted = compact_dir / fname
            if compacted.exists():
                if orig.exists():
                    orig.unlink()
                shutil.move(str(compacted), str(orig))
        after_size = 0
        for fname in ["data.mdb", "lock.mdb"]:
            f = self.config.db_path / fname
            if f.exists():
                after_size += f.stat().st_size
        shutil.rmtree(compact_dir)
        self.logger.info(f"[LMDB] Compaction complete. LMDB at {self.config.db_path} is now minimal size.")
        self.logger.info(f"[LMDB] After compaction: data.mdb + lock.mdb size = {after_size/1024/1024:.2f} MB (saved {(before_size-after_size)/1024/1024:.2f} MB)")
        self.logger.info(f"[LMDB] LMDB files present: {[f.name for f in self.config.db_path.glob('*')]}")
        print(f"[LMDB] Compaction complete. LMDB at {self.config.db_path} is now minimal size.")
        print(f"[LMDB] After compaction: data.mdb + lock.mdb size = {after_size/1024/1024:.2f} MB (saved {(before_size-after_size)/1024/1024:.2f} MB)")
        print(f"[LMDB] LMDB files present: {[f.name for f in self.config.db_path.glob('*')]}")

    def verify(self, sample_words: List[str]) -> Dict[str, Any]:
        env = lmdb.open(str(self.config.db_path), readonly=True, lock=False)
        found = 0
        with env.begin() as txn:
            for word in sample_words:
                if txn.get(word.encode('utf-8')):
                    found += 1
            stats = txn.stat()
        env.close()
        return {"entries": stats["entries"], "sample_found": f"{found}/{len(sample_words)}"}
import os
import hashlib
from src.data_management.transform.cache_utils import cache_path_for_key, save_to_cache_streaming, load_from_cache_streaming, to_serializable
from src.data_management.transform.parsing_merging_service import SOURCES_CONFIGS, compute_parser_hash, merge_linguistic_dicts

def compute_merged_cache_key(cache_paths):
    h = hashlib.sha256()
    for path in cache_paths:
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
    return h.hexdigest()

def merge_caches_and_save(names, merged_prefix="MERGED"):
    import logging
    from tqdm import tqdm
    import time
    cache_keys = [compute_parser_hash(SOURCES_CONFIGS[name]["parser_path"], SOURCES_CONFIGS[name]["db_path"]) for name in names]
    cache_paths = [cache_path_for_key(key, prefix=name) for key, name in zip(cache_keys, names)]
    merged_cache_key = compute_merged_cache_key(cache_paths)
    merged_cache_path = cache_path_for_key(merged_cache_key, prefix=merged_prefix)
    # Use the same cache folder as msgpack caches
    cache_folder = os.path.join(os.path.dirname(__file__), "cache")
    lmdb_dir = Path(os.path.join(cache_folder, f"{merged_prefix}_{merged_cache_key}_lmdb"))

    logger = logging.getLogger("merger")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    start_time = time.time()
    if lmdb_dir.exists():
        logger.info(f"[LMDB] Using merged LMDB cache at {lmdb_dir}")
        print(f"[LMDB] Using merged LMDB cache at {lmdb_dir}")
        cache_used = True
        merged = None  # Not loaded into memory
    else:
        logger.info(f"[LMDB] No merged LMDB cache found at {lmdb_dir}. Will merge and export to LMDB.")
        print(f"[LMDB] No merged LMDB cache found at {lmdb_dir}. Will merge and export to LMDB.")
        dicts = []
        for name, cache_path in zip(names, cache_paths):
            if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
                logger.info(f"[CACHE] Using cache for {name} from {cache_path}")
                print(f"[CACHE] Using cache for {name} from {cache_path}")
                d = load_from_cache_streaming(cache_keys[names.index(name)], prefix=name)
            else:
                logger.info(f"[CACHE] No cache for {name} at {cache_path}. Will parse {name} from scratch.")
                print(f"[CACHE] No cache for {name} at {cache_path}. Will parse {name} from scratch.")
                d = None
            dicts.append(d)
        logger.info(f"[MERGE] Merging {len(dicts)} dictionaries...")
        print(f"[MERGE] Merging {len(dicts)} dictionaries...")
        start = time.time()
        merged = merge_linguistic_dicts(dicts)
        elapsed = time.time() - start
        logger.info(f"[MERGE] Merged in {elapsed:.2f} seconds. Exporting merged cache to LMDB...")
        print(f"[MERGE] Merged in {elapsed:.2f} seconds. Exporting merged cache to LMDB...")
        export_config = LMDBExportConfig(db_path=lmdb_dir, overwrite=True)
        exporter = LMDBExporter(export_config, logger=logger)
        print(f"[LMDB] About to export merged data to {lmdb_dir}")
        exporter.export(merged)
        print(f"[LMDB] Merged cache exported to {lmdb_dir}")
        logger.info(f"[LMDB] Merged cache exported to {lmdb_dir}")
        # Ensure LMDB directory exists after export
        if not lmdb_dir.exists():
            raise RuntimeError(f"[CRITICAL] LMDB export failed: {lmdb_dir} was not created!")
        cache_used = False
    elapsed = time.time() - start_time
    logger.info(f"Merging complete. Unique lemmas: {len(merged) if merged is not None else 'N/A (LMDB only)'}")
    logger.info(f"Total merging time: {elapsed:.2f} seconds")
    print("\n=== Merging Statistics ===")
    print(f"Merged LMDB cache used: {cache_used}")
    print(f"Unique lemmas: {len(merged) if merged is not None else 'N/A (LMDB only)'}")
    print(f"Merged LMDB cache path: {lmdb_dir}")
    print(f"Total merging time: {elapsed:.2f} seconds\n")
    return merged, str(lmdb_dir)
