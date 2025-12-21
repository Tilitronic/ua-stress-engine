#!/usr/bin/env python3
"""
LMDB exporter for Ukrainian stress dictionary.
Exports parsed dictionary to LMDB format for ultra-fast read-only queries.

Optimized for read-only mode with minimal file size and maximum performance.
Uses MsgPack serialization and LMDB append mode for optimal speed.
"""

from pathlib import Path
from logging import getLogger
from typing import List, Dict, Optional, Callable
import msgpack
import lmdb
import sys

logger = getLogger(__name__)


class LMDBExporter:
    """
    Export dictionary to LMDB format optimized for read-only access.
    
    LMDB (Lightning Memory-Mapped Database) provides:
    - Ultra-fast reads (millions/sec)
    - Memory-mapped access (zero-copy)
    - Minimal disk space (map_size matched to data)
    - Perfect for read-only dictionary lookups
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize LMDB exporter.
        
        Args:
            db_path: Path to LMDB database directory
        """
        self.db_path = Path(db_path)
    
    def _estimate_data_size(self, data: Dict[str, List[Dict]]) -> int:
        """
        Estimate the actual data size needed for LMDB.
        
        Args:
            data: Dictionary to export
            
        Returns:
            Estimated size in bytes with safety margin
        """
        # Sample ~1000 entries to estimate average size
        sample_size = 0
        sample_count = 0
        max_samples = 1000
        
        for key, forms in data.items():
            if sample_count >= max_samples:
                break
            
            # Key size (UTF-8 encoded)
            sample_size += len(key.encode('utf-8'))
            
            # Value size (MsgPack serialized - more compact than JSON)
            value = msgpack.packb(forms, use_bin_type=True)
            sample_size += len(value)
            
            sample_count += 1
        
        # Calculate average and extrapolate to total
        if sample_count > 0:
            avg_entry_size = sample_size / sample_count
            estimated_data = int(avg_entry_size * len(data))
        else:
            # Fallback estimate if no data
            estimated_data = 100 * 1024 * 1024  # 100MB
        
        # Add overhead for LMDB internal structures
        # LMDB uses B+ trees: pages, branches, leaves
        # MsgPack is more predictable than JSON, so less margin needed
        overhead_factor = 1.15  # Reduced from 1.30 due to MsgPack efficiency
        
        # Add safety margin for growth
        safety_margin = 1.10  # Reduced from 1.20 due to better size prediction
        
        final_size = int(estimated_data * overhead_factor * safety_margin)
        
        logger.info(f"Estimated data size: {estimated_data / (1024*1024):.2f} MB")
        logger.info(f"Map size with overhead: {final_size / (1024*1024):.2f} MB")
        
        return final_size
    
    def export_raw(self, data: Dict[str, List[Dict]], progress_callback: Optional[Callable] = None) -> None:
        """
        Export raw dictionary data to LMDB with optimized map_size.
        Deletes existing database if present and creates a fresh one.
        
        Uses MsgPack serialization and MDB_APPEND for maximum performance.
        
        Args:
            data: Dict mapping keys to list of form dicts
            progress_callback: Optional callback(current, total) for progress tracking
        """
        logger.info(f"Exporting {len(data):,} words to LMDB at {self.db_path}")
        
        # Calculate optimal map_size based on actual data
        map_size = self._estimate_data_size(data)
        
        # Delete existing database if it exists
        if self.db_path.exists():
            import shutil
            logger.info(f"Removing existing database at {self.db_path}")
            shutil.rmtree(self.db_path)
        
        # Create directory
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Sort keys alphabetically for MDB_APPEND optimization
        # This allows LMDB to write sequentially without tree rebalancing
        sorted_keys = sorted(data.keys())
        logger.info("Keys sorted alphabetically for append mode optimization")
        
        # Open LMDB environment with maximum performance settings
        env = lmdb.open(
            str(self.db_path),
            map_size=map_size,
            max_dbs=0,
            readonly=False,
            writemap=True,    # Write directly to memory map (faster)
            map_async=True,   # Let OS handle flushing (maximizes throughput)
            sync=True,        # Ensure final sync on close
            metasync=True     # Ensure metadata is synced
        )
        
        try:
            with env.begin(write=True) as txn:
                total = len(sorted_keys)
                
                for idx, word in enumerate(sorted_keys, 1):
                    forms = data[word]
                    
                    # Serialize to MsgPack (30% smaller than JSON, faster CPU)
                    value = msgpack.packb(forms, use_bin_type=True)
                    key = word.encode('utf-8')
                    
                    # Use append=True since keys are sorted (10x-50x faster writes)
                    txn.put(key, value, append=True)
                    
                    # Progress callback
                    if progress_callback and (idx % 10000 == 0 or idx == total):
                        progress_callback(idx, total)
            
            # Get actual database stats
            with env.begin() as txn:
                stats = txn.stat()
                actual_pages = stats['leaf_pages'] + stats['branch_pages']
                actual_size = stats['psize'] * actual_pages
                
                logger.info(f"✓ Export complete: {stats['entries']:,} entries")
                logger.info(f"  Actual data: {actual_size / (1024*1024):.2f} MB")
                logger.info(f"  Efficiency: {(actual_size / map_size * 100):.1f}% of allocated space")
        
        finally:
            env.close()
    
    def verify(self, sample_words: List[str]) -> Dict:
        """
        Verify exported database with sample lookups.
        Opens in read-only mode as it would be used in production.
        
        Args:
            sample_words: Words to test
        
        Returns:
            Dict with verification results
        """
        try:
            # Open in read-only mode (as it will be used)
            env = lmdb.open(
                str(self.db_path),
                readonly=True,
                lock=False  # No lock file needed for read-only
            )
            
            try:
                with env.begin() as txn:
                    stats = txn.stat()
                    
                    # Test sample lookups
                    found = 0
                    for word in sample_words:
                        key = word.encode('utf-8')
                        value = txn.get(key)
                        if value:
                            found += 1
                    
                    # Get actual data size
                    actual_pages = stats['leaf_pages'] + stats['branch_pages']
                    actual_size = stats['psize'] * actual_pages
                    
                    return {
                        "status": "success",
                        "entries": stats['entries'],
                        "size_bytes": actual_size,
                        "sample_found": f"{found}/{len(sample_words)}"
                    }
            finally:
                env.close()
        except Exception as e:
            return {"status": "error", "message": str(e)}


class LMDBQuery:
    """
    Query LMDB stress dictionary.
    
    Usage:
        db = LMDBQuery("data/stress.lmdb")
        forms = db.lookup("атлас")
        db.close()
    
    Or with context manager:
        with LMDBQuery("data/stress.lmdb") as db:
            forms = db.lookup("атлас")
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize LMDB query interface.
        
        Args:
            db_path: Path to LMDB database directory
        """
        self.db_path = Path(db_path)
        self.env = None
        self._open()
    
    def _open(self):
        """Open LMDB environment with read-optimized settings"""
        if not self.db_path.exists():
            raise FileNotFoundError(f"LMDB database not found at {self.db_path}")
        
        self.env = lmdb.open(
            str(self.db_path),
            readonly=True,
            lock=False,      # Multiple readers without locking
            readahead=True,  # Enable OS read-ahead for sequential scans
            max_dbs=0
        )
        logger.debug(f"Opened LMDB database at {self.db_path}")
    
    def lookup(self, word: str) -> Optional[List[Dict]]:
        """
        Look up word in database using zero-copy memory-mapped access.
        
        Args:
            word: Word to look up (normalized: lowercase, normalized apostrophe)
        
        Returns:
            List of form dictionaries, or None if not found
        """
        if not self.env:
            raise RuntimeError("Database not open")
        
        key = word.encode('utf-8')
        
        with self.env.begin(buffers=True) as txn:
            # Use buffers=True for zero-copy access (reads directly from mmap)
            value = txn.get(key)
            
            if value is None:
                return None
            
            # Deserialize MsgPack directly to Python object
            forms_data = msgpack.unpackb(value, raw=False)
            return forms_data
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        if not self.env:
            raise RuntimeError("Database not open")
        
        with self.env.begin() as txn:
            stats = txn.stat()
            return {
                'entries': stats['entries'],
                'page_size': stats['psize'],
                'depth': stats['depth'],
                'size_bytes': stats['psize'] * stats['depth']
            }
    
    def list_words(self, limit: Optional[int] = None) -> List[str]:
        """
        List all words in database.
        
        Args:
            limit: Maximum number of words to return (None = all)
        
        Returns:
            List of word strings
        """
        if not self.env:
            raise RuntimeError("Database not open")
        
        words = []
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for i, (key, _) in enumerate(cursor):
                if limit and i >= limit:
                    break
                words.append(key.decode('utf-8'))
        
        return words
    
    def prefix_search(self, prefix: str, limit: Optional[int] = None) -> List[str]:
        """
        Find words starting with prefix.
        
        Args:
            prefix: Prefix to search for
            limit: Maximum results to return
        
        Returns:
            List of matching words
        """
        if not self.env:
            raise RuntimeError("Database not open")
        
        prefix_bytes = prefix.encode('utf-8')
        matches = []
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            
            # Position cursor at first key >= prefix
            if cursor.set_range(prefix_bytes):
                for key, _ in cursor:
                    key_str = key.decode('utf-8')
                    
                    # Check if still matches prefix
                    if not key_str.startswith(prefix):
                        break
                    
                    matches.append(key_str)
                    
                    if limit and len(matches) >= limit:
                        break
        
        return matches
    
    def close(self):
        """Close database connection"""
        if self.env:
            self.env.close()
            self.env = None
            logger.debug("Closed LMDB database")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        self.close()
