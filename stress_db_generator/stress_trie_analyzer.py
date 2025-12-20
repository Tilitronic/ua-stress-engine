#!/usr/bin/env python3
"""
Advanced StressTrieData analyzer for Ukrainian word stress trie.
Uses marisa_trie library to properly parse the binary trie format.
Extracts word stress patterns and morphological data.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple
from pathlib import Path
import re

try:
    import marisa_trie
except ImportError:
    print("Error: marisa-trie library is required")
    print("Install with: pip install marisa-trie")
    exit(1)


@dataclass
class StressTrieData:
    """
    Advanced dataclass for Ukrainian stress trie analysis using marisa_trie.
    
    The stress.trie is a marisa_trie.BytesTrie containing:
    - Keys: Ukrainian words (base forms) without stress
    - Values: Encoded stress positions and morphological tags
    
    Value format:
    - Single byte value = accent position(s)
    - Complex value with POS_SEP (0xFE) and REC_SEP (0xFF):
      [accent_bytes] + 0xFE + [compressed_tags] + 0xFF
    """
    
    file_path: Optional[Path] = None
    file_size: int = 0
    is_loaded: bool = False
    trie: Optional[marisa_trie.BytesTrie] = None
    total_entries: int = 0
    unique_words: Set[str] = field(default_factory=set)
    stress_patterns: Dict[str, int] = field(default_factory=dict)
    sample_entries: List[Tuple[str, bytes]] = field(default_factory=list)
    
    def load_from_file(self, file_path: str | Path) -> bool:
        """Load and parse stress.trie file using marisa_trie."""
        try:
            self.file_path = Path(file_path)
            
            if not self.file_path.exists():
                print(f"Error: File not found: {self.file_path}")
                return False
            
            # Load trie
            self.trie = marisa_trie.BytesTrie()
            self.trie.load(str(self.file_path))
            
            self.file_size = self.file_path.stat().st_size
            self.total_entries = len(self.trie)
            
            print(f"✓ Loaded {self.file_path.name}")
            print(f"  Size: {self.file_size / (1024*1024):.2f} MB")
            print(f"  Total entries: {self.total_entries:,}")
            
            return True
            
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def _get_tags(self) -> Dict[bytes, str]:
        """Get tag mapping."""
        return {
            b'\xFE': "POS-separator",
            b'\xFF': "Record-separator",
            b'\x11': "Number=Sing",
            b'\x12': "Number=Plur",
            b'\x20': "Case=Nom",
            b'\x21': "Case=Gen",
            b'\x22': "Case=Dat",
            b'\x23': "Case=Acc",
            b'\x24': "Case=Ins",
            b'\x25': "Case=Loc",
            b'\x26': "Case=Voc",
            b'\x30': "Gender=Neut",
            b'\x31': "Gender=Masc",
            b'\x32': "Gender=Fem",
            b'\x41': "VerbForm=Inf",
            b'\x42': "VerbForm=Conv",
            b'\x50': "Person=0",
            b'\x61': "upos=NOUN",
            b'\x62': "upos=ADJ",
            b'\x63': "upos=INTJ",
            b'\x64': "upos=CCONJ",
            b'\x65': "upos=PART",
            b'\x66': "upos=PRON",
            b'\x67': "upos=VERB",
            b'\x68': "upos=PROPN",
            b'\x69': "upos=ADV",
            b'\x6A': "upos=NOUN",
            b'\x6B': "upos=NUM",
            b'\x6C': "upos=ADP",
        }
    
    def analyze_entries(self, sample_size: int = 100) -> Dict:
        """Analyze trie entries and extract statistics."""
        if not self.trie:
            return {"error": "No trie loaded"}
        
        analysis = {
            "total_entries": self.total_entries,
            "file_size": self.file_size,
            "file_size_mb": round(self.file_size / (1024*1024), 2),
            "samples": [],
            "samples_with_tags": [],
            "stress_patterns": {},
            "entries_with_tags": 0,
            "entries_with_ambiguity": 0,
        }
        
        # Analyze sample entries
        POS_SEP = b'\xFE'
        REC_SEP = b'\xFF'
        
        count = 0
        tags_count = 0
        first_samples_count = 0
        
        print("\n🔍 Scanning trie for entries with morphological tags...")
        
        for idx, (word, value) in enumerate(self.trie.items()):
            self.unique_words.add(word)
            
            # Parse value
            stress_info = self._parse_value(value, POS_SEP, REC_SEP)
            
            # Store first N samples for display
            if first_samples_count < sample_size:
                sample_entry = {
                    "word": word,
                    "value_length": len(value),
                    "value_hex": value.hex()[:50],  # First 50 chars of hex
                    "stress_info": stress_info,
                }
                analysis["samples"].append(sample_entry)
                first_samples_count += 1
            
            # Check for morphological tags throughout entire trie
            if REC_SEP in value:
                analysis["entries_with_tags"] += 1
                tags_count += 1
                # Keep first 20 entries with tags for detailed display
                if tags_count <= 20:
                    sample_entry = {
                        "word": word,
                        "value_length": len(value),
                        "value_hex": value.hex(),  # Full hex for tagged entries
                        "stress_info": stress_info,
                    }
                    analysis["samples_with_tags"].append(sample_entry)
            
            # Check for ambiguity (multiple stress options)
            if len(stress_info) > 1:
                analysis["entries_with_ambiguity"] += 1
            
            # Progress indicator every 100k entries
            if (idx + 1) % 100000 == 0:
                print(f"   Processed {idx + 1:,} entries... Found {analysis['entries_with_tags']:,} with tags")
        
        print(f"✓ Scan complete: {idx + 1:,} total entries")
        analysis["unique_words_in_sample"] = len(self.unique_words)
        
        return analysis
    
    def _parse_value(self, value: bytes, POS_SEP: bytes, REC_SEP: bytes) -> List[Dict]:
        """Parse encoded value from trie."""
        result = []
        
        if REC_SEP not in value:
            # Simple case: just accent position(s)
            accents = [int(b) for b in value if b != 0]
            result.append({
                "accents": accents,
                "tags": []
            })
        else:
            # Complex case: multiple entries with tags
            items = value.split(REC_SEP)
            for item in items:
                if item:
                    accents_part, _, tags_part = item.partition(POS_SEP)
                    accents = [int(b) for b in accents_part if b != 0]
                    tags = self._decompress_tags(tags_part)
                    result.append({
                        "accents": accents,
                        "tags": tags
                    })
        
        return result
    
    def _decompress_tags(self, tags_bytes: bytes) -> List[str]:
        """Decompress tag bytes to string tags."""
        tags_mapping = self._get_tags()
        result = []
        for byte in tags_bytes:
            tag_bytes = bytes([byte])
            if tag_bytes in tags_mapping:
                result.append(tags_mapping[tag_bytes])
        return result
    
    def print_detailed_analysis(self, sample_size: int = 100):
        """Print detailed analysis of the trie file."""
        analysis = self.analyze_entries(sample_size)
        
        print("\n" + "=" * 100)
        print("UKRAINIAN STRESS TRIE FILE - DETAILED ANALYSIS")
        print("=" * 100)
        print(f"File: {self.file_path}")
        print(f"Size: {analysis['file_size_mb']} MB ({analysis['file_size']:,} bytes)")
        print(f"Total dictionary entries: {analysis['total_entries']:,}")
        
        print(f"\nStatistics (from sample of {sample_size} entries):")
        print(f"  Unique words: {analysis['unique_words_in_sample']}")
        print(f"  Entries with morphological tags: {analysis['entries_with_tags']}")
        print(f"  Entries with stress ambiguity: {analysis['entries_with_ambiguity']}")
        
        print(f"\n" + "-" * 100)
        print(f"Sample Entries (first {min(sample_size, len(analysis['samples']))} words):")
        print(f"{'#':<4} {'Word':<25} {'Accents':<15} {'Value (hex)':<30}")
        print("-" * 100)
        
        for i, sample in enumerate(analysis['samples'][:50], 1):
            word = sample['word']
            stress_info = sample['stress_info']
            
            # Format accents
            accents_str = ", ".join(str(s['accents']) for s in stress_info if s['accents'])
            if not accents_str:
                accents_str = "—"
            
            value_hex = sample['value_hex']
            
            print(f"{i:<4} {word:<25} {accents_str:<15} {value_hex:<30}")
        
        # Show entries WITH morphological tags
        if analysis['samples_with_tags']:
            print(f"\n" + "=" * 100)
            print(f"ENTRIES WITH MORPHOLOGICAL TAGS (first {len(analysis['samples_with_tags'])} found):")
            print("=" * 100)
            
            for i, sample in enumerate(analysis['samples_with_tags'], 1):
                word = sample['word']
                print(f"\n{i}. Word: '{word}'")
                print(f"   Value (hex): {sample['value_hex']}")
                print(f"   Value length: {sample['value_length']} bytes")
                
                stress_info = sample['stress_info']
                for j, info in enumerate(stress_info, 1):
                    print(f"   Option {j}:")
                    print(f"     - Stress positions: {info['accents']}")
                    if info['tags']:
                        print(f"     - Morphological tags:")
                        for tag in info['tags']:
                            print(f"       • {tag}")
                    else:
                        print(f"     - Morphological tags: (none)")
        else:
            print(f"\n⚠ Note: No entries with morphological tags found in the first {sample_size} entries.")
            print(f"  This may indicate the trie uses a simpler format than expected,")
            print(f"  or morphological data is stored differently.")
        
        print("\n" + "=" * 100 + "\n")


def analyze_stress_trie(file_path: str | Path) -> StressTrieData:
    """
    Convenience function to analyze stress.trie file.
    """
    trie = StressTrieData()
    if trie.load_from_file(file_path):
        trie.print_detailed_analysis()
    return trie


if __name__ == "__main__":
    import sys
    
    default_path = Path(__file__).parent / "raw_data" / "stress.trie"
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_path
    
    if Path(file_path).exists():
        analyze_stress_trie(file_path)
    else:
        print(f"Error: File not found: {file_path}")
        print(f"Expected default location: {default_path}")
        sys.exit(1)
        sys.exit(1)
