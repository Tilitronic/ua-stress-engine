"""
trie.py — Compact binary trie encoder / decoder for Ukrainian stress data.

Binary format (.ctrie v2)
=========================
Header — 20 bytes
  [0..3]  magic      b'UKST'
  [4]     version    0x02
  [5]     reserved   0x00
  [6..9]  node_count uint32 LE  — total number of nodes
  [10..13] word_count uint32 LE — total number of terminal words
  [14..17] alphabet_size uint8 padded to 4 bytes
  [18..19] reserved

Alphabet table — alphabet_size × 3 bytes each
  [0]     char_byte  UTF-8 single byte OR 0xFF for multi-byte
  [1..2]  codepoint  uint16 LE  — Unicode codepoint of the character
  Sorted by codepoint ascending.  Index into this table = char_id (0..N-1).

Node array — node_count × NODE_SIZE bytes each
  NODE_SIZE = 8 bytes per node:
  [0]    char_id    uint8  — index into alphabet table (0xFF = root sentinel)
  [1]    stress     uint8  — 0xFF = not a word end; 0..10 = primary stressed vowel index
  [2]    flags      uint8  — bit 0: variative; bit 1: has_children;
                             bit 2: heteronym; bit 3: has_secondary_stress
  [3]    stress2    uint8  — secondary stress vowel index (0xFF = absent)
  [4..7] first_child_idx  uint32 LE — index of first child node (0 = no children)

Stress classification:
  flags & 0x0F == 0x00               → unique   (one unambiguous stress)
  flags & FLAG_HAS_SECONDARY set,
    flags & FLAG_VARIATIVE set        → variative (both stresses equally valid;
                                        e.g. по́милка / поми́лка)
  flags & FLAG_HAS_SECONDARY set,
    flags & FLAG_HETERONYM  set        → heteronym (different meanings / forms;
                                        e.g. за́мок / замо́к, бло́хи / блохи́)

Children of each node are contiguous in the array and can be scanned linearly.
Lookup is O(word_length).
"""

from __future__ import annotations

import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ── Alphabet ──────────────────────────────────────────────────────────────────
# Ukrainian lowercase letters + apostrophe + hyphen.
# Everything outside this set is rejected during normalisation.
UA_ALPHABET_CHARS = (
    "абвгґдеєжзиіїйклмнопрстуфхцчшщьюя"   # 33 Ukrainian letters
    "\u02bc"   # ʼ modifier letter apostrophe
    "-"        # hyphen (compound words)
)
# Build a codepoint → char_id mapping at module load time
_CHAR_TO_ID: Dict[int, int] = {ord(c): i for i, c in enumerate(UA_ALPHABET_CHARS)}
ALPHABET_SIZE = len(UA_ALPHABET_CHARS)

NODE_SIZE = 8       # bytes per node in the serialised array
HEADER_SIZE = 20    # bytes
ALPHABET_ENTRY_SIZE = 3

NO_CHILD = 0        # sentinel: node has no children
NO_STRESS = 0xFF    # sentinel: node is not a word terminal

FLAG_VARIATIVE      = 0x01  # both stresses equally valid (по́милка / поми́лка)
FLAG_HAS_CHILDREN   = 0x02
FLAG_HETERONYM      = 0x04  # different meanings / forms (за́мок / замо́к)
FLAG_HAS_SECONDARY  = 0x08  # stress2 byte is valid

# Backward-compat alias (v1 consumers checked this bit for "uncertain")
_FLAG_UNCERTAIN_LEGACY = FLAG_VARIATIVE | FLAG_HETERONYM

VERSION = 0x02


# ── In-memory trie ────────────────────────────────────────────────────────────

@dataclass
class TrieNode:
    char_id: int = 0xFF          # 0xFF for root
    stress: int = NO_STRESS      # 0xFF or 0..10 (primary)
    stress2: int = NO_STRESS     # secondary stress (0xFF = absent)
    flags: int = 0
    children: Dict[int, "TrieNode"] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.stress != NO_STRESS

    @property
    def is_heteronym(self) -> bool:
        return bool(self.flags & FLAG_HETERONYM)

    @property
    def is_variative(self) -> bool:
        return bool(self.flags & FLAG_VARIATIVE)


class TrieBuilder:
    """Insert (word, stress_index, stress2, is_variative, is_heteronym) tuples and produce a TrieNode tree."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self.word_count = 0
        self._node_count = 1  # root

    def insert(
        self,
        word: str,
        stress: int,
        stress2: int = NO_STRESS,
        variative: bool = False,
        heteronym: bool = False,
    ) -> bool:
        """
        Insert a normalised lowercase word.

        Args:
            word:      Normalised lowercase Ukrainian word.
            stress:    Primary (most common) stressed vowel index.
            stress2:   Secondary stressed vowel index, or NO_STRESS.
            variative: True when both stress positions are orthographically
                       valid simultaneously (e.g. по́милка / поми́лка).
            heteronym: True when different stress positions correspond to
                       different meanings or grammatical forms
                       (e.g. за́мок/замо́к, бло́хи/блохи́).

        Returns True if word was inserted, False if any character is outside
        the Ukrainian alphabet and the word was skipped.
        """
        node = self.root
        for ch in word:
            cid = _CHAR_TO_ID.get(ord(ch))
            if cid is None:
                return False   # skip words with unmapped characters
            if cid not in node.children:
                node.children[cid] = TrieNode(char_id=cid)
                self._node_count += 1
            node = node.children[cid]

        if not node.is_terminal:
            self.word_count += 1

        node.stress = stress
        if stress2 != NO_STRESS:
            node.stress2 = stress2
            node.flags |= FLAG_HAS_SECONDARY
        if variative:
            node.flags |= FLAG_VARIATIVE
        if heteronym:
            node.flags |= FLAG_HETERONYM
        return True

    @property
    def node_count(self) -> int:
        return self._node_count


# ── Serialisation ─────────────────────────────────────────────────────────────

def serialize(
    builder: TrieBuilder,
    progress: Optional[Callable[[int, int], None]] = None,
) -> bytes:
    """
    Encode a TrieBuilder to .ctrie binary bytes.

    The root node is always at index 0. Children of each node are laid out
    contiguously immediately after their parent's last sibling group.

    Args:
        builder:  Populated TrieBuilder instance.
        progress: Optional callback ``fn(current_node, total_nodes)`` called
                  every 100 000 nodes so callers can display a progress bar.
    """
    total_nodes = builder.node_count
    report_every = 100_000

    # BFS-order flattening — deque gives O(1) popleft vs O(n) list.pop(0)
    nodes_flat: List[TrieNode] = []
    index_map: Dict[int, int] = {}

    queue: deque[TrieNode] = deque([builder.root])
    while queue:
        node = queue.popleft()
        idx = len(nodes_flat)
        index_map[id(node)] = idx
        nodes_flat.append(node)

        if progress is not None and idx % report_every == 0 and idx > 0:
            progress(idx, total_nodes)

        for k in sorted(node.children):
            queue.append(node.children[k])

    if progress is not None:
        progress(len(nodes_flat), total_nodes)

    node_count = len(nodes_flat)
    word_count = builder.word_count

    # ── Header ────────────────────────────────────────────────────────────────
    header = struct.pack(
        "<4sBBIII",
        b"UKST",         # magic
        VERSION,         # version 0x02
        0x00,            # reserved
        node_count,
        word_count,
        ALPHABET_SIZE,
    )
    # pad header to HEADER_SIZE
    header = header + b"\x00" * (HEADER_SIZE - len(header))

    # ── Alphabet table ────────────────────────────────────────────────────────
    alphabet_bytes = bytearray()
    for ch in UA_ALPHABET_CHARS:
        cp = ord(ch)
        ch_byte = cp if cp < 256 else 0xFF
        alphabet_bytes += struct.pack("<BH", ch_byte, cp)

    # ── Node array ────────────────────────────────────────────────────────────
    node_bytes = bytearray()
    for node in nodes_flat:
        flags = node.flags
        if node.children:
            flags |= FLAG_HAS_CHILDREN
            first_child = index_map[id(node.children[min(node.children)])]
        else:
            first_child = NO_CHILD

        node_bytes += struct.pack(
            "<BBBBI",
            node.char_id,
            node.stress,
            flags,
            node.stress2,   # 0xFF when absent
            first_child,
        )

    return bytes(header) + bytes(alphabet_bytes) + bytes(node_bytes)


# ── Deserialisation (Python-side, for tests) ──────────────────────────────────

@dataclass
class FlatNode:
    char_id: int
    stress: int
    flags: int
    stress2: int
    first_child: int


def deserialize(data: bytes) -> Tuple[List[FlatNode], List[str]]:
    """
    Decode a .ctrie binary blob.

    Returns (flat_nodes, alphabet_chars) for use in tests / inspection.
    """
    # Header
    magic = data[0:4]
    assert magic == b"UKST", f"Bad magic: {magic!r}"
    version = data[4]
    assert version in (0x01, 0x02), f"Unsupported .ctrie version: {version}"
    node_count, word_count, alpha_size = struct.unpack_from("<III", data, 6)

    # Alphabet
    offset = HEADER_SIZE
    alphabet: List[str] = []
    for _ in range(alpha_size):
        _ch_byte, cp = struct.unpack_from("<BH", data, offset)
        alphabet.append(chr(cp))
        offset += ALPHABET_ENTRY_SIZE

    # Nodes
    flat: List[FlatNode] = []
    for _ in range(node_count):
        char_id, stress, flags, stress2, first_child = struct.unpack_from("<BBBBI", data, offset)
        flat.append(FlatNode(char_id=char_id, stress=stress, flags=flags, stress2=stress2, first_child=first_child))
        offset += NODE_SIZE

    return flat, alphabet


def lookup(data: bytes, word: str) -> Optional[Tuple[int, bool]]:
    """
    Look up a word in a serialised .ctrie blob.

    Returns (stress_index, is_uncertain) or None if not found.
    is_uncertain is True for both variative and heteronym words.
    Used by the Python test suite — the browser uses the JS version.
    For richer results (stresses list + type) use lookup_full().
    """
    flat, alphabet = deserialize(data)
    alpha_map = {ch: i for i, ch in enumerate(alphabet)}

    node_idx = 0   # root
    for ch in word.lower():
        cid = alpha_map.get(ch)
        if cid is None:
            return None

        # scan children of node_idx
        fc = flat[node_idx].first_child
        if fc == NO_CHILD or not (flat[node_idx].flags & FLAG_HAS_CHILDREN):
            return None

        found = False
        i = fc
        # children are contiguous; scan until char_id doesn't match any sibling
        while i < len(flat):
            child = flat[i]
            # siblings share same parent; detect end by checking parent's
            # first_child range — since children are BFS-contiguous and sorted,
            # we scan until we find a node whose parent_idx != node_idx.
            # Simpler: scan sorted siblings (char_ids are inserted sorted).
            if child.char_id == cid:
                node_idx = i
                found = True
                break
            # If we've gone past possible siblings, stop.
            # Siblings of node_idx are exactly the nodes inserted from fc onward
            # that share the same parent. We stop when we reach a node that
            # was inserted by a different parent. Since BFS order guarantees
            # all children of a node are contiguous, we stop when a node's
            # char_id is not in the parent's children set — but we don't store
            # parent pointers. Conservative bound: scan max ALPHABET_SIZE nodes.
            if i - fc >= ALPHABET_SIZE:
                break
            i += 1

        if not found:
            return None

    node = flat[node_idx]
    if node.stress == NO_STRESS:
        return None
    is_uncertain = bool(node.flags & (FLAG_VARIATIVE | FLAG_HETERONYM))
    return (node.stress, is_uncertain)


def lookup_full(
    data: bytes, word: str
) -> Optional[Tuple[List[int], str]]:
    """
    Look up a word and return (stresses, type) or None.

    type is one of:
      "unique"    — single unambiguous stress
      "variative" — multiple orthographically valid stress positions
      "heteronym" — multiple positions corresponding to different meanings/forms
    """
    flat, alphabet = deserialize(data)
    alpha_map = {ch: i for i, ch in enumerate(alphabet)}

    node_idx = 0
    for ch in word.lower():
        cid = alpha_map.get(ch)
        if cid is None:
            return None
        fc = flat[node_idx].first_child
        if fc == NO_CHILD or not (flat[node_idx].flags & FLAG_HAS_CHILDREN):
            return None
        found = False
        i = fc
        while i < len(flat):
            child = flat[i]
            if child.char_id == cid:
                node_idx = i
                found = True
                break
            if i - fc >= ALPHABET_SIZE:
                break
            i += 1
        if not found:
            return None

    node = flat[node_idx]
    if node.stress == NO_STRESS:
        return None

    stresses: List[int] = [node.stress]
    if (node.flags & FLAG_HAS_SECONDARY) and node.stress2 != NO_STRESS:
        stresses.append(node.stress2)

    if node.flags & FLAG_VARIATIVE:
        kind = "variative"
    elif node.flags & FLAG_HETERONYM:
        kind = "heteronym"
    else:
        kind = "unique"

    return stresses, kind
