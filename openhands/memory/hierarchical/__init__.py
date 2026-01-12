"""Hierarchical Working Memory System.

Implementation of CCA (Confucius Code Agent) F1: Hierarchical working memory
with visibility scopes and architect-based compression.

Reference: arXiv:2512.10398
"""

from openhands.memory.hierarchical.scopes import MemoryScope
from openhands.memory.hierarchical.working_memory import (
    HierarchicalWorkingMemory,
    MemoryEntry,
    MemoryNode,
)

__all__ = [
    'MemoryScope',
    'MemoryEntry',
    'MemoryNode',
    'HierarchicalWorkingMemory',
]
