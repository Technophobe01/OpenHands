"""Memory visibility scopes for hierarchical working memory.

CCA defines three visibility scopes that control when memory entries
are included in the agent's context:

- SESSION: Persistent across the entire session (e.g., repo structure, user goals)
- ENTRY: Visible for the current task/entry point (e.g., current file being edited)
- RUNNABLE: Immediate action context, most volatile (e.g., command output)

Higher scope = more persistent, lower scope = more transient.
"""

from enum import IntEnum


class MemoryScope(IntEnum):
    """Visibility scope for memory entries.

    Scopes are ordered by persistence level (higher = more persistent).
    When compressing, lower scope entries are compressed first.
    """

    # Immediate action context - most transient
    # Examples: command outputs, tool results, intermediate calculations
    RUNNABLE = 0

    # Current task/entry context - moderately persistent
    # Examples: current file contents, active function being worked on
    ENTRY = 1

    # Session-level context - most persistent
    # Examples: repository structure, user requirements, architecture decisions
    SESSION = 2

    @classmethod
    def from_string(cls, value: str) -> 'MemoryScope':
        """Create scope from string name."""
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f'Invalid scope: {value}. Valid: {[s.name for s in cls]}')

    def __str__(self) -> str:
        return self.name.lower()


# Default compression order: compress lower scopes first
COMPRESSION_ORDER = [MemoryScope.RUNNABLE, MemoryScope.ENTRY, MemoryScope.SESSION]
