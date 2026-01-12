"""Hierarchical Working Memory implementation.

CCA F1: Working memory organized as a hierarchy with visibility scopes.
The Architect planner compresses context when thresholds are exceeded.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel

from openhands.core.logger import openhands_logger as logger
from openhands.events.event import Event
from openhands.memory.hierarchical.scopes import COMPRESSION_ORDER, MemoryScope

if TYPE_CHECKING:
    from openhands.memory.view import View


@dataclass
class MemoryEntry:
    """A single entry in hierarchical memory.

    Each entry has content, a scope determining its visibility/persistence,
    and metadata for tracking and retrieval.
    """

    content: str
    scope: MemoryScope
    entry_id: str = ''
    timestamp: float = field(default_factory=time.time)
    source_event_id: int | None = None
    keywords: list[str] = field(default_factory=list)
    importance: float = 1.0  # 0.0 to 1.0, used for compression decisions
    compressed: bool = False
    summary: str | None = None  # If compressed, this holds the summary

    def __post_init__(self):
        if not self.entry_id:
            # Generate unique ID from content hash and timestamp
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:8]
            self.entry_id = f'{self.scope.name.lower()}_{content_hash}_{int(self.timestamp)}'

    @property
    def effective_content(self) -> str:
        """Return summary if compressed, otherwise full content."""
        return self.summary if self.compressed and self.summary else self.content

    def compress(self, summary: str) -> None:
        """Mark entry as compressed with given summary."""
        self.compressed = True
        self.summary = summary


@dataclass
class MemoryNode:
    """A node in the memory hierarchy tree.

    Each node can have children (sub-nodes) and entries at the same scope.
    This enables hierarchical organization like:
      SESSION (root)
      ├── repo_structure
      ├── user_requirements
      └── ENTRY (task1)
          ├── file_contents
          └── RUNNABLE (action1)
              └── command_output
    """

    scope: MemoryScope
    name: str = ''
    entries: list[MemoryEntry] = field(default_factory=list)
    children: list[MemoryNode] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_entry(self, entry: MemoryEntry) -> None:
        """Add an entry to this node."""
        self.entries.append(entry)

    def add_child(self, child: MemoryNode) -> None:
        """Add a child node."""
        self.children.append(child)

    def get_all_entries(self, min_scope: MemoryScope | None = None) -> list[MemoryEntry]:
        """Get all entries from this node and descendants, optionally filtered by scope."""
        result = []
        if min_scope is None or self.scope >= min_scope:
            result.extend(self.entries)
        for child in self.children:
            result.extend(child.get_all_entries(min_scope))
        return result

    def total_tokens_estimate(self) -> int:
        """Estimate total tokens in this subtree (rough: 4 chars per token)."""
        total_chars = self._total_chars()
        return total_chars // 4

    def _total_chars(self) -> int:
        """Get total characters in this subtree."""
        total = sum(len(e.effective_content) for e in self.entries)
        for child in self.children:
            total += child._total_chars()
        return total


class HierarchicalWorkingMemory(BaseModel):
    """CCA F1: Hierarchical working memory with visibility scopes.

    Organizes memory into a tree structure with three scope levels:
    - SESSION: Persistent context (repo info, user goals)
    - ENTRY: Current task context (files being edited)
    - RUNNABLE: Immediate action context (command outputs)

    The Architect compresses lower scopes first when context limits are exceeded.
    """

    class Config:
        arbitrary_types_allowed = True

    # Root node at SESSION scope
    root: MemoryNode = MemoryNode(scope=MemoryScope.SESSION, name='session_root')

    # Current active nodes at each scope level
    current_entry: MemoryNode | None = None
    current_runnable: MemoryNode | None = None

    # Configuration
    max_tokens: int = 100000  # Maximum tokens before compression
    compression_threshold: float = 0.8  # Trigger compression at 80% capacity
    min_entries_before_compress: int = 10  # Don't compress if very few entries

    # Tracking
    _checksum: str = ''
    _last_compression_time: float = 0.0

    def add_entry(
        self,
        content: str,
        scope: MemoryScope,
        source_event_id: int | None = None,
        keywords: list[str] | None = None,
        importance: float = 1.0,
    ) -> MemoryEntry:
        """Add content to memory with specified visibility scope.

        Args:
            content: The text content to store
            scope: Visibility scope (SESSION, ENTRY, RUNNABLE)
            source_event_id: Optional ID of the event that generated this content
            keywords: Optional keywords for retrieval
            importance: Importance score 0.0-1.0 for compression decisions

        Returns:
            The created MemoryEntry
        """
        entry = MemoryEntry(
            content=content,
            scope=scope,
            source_event_id=source_event_id,
            keywords=keywords or [],
            importance=importance,
        )

        # Add to appropriate node based on scope
        target_node = self._get_node_for_scope(scope)
        target_node.add_entry(entry)

        # Invalidate checksum
        self._checksum = ''

        logger.debug(f'Added memory entry: scope={scope.name}, id={entry.entry_id}')
        return entry

    def _get_node_for_scope(self, scope: MemoryScope) -> MemoryNode:
        """Get the appropriate node for the given scope."""
        if scope == MemoryScope.SESSION:
            return self.root
        elif scope == MemoryScope.ENTRY:
            if self.current_entry is None:
                self.current_entry = MemoryNode(scope=MemoryScope.ENTRY, name='entry')
                self.root.add_child(self.current_entry)
            return self.current_entry
        else:  # RUNNABLE
            if self.current_runnable is None:
                parent = self.current_entry or self.root
                self.current_runnable = MemoryNode(
                    scope=MemoryScope.RUNNABLE, name='runnable'
                )
                parent.add_child(self.current_runnable)
            return self.current_runnable

    def start_new_entry(self, name: str = '') -> MemoryNode:
        """Start a new ENTRY scope (e.g., new task).

        This closes the current ENTRY and RUNNABLE scopes and creates a new one.
        """
        # Close current runnable
        self.current_runnable = None

        # Create new entry node
        self.current_entry = MemoryNode(
            scope=MemoryScope.ENTRY, name=name or f'entry_{int(time.time())}'
        )
        self.root.add_child(self.current_entry)
        self._checksum = ''

        logger.debug(f'Started new entry scope: {self.current_entry.name}')
        return self.current_entry

    def start_new_runnable(self, name: str = '') -> MemoryNode:
        """Start a new RUNNABLE scope (e.g., new action)."""
        parent = self.current_entry or self.root
        self.current_runnable = MemoryNode(
            scope=MemoryScope.RUNNABLE, name=name or f'runnable_{int(time.time())}'
        )
        parent.add_child(self.current_runnable)
        self._checksum = ''

        logger.debug(f'Started new runnable scope: {self.current_runnable.name}')
        return self.current_runnable

    def should_compress(self) -> bool:
        """Check if compression should be triggered."""
        total_tokens = self.root.total_tokens_estimate()
        threshold_tokens = int(self.max_tokens * self.compression_threshold)

        total_entries = len(self.get_all_entries())
        if total_entries < self.min_entries_before_compress:
            return False

        return total_tokens > threshold_tokens

    def get_all_entries(self, min_scope: MemoryScope | None = None) -> list[MemoryEntry]:
        """Get all entries, optionally filtered by minimum scope."""
        return self.root.get_all_entries(min_scope)

    def get_entries_for_context(self) -> list[MemoryEntry]:
        """Get entries suitable for LLM context, ordered by scope then time."""
        entries = self.get_all_entries()
        # Sort by scope (higher first) then by timestamp
        entries.sort(key=lambda e: (-e.scope, e.timestamp))
        return entries

    def get_checksum(self) -> str:
        """Get checksum of current memory state for caching."""
        if not self._checksum:
            content = ''.join(e.entry_id for e in self.get_all_entries())
            self._checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self._checksum

    def to_context_string(self, max_tokens: int | None = None) -> str:
        """Convert memory to a string suitable for LLM context.

        Args:
            max_tokens: Optional token limit (uses self.max_tokens if not specified)

        Returns:
            Formatted string with memory organized by scope
        """
        max_tokens = max_tokens or self.max_tokens
        entries = self.get_entries_for_context()

        sections = {
            MemoryScope.SESSION: [],
            MemoryScope.ENTRY: [],
            MemoryScope.RUNNABLE: [],
        }

        for entry in entries:
            sections[entry.scope].append(entry.effective_content)

        result_parts = []
        if sections[MemoryScope.SESSION]:
            result_parts.append(
                '<SESSION_CONTEXT>\n'
                + '\n---\n'.join(sections[MemoryScope.SESSION])
                + '\n</SESSION_CONTEXT>'
            )
        if sections[MemoryScope.ENTRY]:
            result_parts.append(
                '<TASK_CONTEXT>\n'
                + '\n---\n'.join(sections[MemoryScope.ENTRY])
                + '\n</TASK_CONTEXT>'
            )
        if sections[MemoryScope.RUNNABLE]:
            result_parts.append(
                '<IMMEDIATE_CONTEXT>\n'
                + '\n---\n'.join(sections[MemoryScope.RUNNABLE])
                + '\n</IMMEDIATE_CONTEXT>'
            )

        return '\n\n'.join(result_parts)

    def get_entries_to_compress(self) -> list[MemoryEntry]:
        """Get entries that should be compressed, in compression order.

        Returns entries from lower scopes first, then by importance within scope.
        """
        entries_to_compress = []

        for scope in COMPRESSION_ORDER:
            scope_entries = [
                e for e in self.get_all_entries() if e.scope == scope and not e.compressed
            ]
            # Sort by importance (lower importance compressed first)
            scope_entries.sort(key=lambda e: e.importance)
            entries_to_compress.extend(scope_entries)

        return entries_to_compress

    @classmethod
    def from_events(cls, events: list[Event]) -> 'HierarchicalWorkingMemory':
        """Create hierarchical memory from event list.

        Analyzes events and assigns them to appropriate scopes:
        - System messages, repo info -> SESSION
        - File operations, task context -> ENTRY
        - Command outputs, tool results -> RUNNABLE
        """
        memory = cls()

        for event in events:
            # Determine scope based on event type
            scope = cls._classify_event_scope(event)
            content = str(event)
            keywords = cls._extract_keywords(event)

            memory.add_entry(
                content=content,
                scope=scope,
                source_event_id=event.id if event.id >= 0 else None,
                keywords=keywords,
            )

        return memory

    @staticmethod
    def _classify_event_scope(event: Event) -> MemoryScope:
        """Classify an event into a memory scope based on its type."""
        from openhands.events.action.agent import AgentThinkAction
        from openhands.events.action.commands import CmdRunAction, IPythonRunCellAction
        from openhands.events.action.files import FileEditAction, FileReadAction
        from openhands.events.action.message import MessageAction, SystemMessageAction
        from openhands.events.observation.commands import (
            CmdOutputObservation,
            IPythonRunCellObservation,
        )
        from openhands.events.observation.files import (
            FileEditObservation,
            FileReadObservation,
        )

        # SESSION scope: system messages, high-level context
        if isinstance(event, SystemMessageAction):
            return MemoryScope.SESSION

        # ENTRY scope: file operations, task-related thinking
        if isinstance(
            event, (FileReadAction, FileEditAction, FileReadObservation, FileEditObservation)
        ):
            return MemoryScope.ENTRY

        if isinstance(event, AgentThinkAction):
            return MemoryScope.ENTRY

        # RUNNABLE scope: command execution, immediate results
        if isinstance(
            event,
            (
                CmdRunAction,
                IPythonRunCellAction,
                CmdOutputObservation,
                IPythonRunCellObservation,
            ),
        ):
            return MemoryScope.RUNNABLE

        # Default: user messages and other actions go to ENTRY
        if isinstance(event, MessageAction):
            return MemoryScope.ENTRY

        # Everything else defaults to RUNNABLE (most transient)
        return MemoryScope.RUNNABLE

    @staticmethod
    def _extract_keywords(event: Event) -> list[str]:
        """Extract keywords from an event for retrieval."""
        keywords = []

        # Extract from event class name
        keywords.append(type(event).__name__)

        # Extract file paths if present
        if hasattr(event, 'path'):
            keywords.append(str(event.path))
        if hasattr(event, 'file_path'):
            keywords.append(str(event.file_path))

        return keywords
