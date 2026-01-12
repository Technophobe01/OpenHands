"""Tests for hierarchical working memory."""

import pytest

from openhands.memory.hierarchical.scopes import MemoryScope
from openhands.memory.hierarchical.working_memory import (
    HierarchicalWorkingMemory,
    MemoryEntry,
    MemoryNode,
)


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""

    def test_entry_creation(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            content='Test content',
            scope=MemoryScope.SESSION,
        )
        assert entry.content == 'Test content'
        assert entry.scope == MemoryScope.SESSION
        assert entry.entry_id.startswith('session_')
        assert entry.importance == 1.0
        assert not entry.compressed

    def test_entry_with_custom_id(self):
        """Test creating entry with custom ID."""
        entry = MemoryEntry(
            content='Test',
            scope=MemoryScope.ENTRY,
            entry_id='custom_id_123',
        )
        assert entry.entry_id == 'custom_id_123'

    def test_entry_effective_content(self):
        """Test effective_content property."""
        entry = MemoryEntry(content='Full content', scope=MemoryScope.RUNNABLE)
        assert entry.effective_content == 'Full content'

        # After compression
        entry.compress('Summary')
        assert entry.effective_content == 'Summary'
        assert entry.compressed is True

    def test_entry_compression(self):
        """Test compressing an entry."""
        entry = MemoryEntry(content='Long content here', scope=MemoryScope.RUNNABLE)
        assert not entry.compressed
        assert entry.summary is None

        entry.compress('Short summary')
        assert entry.compressed
        assert entry.summary == 'Short summary'


class TestMemoryNode:
    """Tests for MemoryNode."""

    def test_node_creation(self):
        """Test creating a memory node."""
        node = MemoryNode(scope=MemoryScope.SESSION, name='root')
        assert node.scope == MemoryScope.SESSION
        assert node.name == 'root'
        assert node.entries == []
        assert node.children == []

    def test_add_entry(self):
        """Test adding entries to a node."""
        node = MemoryNode(scope=MemoryScope.SESSION)
        entry = MemoryEntry(content='Test', scope=MemoryScope.SESSION)
        node.add_entry(entry)
        assert len(node.entries) == 1
        assert node.entries[0] == entry

    def test_add_child(self):
        """Test adding child nodes."""
        parent = MemoryNode(scope=MemoryScope.SESSION)
        child = MemoryNode(scope=MemoryScope.ENTRY, name='task1')
        parent.add_child(child)
        assert len(parent.children) == 1
        assert parent.children[0] == child

    def test_get_all_entries(self):
        """Test getting all entries from subtree."""
        root = MemoryNode(scope=MemoryScope.SESSION)
        root.add_entry(MemoryEntry(content='Session entry', scope=MemoryScope.SESSION))

        child = MemoryNode(scope=MemoryScope.ENTRY)
        child.add_entry(MemoryEntry(content='Entry entry', scope=MemoryScope.ENTRY))
        root.add_child(child)

        grandchild = MemoryNode(scope=MemoryScope.RUNNABLE)
        grandchild.add_entry(
            MemoryEntry(content='Runnable entry', scope=MemoryScope.RUNNABLE)
        )
        child.add_child(grandchild)

        all_entries = root.get_all_entries()
        assert len(all_entries) == 3

    def test_get_all_entries_filtered(self):
        """Test getting entries filtered by minimum scope."""
        root = MemoryNode(scope=MemoryScope.SESSION)
        root.add_entry(MemoryEntry(content='Session', scope=MemoryScope.SESSION))

        child = MemoryNode(scope=MemoryScope.ENTRY)
        child.add_entry(MemoryEntry(content='Entry', scope=MemoryScope.ENTRY))
        root.add_child(child)

        grandchild = MemoryNode(scope=MemoryScope.RUNNABLE)
        grandchild.add_entry(MemoryEntry(content='Runnable', scope=MemoryScope.RUNNABLE))
        child.add_child(grandchild)

        # Filter to ENTRY and above
        filtered = root.get_all_entries(min_scope=MemoryScope.ENTRY)
        assert len(filtered) == 2  # SESSION and ENTRY only

        # Filter to SESSION only
        session_only = root.get_all_entries(min_scope=MemoryScope.SESSION)
        assert len(session_only) == 1

    def test_total_tokens_estimate(self):
        """Test token estimation."""
        node = MemoryNode(scope=MemoryScope.SESSION)
        # Add entry with 40 characters -> ~10 tokens
        node.add_entry(
            MemoryEntry(content='A' * 40, scope=MemoryScope.SESSION)
        )
        assert node.total_tokens_estimate() == 10


class TestHierarchicalWorkingMemory:
    """Tests for HierarchicalWorkingMemory."""

    def test_memory_creation(self):
        """Test creating hierarchical memory."""
        memory = HierarchicalWorkingMemory()
        assert memory.root.scope == MemoryScope.SESSION
        assert memory.current_entry is None
        assert memory.current_runnable is None

    def test_add_entry_session(self):
        """Test adding SESSION scope entry."""
        memory = HierarchicalWorkingMemory()
        entry = memory.add_entry('Session content', MemoryScope.SESSION)

        assert entry.scope == MemoryScope.SESSION
        assert len(memory.root.entries) == 1
        assert memory.root.entries[0] == entry

    def test_add_entry_creates_hierarchy(self):
        """Test that adding entries creates appropriate hierarchy."""
        memory = HierarchicalWorkingMemory()

        # Add ENTRY scope entry - should create current_entry
        memory.add_entry('Entry content', MemoryScope.ENTRY)
        assert memory.current_entry is not None
        assert memory.current_entry.scope == MemoryScope.ENTRY
        assert len(memory.current_entry.entries) == 1

        # Add RUNNABLE scope entry - should create current_runnable
        memory.add_entry('Runnable content', MemoryScope.RUNNABLE)
        assert memory.current_runnable is not None
        assert memory.current_runnable.scope == MemoryScope.RUNNABLE
        assert len(memory.current_runnable.entries) == 1

    def test_start_new_entry(self):
        """Test starting a new entry scope."""
        memory = HierarchicalWorkingMemory()

        # Add some entries
        memory.add_entry('First entry', MemoryScope.ENTRY)
        memory.add_entry('Some action', MemoryScope.RUNNABLE)

        first_entry = memory.current_entry

        # Start new entry
        new_entry = memory.start_new_entry('task2')
        assert new_entry != first_entry
        assert new_entry.name == 'task2'
        assert memory.current_runnable is None  # Runnable closed

    def test_start_new_runnable(self):
        """Test starting a new runnable scope."""
        memory = HierarchicalWorkingMemory()

        memory.add_entry('Action 1', MemoryScope.RUNNABLE)
        first_runnable = memory.current_runnable

        new_runnable = memory.start_new_runnable('action2')
        assert new_runnable != first_runnable
        assert new_runnable.name == 'action2'

    def test_should_compress(self):
        """Test compression threshold detection."""
        memory = HierarchicalWorkingMemory(
            max_tokens=100,
            compression_threshold=0.8,
            min_entries_before_compress=5,
        )

        # Add entries until compression should trigger
        for i in range(10):
            # Each entry ~10 tokens (40 chars)
            memory.add_entry('A' * 40, MemoryScope.RUNNABLE)

        # 10 entries * 10 tokens = 100 tokens = 100% > 80% threshold
        assert memory.should_compress()

    def test_should_not_compress_few_entries(self):
        """Test that compression is skipped with few entries."""
        memory = HierarchicalWorkingMemory(
            max_tokens=10,  # Very low
            compression_threshold=0.5,
            min_entries_before_compress=10,  # Requires 10 entries
        )

        # Add only 5 entries
        for i in range(5):
            memory.add_entry('Content', MemoryScope.RUNNABLE)

        # Even though over token limit, not enough entries
        assert not memory.should_compress()

    def test_get_all_entries(self):
        """Test getting all entries."""
        memory = HierarchicalWorkingMemory()

        memory.add_entry('Session', MemoryScope.SESSION)
        memory.add_entry('Entry', MemoryScope.ENTRY)
        memory.add_entry('Runnable', MemoryScope.RUNNABLE)

        all_entries = memory.get_all_entries()
        assert len(all_entries) == 3

    def test_get_entries_for_context(self):
        """Test getting entries ordered for context."""
        memory = HierarchicalWorkingMemory()

        # Add in reverse order
        memory.add_entry('Runnable', MemoryScope.RUNNABLE)
        memory.add_entry('Entry', MemoryScope.ENTRY)
        memory.add_entry('Session', MemoryScope.SESSION)

        entries = memory.get_entries_for_context()

        # Should be sorted by scope (SESSION first) then time
        assert entries[0].scope == MemoryScope.SESSION
        assert entries[-1].scope == MemoryScope.RUNNABLE

    def test_to_context_string(self):
        """Test converting memory to context string."""
        memory = HierarchicalWorkingMemory()

        memory.add_entry('Session info', MemoryScope.SESSION)
        memory.add_entry('Task info', MemoryScope.ENTRY)
        memory.add_entry('Command output', MemoryScope.RUNNABLE)

        context = memory.to_context_string()

        assert '<SESSION_CONTEXT>' in context
        assert 'Session info' in context
        assert '<TASK_CONTEXT>' in context
        assert 'Task info' in context
        assert '<IMMEDIATE_CONTEXT>' in context
        assert 'Command output' in context

    def test_get_entries_to_compress(self):
        """Test getting entries in compression order."""
        memory = HierarchicalWorkingMemory()

        memory.add_entry('Session', MemoryScope.SESSION, importance=1.0)
        memory.add_entry('Entry', MemoryScope.ENTRY, importance=0.5)
        memory.add_entry('Runnable high', MemoryScope.RUNNABLE, importance=0.8)
        memory.add_entry('Runnable low', MemoryScope.RUNNABLE, importance=0.2)

        to_compress = memory.get_entries_to_compress()

        # RUNNABLE first (most transient), sorted by importance
        assert to_compress[0].scope == MemoryScope.RUNNABLE
        assert to_compress[0].importance == 0.2  # Lower importance first
        assert to_compress[1].importance == 0.8

        # Then ENTRY
        assert to_compress[2].scope == MemoryScope.ENTRY

        # Then SESSION (last, most persistent)
        assert to_compress[3].scope == MemoryScope.SESSION

    def test_checksum_invalidation(self):
        """Test that checksum is invalidated on changes."""
        memory = HierarchicalWorkingMemory()
        memory.add_entry('Content', MemoryScope.SESSION)

        checksum1 = memory.get_checksum()

        # Add another entry
        memory.add_entry('More content', MemoryScope.ENTRY)

        checksum2 = memory.get_checksum()
        assert checksum1 != checksum2
