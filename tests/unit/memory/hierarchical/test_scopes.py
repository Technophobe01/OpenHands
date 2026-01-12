"""Tests for hierarchical memory scopes."""

import pytest

from openhands.memory.hierarchical.scopes import (
    COMPRESSION_ORDER,
    MemoryScope,
)


class TestMemoryScope:
    """Tests for MemoryScope enum."""

    def test_scope_ordering(self):
        """Test that scopes are ordered correctly by persistence."""
        assert MemoryScope.RUNNABLE < MemoryScope.ENTRY < MemoryScope.SESSION
        assert MemoryScope.RUNNABLE.value == 0
        assert MemoryScope.ENTRY.value == 1
        assert MemoryScope.SESSION.value == 2

    def test_from_string_valid(self):
        """Test creating scope from valid string."""
        assert MemoryScope.from_string('session') == MemoryScope.SESSION
        assert MemoryScope.from_string('ENTRY') == MemoryScope.ENTRY
        assert MemoryScope.from_string('Runnable') == MemoryScope.RUNNABLE

    def test_from_string_invalid(self):
        """Test creating scope from invalid string raises error."""
        with pytest.raises(ValueError, match='Invalid scope'):
            MemoryScope.from_string('invalid')

    def test_str_representation(self):
        """Test string representation of scopes."""
        assert str(MemoryScope.SESSION) == 'session'
        assert str(MemoryScope.ENTRY) == 'entry'
        assert str(MemoryScope.RUNNABLE) == 'runnable'

    def test_compression_order(self):
        """Test that compression order is correct (transient first)."""
        assert COMPRESSION_ORDER == [
            MemoryScope.RUNNABLE,
            MemoryScope.ENTRY,
            MemoryScope.SESSION,
        ]
        # Verify order compresses most transient first
        assert COMPRESSION_ORDER[0] == MemoryScope.RUNNABLE
        assert COMPRESSION_ORDER[-1] == MemoryScope.SESSION
