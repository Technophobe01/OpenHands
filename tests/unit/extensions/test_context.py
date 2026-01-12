"""Tests for the RunContext class."""

import pytest

from openhands.extensions.context import RunContext


class TestRunContext:
    """Tests for RunContext."""

    def test_basic_creation(self):
        """Test basic RunContext creation."""
        ctx = RunContext(session_id='test-123')
        assert ctx.session_id == 'test-123'
        assert ctx.iteration == 0
        assert ctx.state is None
        assert ctx.history == []
        assert ctx.working_dir == ''
        assert ctx.extra_data == {}
        assert ctx.metadata == {}

    def test_with_all_fields(self):
        """Test RunContext with all fields populated."""
        ctx = RunContext(
            session_id='test-456',
            iteration=5,
            working_dir='/workspace',
            extra_data={'key': 'value'},
            metadata={'ext_key': 'ext_value'},
        )
        assert ctx.session_id == 'test-456'
        assert ctx.iteration == 5
        assert ctx.working_dir == '/workspace'
        assert ctx.extra_data == {'key': 'value'}
        assert ctx.metadata == {'ext_key': 'ext_value'}

    def test_get_recent_events_empty(self):
        """Test get_recent_events with empty history."""
        ctx = RunContext(session_id='test')
        assert ctx.get_recent_events() == []
        assert ctx.get_recent_events(5) == []

    def test_get_recent_events(self):
        """Test get_recent_events returns most recent events."""
        class MockEvent:
            def __init__(self, id):
                self.id = id

        events = [MockEvent(i) for i in range(10)]
        ctx = RunContext(session_id='test', history=events)

        recent = ctx.get_recent_events(3)
        assert len(recent) == 3
        assert recent[0].id == 7
        assert recent[1].id == 8
        assert recent[2].id == 9

    def test_get_recent_events_more_than_available(self):
        """Test get_recent_events when requesting more than available."""
        class MockEvent:
            def __init__(self, id):
                self.id = id

        events = [MockEvent(i) for i in range(3)]
        ctx = RunContext(session_id='test', history=events)

        recent = ctx.get_recent_events(10)
        assert len(recent) == 3

    def test_get_events_by_type(self):
        """Test filtering events by type."""
        class TypeA:
            pass

        class TypeB:
            pass

        events = [TypeA(), TypeB(), TypeA(), TypeB(), TypeA()]
        ctx = RunContext(session_id='test', history=events)

        type_a_events = ctx.get_events_by_type(TypeA)
        assert len(type_a_events) == 3

        type_b_events = ctx.get_events_by_type(TypeB)
        assert len(type_b_events) == 2

    def test_get_events_by_type_none_found(self):
        """Test get_events_by_type when no events match."""
        class TypeA:
            pass

        class TypeB:
            pass

        events = [TypeA(), TypeA()]
        ctx = RunContext(session_id='test', history=events)

        result = ctx.get_events_by_type(TypeB)
        assert result == []

    def test_set_and_get_metadata(self):
        """Test setting and getting metadata."""
        ctx = RunContext(session_id='test')

        ctx.set_metadata('key1', 'value1')
        ctx.set_metadata('key2', {'nested': 'dict'})
        ctx.set_metadata('key3', [1, 2, 3])

        assert ctx.get_metadata('key1') == 'value1'
        assert ctx.get_metadata('key2') == {'nested': 'dict'}
        assert ctx.get_metadata('key3') == [1, 2, 3]

    def test_get_metadata_default(self):
        """Test get_metadata with default value."""
        ctx = RunContext(session_id='test')

        assert ctx.get_metadata('nonexistent') is None
        assert ctx.get_metadata('nonexistent', 'default') == 'default'
        assert ctx.get_metadata('nonexistent', 42) == 42

    def test_metadata_overwrite(self):
        """Test that metadata can be overwritten."""
        ctx = RunContext(session_id='test')

        ctx.set_metadata('key', 'original')
        assert ctx.get_metadata('key') == 'original'

        ctx.set_metadata('key', 'updated')
        assert ctx.get_metadata('key') == 'updated'

    def test_metadata_persists_across_operations(self):
        """Test that metadata persists across context operations."""
        ctx = RunContext(session_id='test')

        ctx.set_metadata('counter', 0)

        for i in range(5):
            current = ctx.get_metadata('counter')
            ctx.set_metadata('counter', current + 1)

        assert ctx.get_metadata('counter') == 5
