"""Tests for NoteTaker."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from openhands.events.event import Event, EventSource
from openhands.memory.notes.note import Note, NoteType
from openhands.memory.notes.note_store import NoteStore
from openhands.memory.notes.note_taker import NoteTaker


def create_test_event(message: str, id: int = -1) -> Event:
    """Create a test event."""
    event = Event()
    event._message = message
    event.timestamp = datetime.now()
    if id >= 0:
        event._id = id
    event._source = EventSource.USER
    return event


@pytest.fixture
def temp_notes_dir():
    """Create a temporary directory for notes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def note_store(temp_notes_dir):
    """Create a NoteStore with temporary directory."""
    return NoteStore(notes_dir=temp_notes_dir)


@pytest.fixture
def note_taker(note_store):
    """Create a NoteTaker without LLM."""
    return NoteTaker(note_store=note_store, session_id='test_session')


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    mock = MagicMock()

    mock_message = MagicMock()
    mock_message.content = """
<title>Test Note Title</title>
<keywords>test, example, mock</keywords>
<problem>Test problem description</problem>
<solution>Test solution description</solution>
"""

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock.completion.return_value = mock_response

    return mock


class TestNoteTakerInit:
    """Tests for NoteTaker initialization."""

    def test_init_with_defaults(self, note_store):
        """Test initialization with default parameters."""
        taker = NoteTaker(note_store=note_store)

        assert taker.note_store == note_store
        assert taker.llm is None
        assert taker.session_id == ''
        assert taker.auto_note_threshold == 5

    def test_init_with_session_id(self, note_store):
        """Test initialization with session ID."""
        taker = NoteTaker(note_store=note_store, session_id='session_123')
        assert taker.session_id == 'session_123'

    def test_init_with_custom_threshold(self, note_store):
        """Test initialization with custom auto-note threshold."""
        taker = NoteTaker(note_store=note_store, auto_note_threshold=10)
        assert taker.auto_note_threshold == 10


class TestNoteTakerCreateNote:
    """Tests for creating notes."""

    def test_create_basic_note(self, note_taker):
        """Test creating a basic note."""
        note = note_taker.create_note(
            title='Test Note',
            content='Test content',
        )

        assert note.title == 'Test Note'
        assert note.content == 'Test content'
        assert note.session_id == 'test_session'
        assert note_taker.note_store.count() == 1

    def test_create_note_with_all_fields(self, note_taker):
        """Test creating a note with all fields."""
        note = note_taker.create_note(
            title='Complete Note',
            content='Full content',
            note_type=NoteType.PATTERN,
            keywords=['pattern', 'best-practice'],
            problem_context='The problem was...',
            solution='The solution is...',
            related_files=['file1.py', 'file2.py'],
            source_event_ids=[1, 2, 3],
        )

        assert note.note_type == NoteType.PATTERN
        assert note.keywords == ['pattern', 'best-practice']
        assert note.problem_context == 'The problem was...'
        assert note.solution == 'The solution is...'
        assert note.related_files == ['file1.py', 'file2.py']
        assert note.source_event_ids == [1, 2, 3]

    def test_create_failure_note(self, note_taker):
        """Test creating a failure note."""
        note = note_taker.create_note(
            title='Failed Attempt',
            content='Description of failure',
            note_type=NoteType.FAILURE,
            error_message='IndexError: list index out of range',
        )

        assert note.note_type == NoteType.FAILURE
        assert note.error_message == 'IndexError: list index out of range'


class TestNoteTakerHindsight:
    """Tests for hindsight note creation."""

    def test_create_hindsight_note_without_llm(self, note_taker):
        """Test creating hindsight note without LLM."""
        events = [create_test_event(f'Event {i}', id=i) for i in range(3)]

        note = note_taker.create_hindsight_note(
            error_message='Test error occurred',
            events=events,
            related_files=['broken.py'],
        )

        assert note is not None
        assert note.note_type == NoteType.FAILURE
        assert 'Test error' in note.title
        assert note.error_message == 'Test error occurred'
        assert 'broken.py' in note.related_files

    def test_create_hindsight_note_with_llm(self, note_store, mock_llm):
        """Test creating hindsight note with LLM."""
        # Set up mock response for hindsight
        mock_llm.completion.return_value.choices[0].message.content = """
<title>IndexError in Data Processing</title>
<keywords>index, error, list, data</keywords>
<problem>Processing empty list without check</problem>
<error_analysis>No bounds checking before access</error_analysis>
<lessons>Always validate list length before indexing</lessons>
"""

        taker = NoteTaker(note_store=note_store, llm=mock_llm)
        events = [create_test_event('Processing data', id=1)]

        note = taker.create_hindsight_note(
            error_message='IndexError: list index out of range',
            events=events,
        )

        assert note is not None
        assert 'IndexError' in note.title
        assert 'error' in note.keywords
        mock_llm.completion.assert_called()


class TestNoteTakerRecall:
    """Tests for recalling notes."""

    def test_recall_notes_by_keyword(self, note_taker):
        """Test recalling notes by keyword."""
        note_taker.create_note(
            title='Python Tips',
            content='Python best practices',
            keywords=['python', 'tips', 'best-practices'],
        )
        note_taker.create_note(
            title='Java Tips',
            content='Java best practices',
            keywords=['java', 'tips'],
        )

        results = note_taker.recall_notes('python programming')

        assert len(results) == 1
        assert results[0].title == 'Python Tips'

    def test_recall_notes_empty_results(self, note_taker):
        """Test recall with no matches."""
        note_taker.create_note(
            title='Python Tips',
            content='Content',
            keywords=['python'],
        )

        results = note_taker.recall_notes('rust programming')
        assert results == []

    def test_recall_notes_with_limit(self, note_taker):
        """Test recall with limit."""
        for i in range(10):
            note_taker.create_note(
                title=f'Note {i}',
                content='Content',
                keywords=['common'],
            )

        results = note_taker.recall_notes('common', limit=3)
        assert len(results) == 3

    def test_recall_for_context_with_files(self, note_taker):
        """Test context-based recall with related files."""
        note_taker.create_note(
            title='Main Module',
            content='Content',
            keywords=['main'],
            related_files=['src/main.py'],
        )
        note_taker.create_note(
            title='Other Module',
            content='Content',
            keywords=['other'],
            related_files=['src/other.py'],
        )

        results = note_taker.recall_for_context(
            context='working on code',
            related_files=['src/main.py'],
        )

        assert len(results) >= 1
        # Should find the note related to main.py
        titles = [n.title for n in results]
        assert 'Main Module' in titles

    def test_recall_failure_notes(self, note_taker):
        """Test recalling failure notes."""
        note_taker.create_note(
            title='Success',
            content='C',
            note_type=NoteType.SOLUTION,
        )
        note_taker.create_note(
            title='Failure 1',
            content='C',
            note_type=NoteType.FAILURE,
            error_message='IndexError',
        )

        failures = note_taker.recall_failure_notes()

        assert len(failures) == 1
        assert failures[0].title == 'Failure 1'

    def test_recall_failure_notes_by_pattern(self, note_taker):
        """Test recalling failure notes filtered by error pattern."""
        note_taker.create_note(
            title='Index Error',
            content='C',
            note_type=NoteType.FAILURE,
            error_message='IndexError: out of range',
        )
        note_taker.create_note(
            title='Value Error',
            content='C',
            note_type=NoteType.FAILURE,
            error_message='ValueError: invalid',
        )

        results = note_taker.recall_failure_notes(error_pattern='index')

        assert len(results) == 1
        assert 'Index' in results[0].title


class TestNoteTakerEvents:
    """Tests for event handling."""

    def test_add_event(self, note_taker):
        """Test adding events to pending queue."""
        event = create_test_event('Test event')
        note_taker.add_event(event)

        assert len(note_taker._pending_events) == 1

    def test_clear_pending_events(self, note_taker):
        """Test clearing pending events."""
        for i in range(5):
            note_taker.add_event(create_test_event(f'Event {i}'))

        note_taker.clear_pending_events()

        assert len(note_taker._pending_events) == 0

    def test_set_context(self, note_taker):
        """Test setting context."""
        note_taker.set_context('Working on API module')
        assert note_taker._context == 'Working on API module'


class TestNoteTakerAutoNote:
    """Tests for automatic note creation detection."""

    def test_should_create_note_few_events(self, note_taker):
        """Test that auto-note isn't triggered with few events."""
        for i in range(3):
            note_taker.add_event(create_test_event(f'Event {i}'))

        assert not note_taker.should_create_note()

    def test_should_create_note_no_indicators(self, note_taker):
        """Test that auto-note isn't triggered without solution indicators."""
        for i in range(10):
            note_taker.add_event(create_test_event(f'Working on task {i}'))

        assert not note_taker.should_create_note()

    def test_should_create_note_with_success(self, note_taker):
        """Test that auto-note is triggered with success indicators."""
        for i in range(5):
            note_taker.add_event(create_test_event(f'Event {i}'))
        note_taker.add_event(create_test_event('Task completed successfully'))

        assert note_taker.should_create_note()

    def test_should_create_note_with_fixed(self, note_taker):
        """Test that auto-note is triggered with 'fixed' indicator."""
        for i in range(5):
            note_taker.add_event(create_test_event(f'Debugging {i}'))
        note_taker.add_event(create_test_event('Bug fixed'))

        assert note_taker.should_create_note()


class TestNoteTakerStats:
    """Tests for statistics."""

    def test_get_stats_empty(self, note_taker):
        """Test stats with no notes."""
        stats = note_taker.get_stats()

        assert stats['total'] == 0
        assert stats['pending_events'] == 0

    def test_get_stats_with_notes(self, note_taker):
        """Test stats with notes."""
        note_taker.create_note(title='S1', content='C', note_type=NoteType.SOLUTION)
        note_taker.create_note(title='S2', content='C', note_type=NoteType.SOLUTION)
        note_taker.create_note(title='F1', content='C', note_type=NoteType.FAILURE)

        for i in range(3):
            note_taker.add_event(create_test_event(f'Event {i}'))

        stats = note_taker.get_stats()

        assert stats['total'] == 3
        assert stats['by_type']['solution'] == 2
        assert stats['by_type']['failure'] == 1
        assert stats['pending_events'] == 3


class TestNoteTakerWithLLM:
    """Tests for LLM-based note creation."""

    def test_create_note_from_llm(self, note_store, mock_llm):
        """Test creating note using LLM."""
        taker = NoteTaker(note_store=note_store, llm=mock_llm)

        # Add some events
        for i in range(5):
            taker.add_event(create_test_event(f'Event {i}', id=i))

        note = taker.create_note_from_llm()

        assert note is not None
        assert note.title == 'Test Note Title'
        assert 'test' in note.keywords
        assert len(taker._pending_events) == 0  # Events cleared

    def test_create_note_from_llm_no_events(self, note_store, mock_llm):
        """Test LLM note creation with no events."""
        taker = NoteTaker(note_store=note_store, llm=mock_llm)

        note = taker.create_note_from_llm()

        assert note is None

    def test_create_note_from_llm_no_llm(self, note_taker):
        """Test LLM note creation without LLM configured."""
        for i in range(5):
            note_taker.add_event(create_test_event(f'Event {i}'))

        note = note_taker.create_note_from_llm()

        assert note is None
