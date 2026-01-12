"""Tests for NoteStore."""

import tempfile
from pathlib import Path

import pytest

from openhands.memory.notes.note import Note, NoteType
from openhands.memory.notes.note_store import NoteStore


@pytest.fixture
def temp_notes_dir():
    """Create a temporary directory for notes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def note_store(temp_notes_dir):
    """Create a NoteStore with temporary directory."""
    return NoteStore(notes_dir=temp_notes_dir)


class TestNoteStoreInit:
    """Tests for NoteStore initialization."""

    def test_init_creates_directories(self, temp_notes_dir):
        """Test that init creates note type directories."""
        store = NoteStore(notes_dir=temp_notes_dir)

        # Check that subdirectories are created for each type
        for note_type in NoteType:
            type_dir = temp_notes_dir / note_type.value
            assert type_dir.exists()
            assert type_dir.is_dir()

    def test_init_with_default_dir(self):
        """Test that default directory is set."""
        store = NoteStore()
        assert store.notes_dir is not None
        assert 'notes' in str(store.notes_dir)


class TestNoteStoreSave:
    """Tests for saving notes."""

    def test_save_note(self, note_store, temp_notes_dir):
        """Test saving a note."""
        note = Note(
            title='Test Save',
            content='Test content',
            keywords=['test'],
        )

        path = note_store.save(note)

        assert path.exists()
        assert path.suffix == '.md'
        assert 'solution' in str(path)

    def test_save_different_types(self, note_store, temp_notes_dir):
        """Test saving notes of different types."""
        for note_type in NoteType:
            note = Note(
                title=f'Test {note_type.value}',
                content='Content',
                note_type=note_type,
            )
            path = note_store.save(note)
            assert note_type.value in str(path)


class TestNoteStoreLoad:
    """Tests for loading notes."""

    def test_load_existing_note(self, note_store):
        """Test loading an existing note."""
        original = Note(
            title='Load Test',
            content='Content to load',
            keywords=['load', 'test'],
        )
        note_store.save(original)

        loaded = note_store.load(original.note_id)

        assert loaded is not None
        assert loaded.title == original.title
        assert loaded.keywords == original.keywords

    def test_load_nonexistent_note(self, note_store):
        """Test loading a note that doesn't exist."""
        result = note_store.load('nonexistent_id')
        assert result is None


class TestNoteStoreDelete:
    """Tests for deleting notes."""

    def test_delete_existing_note(self, note_store):
        """Test deleting an existing note."""
        note = Note(title='To Delete', content='Content')
        note_store.save(note)

        result = note_store.delete(note.note_id)

        assert result is True
        assert note_store.load(note.note_id) is None

    def test_delete_nonexistent_note(self, note_store):
        """Test deleting a nonexistent note."""
        result = note_store.delete('nonexistent')
        assert result is False


class TestNoteStoreList:
    """Tests for listing notes."""

    def test_list_all_empty(self, note_store):
        """Test listing when no notes exist."""
        notes = note_store.list_all()
        assert notes == []

    def test_list_all_notes(self, note_store):
        """Test listing all notes."""
        for i in range(3):
            note = Note(title=f'Note {i}', content='Content')
            note_store.save(note)

        notes = note_store.list_all()
        assert len(notes) == 3

    def test_list_by_type(self, note_store):
        """Test listing notes filtered by type."""
        # Create notes of different types
        note_store.save(Note(title='Solution', content='C', note_type=NoteType.SOLUTION))
        note_store.save(Note(title='Failure', content='C', note_type=NoteType.FAILURE))
        note_store.save(Note(title='Pattern', content='C', note_type=NoteType.PATTERN))

        solutions = note_store.list_all(NoteType.SOLUTION)
        failures = note_store.list_all(NoteType.FAILURE)

        assert len(solutions) == 1
        assert solutions[0].title == 'Solution'
        assert len(failures) == 1
        assert failures[0].title == 'Failure'

    def test_list_sorted_by_date(self, note_store):
        """Test that listing returns notes sorted by date (newest first)."""
        from datetime import datetime, timedelta

        # Create notes with different timestamps
        old_note = Note(title='Old', content='C')
        old_note.created_at = datetime.now() - timedelta(days=1)
        note_store.save(old_note)

        new_note = Note(title='New', content='C')
        note_store.save(new_note)

        notes = note_store.list_all()
        assert notes[0].title == 'New'
        assert notes[1].title == 'Old'


class TestNoteStoreSearch:
    """Tests for searching notes."""

    def test_search_by_keywords(self, note_store):
        """Test keyword search."""
        note_store.save(Note(
            title='Python Tips',
            content='C',
            keywords=['python', 'tips', 'programming'],
        ))
        note_store.save(Note(
            title='Java Tips',
            content='C',
            keywords=['java', 'tips'],
        ))

        results = note_store.search_by_keywords('python programming')

        assert len(results) == 1
        note, matched = results[0]
        assert note.title == 'Python Tips'
        assert 'python' in matched

    def test_search_by_keywords_multiple_matches(self, note_store):
        """Test keyword search with multiple matching notes."""
        note_store.save(Note(
            title='API Design',
            content='C',
            keywords=['api', 'rest', 'design'],
        ))
        note_store.save(Note(
            title='API Testing',
            content='C',
            keywords=['api', 'testing'],
        ))

        results = note_store.search_by_keywords('api')

        assert len(results) == 2

    def test_search_by_keywords_with_limit(self, note_store):
        """Test keyword search with limit."""
        for i in range(10):
            note_store.save(Note(
                title=f'Note {i}',
                content='C',
                keywords=['common'],
            ))

        results = note_store.search_by_keywords('common', limit=3)
        assert len(results) == 3

    def test_search_by_keywords_empty_query(self, note_store):
        """Test keyword search with empty query."""
        results = note_store.search_by_keywords('')
        assert results == []

    def test_search_by_title(self, note_store):
        """Test title search."""
        note_store.save(Note(title='Database Optimization', content='C'))
        note_store.save(Note(title='API Design', content='C'))

        results = note_store.search_by_title('database')

        assert len(results) == 1
        assert results[0].title == 'Database Optimization'

    def test_search_by_title_case_insensitive(self, note_store):
        """Test that title search is case-insensitive."""
        note_store.save(Note(title='Python Best Practices', content='C'))

        results = note_store.search_by_title('PYTHON')
        assert len(results) == 1


class TestNoteStoreRelated:
    """Tests for related file searches."""

    def test_get_related_notes(self, note_store):
        """Test finding notes by related files."""
        note_store.save(Note(
            title='Main Module',
            content='C',
            related_files=['src/main.py', 'src/utils.py'],
        ))
        note_store.save(Note(
            title='Test Module',
            content='C',
            related_files=['tests/test_main.py'],
        ))

        results = note_store.get_related_notes(['src/main.py'])

        assert len(results) == 1
        assert results[0].title == 'Main Module'

    def test_get_related_notes_multiple_files(self, note_store):
        """Test related search with multiple file paths."""
        note_store.save(Note(
            title='Note 1',
            content='C',
            related_files=['file1.py'],
        ))
        note_store.save(Note(
            title='Note 2',
            content='C',
            related_files=['file2.py'],
        ))

        results = note_store.get_related_notes(['file1.py', 'file2.py'])
        assert len(results) == 2


class TestNoteStoreFailures:
    """Tests for failure note retrieval."""

    def test_get_failure_notes(self, note_store):
        """Test getting failure notes."""
        note_store.save(Note(
            title='Success',
            content='C',
            note_type=NoteType.SOLUTION,
        ))
        note_store.save(Note(
            title='Failure 1',
            content='C',
            note_type=NoteType.FAILURE,
            error_message='IndexError',
        ))
        note_store.save(Note(
            title='Failure 2',
            content='C',
            note_type=NoteType.FAILURE,
            error_message='ValueError',
        ))

        failures = note_store.get_failure_notes()

        assert len(failures) == 2
        assert all(f.note_type == NoteType.FAILURE for f in failures)

    def test_get_failure_notes_with_pattern(self, note_store):
        """Test filtering failure notes by error pattern."""
        note_store.save(Note(
            title='Index Error',
            content='C',
            note_type=NoteType.FAILURE,
            error_message='IndexError: list index out of range',
        ))
        note_store.save(Note(
            title='Value Error',
            content='C',
            note_type=NoteType.FAILURE,
            error_message='ValueError: invalid value',
        ))

        results = note_store.get_failure_notes(error_pattern='index')

        assert len(results) == 1
        assert 'IndexError' in results[0].error_message


class TestNoteStoreCount:
    """Tests for counting notes."""

    def test_count_all(self, note_store):
        """Test counting all notes."""
        for i in range(5):
            note_store.save(Note(title=f'Note {i}', content='C'))

        assert note_store.count() == 5

    def test_count_by_type(self, note_store):
        """Test counting by type."""
        note_store.save(Note(title='S1', content='C', note_type=NoteType.SOLUTION))
        note_store.save(Note(title='S2', content='C', note_type=NoteType.SOLUTION))
        note_store.save(Note(title='F1', content='C', note_type=NoteType.FAILURE))

        assert note_store.count(NoteType.SOLUTION) == 2
        assert note_store.count(NoteType.FAILURE) == 1
        assert note_store.count(NoteType.PATTERN) == 0


class TestNoteStoreClear:
    """Tests for clearing notes."""

    def test_clear_all(self, note_store):
        """Test clearing all notes."""
        for i in range(5):
            note_store.save(Note(title=f'Note {i}', content='C'))

        deleted = note_store.clear()

        assert deleted == 5
        assert note_store.count() == 0

    def test_clear_by_type(self, note_store):
        """Test clearing notes by type."""
        note_store.save(Note(title='S1', content='C', note_type=NoteType.SOLUTION))
        note_store.save(Note(title='S2', content='C', note_type=NoteType.SOLUTION))
        note_store.save(Note(title='F1', content='C', note_type=NoteType.FAILURE))

        deleted = note_store.clear(NoteType.SOLUTION)

        assert deleted == 2
        assert note_store.count(NoteType.SOLUTION) == 0
        assert note_store.count(NoteType.FAILURE) == 1
