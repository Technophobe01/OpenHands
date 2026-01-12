"""Tests for Note model."""

from datetime import datetime

import pytest

from openhands.memory.notes.note import Note, NoteType


class TestNoteType:
    """Tests for NoteType enum."""

    def test_note_type_values(self):
        """Test NoteType enum values."""
        assert NoteType.SOLUTION.value == 'solution'
        assert NoteType.FAILURE.value == 'failure'
        assert NoteType.PATTERN.value == 'pattern'
        assert NoteType.ARCHITECTURE.value == 'architecture'
        assert NoteType.DEBUG.value == 'debug'
        assert NoteType.REFERENCE.value == 'reference'

    def test_from_string_valid(self):
        """Test creating NoteType from valid string."""
        assert NoteType.from_string('solution') == NoteType.SOLUTION
        assert NoteType.from_string('SOLUTION') == NoteType.SOLUTION
        assert NoteType.from_string('failure') == NoteType.FAILURE

    def test_from_string_invalid(self):
        """Test creating NoteType from invalid string defaults to REFERENCE."""
        assert NoteType.from_string('invalid') == NoteType.REFERENCE
        assert NoteType.from_string('') == NoteType.REFERENCE


class TestNote:
    """Tests for Note dataclass."""

    def test_note_creation(self):
        """Test creating a basic note."""
        note = Note(
            title='Test Note',
            content='Test content',
        )
        assert note.title == 'Test Note'
        assert note.content == 'Test content'
        assert note.note_type == NoteType.SOLUTION  # default
        assert note.note_id.startswith('solution_')
        assert note.keywords == []

    def test_note_with_keywords(self):
        """Test creating note with keywords."""
        note = Note(
            title='Error Handling',
            content='How to handle errors',
            keywords=['error', 'exception', 'handling'],
        )
        assert note.keywords == ['error', 'exception', 'handling']

    def test_note_id_generation(self):
        """Test automatic note ID generation."""
        note1 = Note(title='Note 1', content='Content')
        note2 = Note(title='Note 2', content='Content')

        # Different titles should generate different IDs
        assert note1.note_id != note2.note_id
        assert 'solution_' in note1.note_id

    def test_note_with_custom_id(self):
        """Test note with custom ID."""
        note = Note(
            title='Test',
            content='Content',
            note_id='custom_id_123',
        )
        assert note.note_id == 'custom_id_123'

    def test_match_keywords_found(self):
        """Test keyword matching with matches."""
        note = Note(
            title='Database Query',
            content='Content',
            keywords=['database', 'sql', 'query', 'optimization'],
        )

        matched = note.match_keywords('How to optimize SQL database queries?')
        assert 'database' in matched
        assert 'sql' in matched
        assert 'query' in matched

    def test_match_keywords_case_insensitive(self):
        """Test that keyword matching is case-insensitive."""
        note = Note(
            title='Test',
            content='Content',
            keywords=['Python', 'ASYNC'],
        )

        matched = note.match_keywords('python async programming')
        assert len(matched) == 2

    def test_match_keywords_no_match(self):
        """Test keyword matching with no matches."""
        note = Note(
            title='Test',
            content='Content',
            keywords=['javascript', 'react'],
        )

        matched = note.match_keywords('python programming')
        assert matched == []

    def test_match_keywords_empty_query(self):
        """Test keyword matching with empty query."""
        note = Note(
            title='Test',
            content='Content',
            keywords=['test'],
        )

        matched = note.match_keywords('')
        assert matched == []


class TestNoteMarkdown:
    """Tests for Note markdown serialization."""

    def test_to_markdown(self):
        """Test converting note to markdown."""
        note = Note(
            title='Test Note',
            content='Main content here',
            note_type=NoteType.SOLUTION,
            keywords=['test', 'example'],
            problem_context='The problem was...',
            solution='The solution is...',
        )

        markdown = note.to_markdown()

        # Check frontmatter markers
        assert markdown.startswith('---')
        assert 'title: Test Note' in markdown
        assert "type: solution" in markdown
        assert 'keywords:' in markdown

        # Check content sections
        assert '## Problem Context' in markdown
        assert 'The problem was...' in markdown
        assert '## Solution' in markdown
        assert 'The solution is...' in markdown

    def test_to_markdown_with_error(self):
        """Test markdown with error message (failure note)."""
        note = Note(
            title='Failed Test',
            content='Content',
            note_type=NoteType.FAILURE,
            error_message='IndexError: list index out of range',
        )

        markdown = note.to_markdown()
        assert '## Error' in markdown
        assert 'IndexError' in markdown

    def test_to_markdown_with_related_files(self):
        """Test markdown with related files."""
        note = Note(
            title='File Changes',
            content='Content',
            related_files=['src/main.py', 'tests/test_main.py'],
        )

        markdown = note.to_markdown()
        assert '## Related Files' in markdown
        assert '`src/main.py`' in markdown
        assert '`tests/test_main.py`' in markdown

    def test_from_markdown(self):
        """Test parsing note from markdown."""
        markdown = """---
id: test_id_123
title: Parsed Note
type: solution
keywords:
  - parsing
  - test
created: '2024-01-15T10:30:00'
session_id: session_abc
---

## Problem Context

The problem context.

## Solution

The solution details.
"""

        note = Note.from_markdown(markdown)

        assert note.note_id == 'test_id_123'
        assert note.title == 'Parsed Note'
        assert note.note_type == NoteType.SOLUTION
        assert 'parsing' in note.keywords
        assert 'test' in note.keywords
        assert note.session_id == 'session_abc'
        assert note.problem_context == 'The problem context.'
        assert note.solution == 'The solution details.'

    def test_from_markdown_with_error(self):
        """Test parsing failure note with error."""
        markdown = """---
id: failure_123
title: Failed Attempt
type: failure
keywords:
  - error
created: '2024-01-15T10:30:00'
---

## Error

```
ValueError: invalid value
```
"""

        note = Note.from_markdown(markdown)

        assert note.note_type == NoteType.FAILURE
        assert note.error_message == 'ValueError: invalid value'

    def test_roundtrip_serialization(self):
        """Test that serialization is reversible."""
        original = Note(
            title='Roundtrip Test',
            content='Test content',
            note_type=NoteType.PATTERN,
            keywords=['test', 'roundtrip'],
            problem_context='Problem description',
            solution='Solution description',
            related_files=['file1.py', 'file2.py'],
            session_id='session_123',
        )

        markdown = original.to_markdown()
        restored = Note.from_markdown(markdown)

        assert restored.title == original.title
        assert restored.note_type == original.note_type
        assert restored.keywords == original.keywords
        assert restored.problem_context == original.problem_context
        assert restored.solution == original.solution
        assert restored.related_files == original.related_files
        assert restored.session_id == original.session_id


class TestNoteStr:
    """Tests for Note string representations."""

    def test_str_representation(self):
        """Test __str__ method."""
        note = Note(
            title='My Note',
            content='Content',
            note_type=NoteType.DEBUG,
        )
        assert str(note) == 'Note(debug): My Note'

    def test_repr_representation(self):
        """Test __repr__ method."""
        note = Note(
            title='My Note',
            content='Content',
            note_id='test_123',
            note_type=NoteType.SOLUTION,
        )
        repr_str = repr(note)
        assert 'test_123' in repr_str
        assert 'My Note' in repr_str
        assert 'solution' in repr_str
