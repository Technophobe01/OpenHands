"""Note storage for the note-taking system.

CCA F2: Persistent storage of notes in markdown format with keyword-based retrieval.
Notes are stored in ~/.openhands/notes/ directory.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from openhands.core.logger import openhands_logger as logger
from openhands.memory.notes.note import Note, NoteType

if TYPE_CHECKING:
    pass


class NoteStore:
    """Storage manager for notes.

    Stores notes as markdown files with YAML frontmatter in a directory structure.
    Supports CRUD operations and keyword-based retrieval.

    Directory structure:
        ~/.openhands/notes/
        ├── solution/
        │   ├── solution_abc123_1234567890.md
        │   └── ...
        ├── failure/
        │   └── ...
        ├── pattern/
        │   └── ...
        └── ...
    """

    DEFAULT_NOTES_DIR = '~/.openhands/notes'

    def __init__(self, notes_dir: str | Path | None = None):
        """Initialize the note store.

        Args:
            notes_dir: Directory to store notes. Defaults to ~/.openhands/notes/
        """
        if notes_dir is None:
            notes_dir = self.DEFAULT_NOTES_DIR

        if isinstance(notes_dir, str):
            if notes_dir.startswith('~'):
                notes_dir = os.path.expanduser(notes_dir)
            notes_dir = Path(notes_dir)

        self.notes_dir = notes_dir
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create the notes directory structure if it doesn't exist."""
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        # Create subdirectories for each note type
        for note_type in NoteType:
            (self.notes_dir / note_type.value).mkdir(exist_ok=True)

    def _get_note_path(self, note: Note) -> Path:
        """Get the file path for a note."""
        return self.notes_dir / note.note_type.value / f'{note.note_id}.md'

    def _get_note_path_by_id(self, note_id: str) -> Path | None:
        """Find the path of a note by its ID (searches all type directories)."""
        for note_type in NoteType:
            path = self.notes_dir / note_type.value / f'{note_id}.md'
            if path.exists():
                return path
        return None

    def save(self, note: Note) -> Path:
        """Save a note to the store.

        Args:
            note: The note to save

        Returns:
            Path to the saved note file
        """
        path = self._get_note_path(note)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(note.to_markdown(), encoding='utf-8')
        logger.debug(f'Saved note: {note.note_id} to {path}')
        return path

    def load(self, note_id: str) -> Note | None:
        """Load a note by its ID.

        Args:
            note_id: The note's unique ID

        Returns:
            The loaded Note, or None if not found
        """
        path = self._get_note_path_by_id(note_id)
        if path is None or not path.exists():
            logger.debug(f'Note not found: {note_id}')
            return None

        try:
            return Note.from_file(path)
        except Exception as e:
            logger.error(f'Error loading note {note_id}: {e}')
            return None

    def delete(self, note_id: str) -> bool:
        """Delete a note by its ID.

        Args:
            note_id: The note's unique ID

        Returns:
            True if deleted, False if not found
        """
        path = self._get_note_path_by_id(note_id)
        if path is None or not path.exists():
            return False

        try:
            path.unlink()
            logger.debug(f'Deleted note: {note_id}')
            return True
        except Exception as e:
            logger.error(f'Error deleting note {note_id}: {e}')
            return False

    def list_all(self, note_type: NoteType | None = None) -> list[Note]:
        """List all notes, optionally filtered by type.

        Args:
            note_type: Optional type filter

        Returns:
            List of all notes matching the filter
        """
        notes = []
        types_to_search = [note_type] if note_type else list(NoteType)

        for nt in types_to_search:
            type_dir = self.notes_dir / nt.value
            if not type_dir.exists():
                continue

            for path in type_dir.glob('*.md'):
                try:
                    note = Note.from_file(path)
                    notes.append(note)
                except Exception as e:
                    logger.warning(f'Error loading note from {path}: {e}')

        # Sort by creation time (newest first)
        notes.sort(key=lambda n: n.created_at, reverse=True)
        return notes

    def search_by_keywords(
        self,
        query: str,
        note_type: NoteType | None = None,
        limit: int = 10,
    ) -> list[tuple[Note, list[str]]]:
        """Search notes by keyword matching.

        Uses case-insensitive substring matching, similar to microagent triggers.

        Args:
            query: The query string to match against keywords
            note_type: Optional type filter
            limit: Maximum number of results to return

        Returns:
            List of (note, matched_keywords) tuples, sorted by match relevance
        """
        if not query:
            return []

        matches: list[tuple[Note, list[str]]] = []
        all_notes = self.list_all(note_type)

        for note in all_notes:
            matched_keywords = note.match_keywords(query)
            if matched_keywords:
                matches.append((note, matched_keywords))

        # Sort by number of matched keywords (descending), then by creation time
        matches.sort(key=lambda x: (-len(x[1]), -x[0].created_at.timestamp()))

        return matches[:limit]

    def search_by_title(self, title_query: str, limit: int = 10) -> list[Note]:
        """Search notes by title (case-insensitive substring match).

        Args:
            title_query: The title substring to search for
            limit: Maximum number of results

        Returns:
            List of matching notes
        """
        if not title_query:
            return []

        title_lower = title_query.lower()
        matching = []

        for note in self.list_all():
            if title_lower in note.title.lower():
                matching.append(note)

        matching.sort(key=lambda n: n.created_at, reverse=True)
        return matching[:limit]

    def get_related_notes(
        self,
        file_paths: list[str],
        limit: int = 5,
    ) -> list[Note]:
        """Find notes related to specific files.

        Args:
            file_paths: List of file paths to match
            limit: Maximum number of results

        Returns:
            List of notes that reference any of the given files
        """
        if not file_paths:
            return []

        related = []
        file_set = set(file_paths)

        for note in self.list_all():
            if note.related_files and file_set.intersection(note.related_files):
                related.append(note)

        related.sort(key=lambda n: n.created_at, reverse=True)
        return related[:limit]

    def get_failure_notes(
        self,
        error_pattern: str | None = None,
        limit: int = 10,
    ) -> list[Note]:
        """Get failure (hindsight) notes, optionally filtered by error pattern.

        Args:
            error_pattern: Optional error message substring to match
            limit: Maximum number of results

        Returns:
            List of failure notes
        """
        failure_notes = self.list_all(NoteType.FAILURE)

        if error_pattern:
            pattern_lower = error_pattern.lower()
            failure_notes = [
                n
                for n in failure_notes
                if n.error_message and pattern_lower in n.error_message.lower()
            ]

        return failure_notes[:limit]

    def count(self, note_type: NoteType | None = None) -> int:
        """Count notes, optionally filtered by type.

        Args:
            note_type: Optional type filter

        Returns:
            Number of notes
        """
        types_to_count = [note_type] if note_type else list(NoteType)
        count = 0

        for nt in types_to_count:
            type_dir = self.notes_dir / nt.value
            if type_dir.exists():
                count += len(list(type_dir.glob('*.md')))

        return count

    def clear(self, note_type: NoteType | None = None) -> int:
        """Delete all notes, optionally filtered by type.

        Args:
            note_type: Optional type filter (if None, deletes ALL notes)

        Returns:
            Number of notes deleted
        """
        types_to_clear = [note_type] if note_type else list(NoteType)
        deleted = 0

        for nt in types_to_clear:
            type_dir = self.notes_dir / nt.value
            if type_dir.exists():
                for path in type_dir.glob('*.md'):
                    try:
                        path.unlink()
                        deleted += 1
                    except Exception as e:
                        logger.warning(f'Error deleting {path}: {e}')

        logger.info(f'Cleared {deleted} notes')
        return deleted
