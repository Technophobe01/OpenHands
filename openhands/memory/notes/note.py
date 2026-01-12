"""Note model for the note-taking system.

CCA F2: Notes are markdown files with YAML frontmatter containing:
- Metadata (id, title, type, keywords, timestamps)
- Structured content (problem context, solution, related files)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import frontmatter


class NoteType(str, Enum):
    """Type of note for categorization."""

    SOLUTION = 'solution'  # A solution to a problem
    FAILURE = 'failure'  # A failed attempt (hindsight note)
    PATTERN = 'pattern'  # A recognized pattern or best practice
    ARCHITECTURE = 'architecture'  # Architectural decision or insight
    DEBUG = 'debug'  # Debugging insight
    REFERENCE = 'reference'  # Reference information

    @classmethod
    def from_string(cls, value: str) -> 'NoteType':
        """Create NoteType from string."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.REFERENCE


@dataclass
class Note:
    """A persistent note for cross-session knowledge.

    Notes use markdown format with YAML frontmatter for metadata.
    They can be retrieved based on keyword matching.
    """

    # Required fields
    title: str
    content: str

    # Metadata
    note_id: str = ''
    note_type: NoteType = NoteType.SOLUTION
    keywords: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime | None = None
    session_id: str = ''

    # Optional structured content
    problem_context: str = ''
    solution: str = ''
    related_files: list[str] = field(default_factory=list)
    error_message: str = ''  # For failure notes

    # Source tracking
    source_event_ids: list[int] = field(default_factory=list)

    def __post_init__(self):
        """Generate note_id if not provided."""
        if not self.note_id:
            # Generate ID from title hash and timestamp
            title_hash = hashlib.sha256(self.title.encode()).hexdigest()[:8]
            timestamp = int(self.created_at.timestamp())
            self.note_id = f'{self.note_type.value}_{title_hash}_{timestamp}'

    def match_keywords(self, query: str) -> list[str]:
        """Find keywords that match the query (case-insensitive).

        Returns list of matched keywords.
        """
        if not query:
            return []

        query_lower = query.lower()
        matched = []
        for keyword in self.keywords:
            if keyword.lower() in query_lower:
                matched.append(keyword)
        return matched

    def to_markdown(self) -> str:
        """Convert note to markdown with frontmatter."""
        metadata = {
            'id': self.note_id,
            'title': self.title,
            'type': self.note_type.value,
            'keywords': self.keywords,
            'created': self.created_at.isoformat(),
            'session_id': self.session_id,
        }

        if self.updated_at:
            metadata['updated'] = self.updated_at.isoformat()
        if self.related_files:
            metadata['related_files'] = self.related_files
        if self.source_event_ids:
            metadata['source_events'] = self.source_event_ids

        # Build content sections
        content_parts = []

        if self.problem_context:
            content_parts.append(f'## Problem Context\n\n{self.problem_context}')

        if self.solution:
            content_parts.append(f'## Solution\n\n{self.solution}')
        elif self.content:
            content_parts.append(self.content)

        if self.error_message:
            content_parts.append(f'## Error\n\n```\n{self.error_message}\n```')

        if self.related_files:
            files_list = '\n'.join(f'- `{f}`' for f in self.related_files)
            content_parts.append(f'## Related Files\n\n{files_list}')

        full_content = '\n\n'.join(content_parts)

        post = frontmatter.Post(full_content, **metadata)
        return frontmatter.dumps(post)

    @classmethod
    def from_markdown(cls, markdown_content: str) -> 'Note':
        """Parse note from markdown with frontmatter."""
        post = frontmatter.loads(markdown_content)
        metadata = post.metadata
        content = post.content

        # Parse metadata
        note_id = metadata.get('id', '')
        title = metadata.get('title', 'Untitled')
        note_type = NoteType.from_string(metadata.get('type', 'solution'))
        keywords = metadata.get('keywords', [])
        session_id = metadata.get('session_id', '')
        related_files = metadata.get('related_files', [])
        source_event_ids = metadata.get('source_events', [])

        # Parse dates
        created_str = metadata.get('created')
        created_at = (
            datetime.fromisoformat(created_str) if created_str else datetime.now()
        )

        updated_str = metadata.get('updated')
        updated_at = datetime.fromisoformat(updated_str) if updated_str else None

        # Extract structured content from markdown
        problem_context = cls._extract_section(content, 'Problem Context')
        solution = cls._extract_section(content, 'Solution')
        error_message = cls._extract_code_block(
            cls._extract_section(content, 'Error')
        )

        return cls(
            note_id=note_id,
            title=title,
            content=content,
            note_type=note_type,
            keywords=keywords,
            created_at=created_at,
            updated_at=updated_at,
            session_id=session_id,
            problem_context=problem_context,
            solution=solution,
            related_files=related_files,
            error_message=error_message,
            source_event_ids=source_event_ids,
        )

    @classmethod
    def from_file(cls, file_path: Path | str) -> 'Note':
        """Load note from a file."""
        path = Path(file_path)
        content = path.read_text(encoding='utf-8')
        note = cls.from_markdown(content)
        return note

    def save(self, file_path: Path | str) -> None:
        """Save note to a file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding='utf-8')

    @staticmethod
    def _extract_section(content: str, section_name: str) -> str:
        """Extract content under a markdown section header."""
        pattern = rf'^##\s+{re.escape(section_name)}\s*\n(.*?)(?=^##\s|\Z)'
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ''

    @staticmethod
    def _extract_code_block(content: str) -> str:
        """Extract content from a code block."""
        pattern = r'```(?:\w*\n)?(.*?)```'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content.strip()

    def __str__(self) -> str:
        """String representation for display."""
        return f'Note({self.note_type.value}): {self.title}'

    def __repr__(self) -> str:
        return f'Note(id={self.note_id!r}, title={self.title!r}, type={self.note_type.value})'
