"""Note-taking agent for the note-taking system.

CCA F2: Agent that creates persistent notes from problem-solving sessions
and retrieves relevant notes for cross-session learning.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

from openhands.core.logger import openhands_logger as logger
from openhands.memory.notes.note import Note, NoteType
from openhands.memory.notes.note_store import NoteStore

if TYPE_CHECKING:
    from openhands.events.event import Event
    from openhands.llm import LLM


# Prompt templates for LLM-based note creation
NOTE_CREATION_PROMPT = """Analyze the following problem-solving session and create a concise note.

Session context:
{context}

Events to analyze:
{events}

Create a note with the following structure:
1. Title: A concise, descriptive title (max 80 chars)
2. Keywords: 3-8 relevant keywords for retrieval (comma-separated)
3. Problem Context: Brief description of the problem
4. Solution: The key solution or insight

Respond in the following format:
<title>Your title here</title>
<keywords>keyword1, keyword2, keyword3</keywords>
<problem>Problem context here</problem>
<solution>Solution details here</solution>
"""

HINDSIGHT_NOTE_PROMPT = """Analyze the following failed attempt and create a hindsight note.

Session context:
{context}

Failed attempt details:
{events}

Error message:
{error}

Create a note documenting what went wrong and how to avoid it:
1. Title: A concise title describing the failure
2. Keywords: Relevant keywords for retrieval
3. Problem Context: What was being attempted
4. Error Analysis: What went wrong and why
5. Lessons: How to avoid this in the future

Respond in the following format:
<title>Your title here</title>
<keywords>keyword1, keyword2, keyword3</keywords>
<problem>What was being attempted</problem>
<error_analysis>What went wrong</error_analysis>
<lessons>How to avoid this</lessons>
"""

RECALL_RELEVANCE_PROMPT = """Given the current context, determine which of these notes are most relevant.

Current context:
{context}

Available notes:
{notes}

Rank the notes by relevance and explain why each is relevant.
Return the note IDs in order of relevance (most relevant first).

Format:
<relevant_notes>note_id1, note_id2, ...</relevant_notes>
<reasoning>Brief explanation of relevance</reasoning>
"""


class NoteTaker:
    """CCA F2: Note-taking agent for cross-session learning.

    Creates persistent notes from problem-solving sessions and retrieves
    relevant notes based on context for knowledge reuse.

    Features:
    - Automatic note creation from successful solutions
    - Hindsight notes from failures
    - Keyword-based retrieval
    - LLM-powered relevance ranking
    """

    def __init__(
        self,
        note_store: NoteStore | None = None,
        llm: LLM | None = None,
        session_id: str = '',
        auto_note_threshold: int = 5,
    ):
        """Initialize the note-taker.

        Args:
            note_store: Storage backend for notes (uses default if None)
            llm: LLM for note creation/analysis (optional, enables smart features)
            session_id: Current session ID for tracking
            auto_note_threshold: Min events before considering auto-note creation
        """
        self.note_store = note_store or NoteStore()
        self.llm = llm
        self.session_id = session_id
        self.auto_note_threshold = auto_note_threshold

        # Track events for potential note creation
        self._pending_events: list[Event] = []
        self._context: str = ''

    def set_context(self, context: str) -> None:
        """Set the current session context for note creation."""
        self._context = context

    def add_event(self, event: Event) -> None:
        """Add an event to the pending queue for potential note creation."""
        self._pending_events.append(event)

    def clear_pending_events(self) -> None:
        """Clear pending events (e.g., after note creation)."""
        self._pending_events.clear()

    def create_note(
        self,
        title: str,
        content: str,
        note_type: NoteType = NoteType.SOLUTION,
        keywords: list[str] | None = None,
        problem_context: str = '',
        solution: str = '',
        related_files: list[str] | None = None,
        error_message: str = '',
        source_event_ids: list[int] | None = None,
    ) -> Note:
        """Create and save a new note.

        Args:
            title: Note title
            content: Main content
            note_type: Type of note
            keywords: Keywords for retrieval
            problem_context: Optional problem description
            solution: Optional solution description
            related_files: Optional list of related file paths
            error_message: Optional error message (for failure notes)
            source_event_ids: Optional event IDs that led to this note

        Returns:
            The created and saved Note
        """
        note = Note(
            title=title,
            content=content,
            note_type=note_type,
            keywords=keywords or [],
            problem_context=problem_context,
            solution=solution,
            related_files=related_files or [],
            error_message=error_message,
            source_event_ids=source_event_ids or [],
            session_id=self.session_id,
        )

        self.note_store.save(note)
        logger.info(f'Created note: {note.title} ({note.note_id})')
        return note

    def create_note_from_llm(
        self,
        events: list[Event] | None = None,
        note_type: NoteType = NoteType.SOLUTION,
        related_files: list[str] | None = None,
    ) -> Note | None:
        """Create a note using LLM analysis of events.

        Args:
            events: Events to analyze (uses pending events if None)
            note_type: Type of note to create
            related_files: Optional related file paths

        Returns:
            The created Note, or None if LLM is not available
        """
        if self.llm is None:
            logger.warning('LLM not available for note creation')
            return None

        events = events or self._pending_events
        if not events:
            logger.debug('No events to create note from')
            return None

        # Format events for prompt
        events_text = '\n'.join(str(e) for e in events[-20:])  # Last 20 events
        event_ids = [e.id for e in events if e.id >= 0]

        prompt = NOTE_CREATION_PROMPT.format(
            context=self._context,
            events=events_text,
        )

        try:
            response = self._call_llm(prompt)
            parsed = self._parse_note_response(response)

            if parsed:
                note = self.create_note(
                    title=parsed['title'],
                    content=parsed.get('solution', ''),
                    note_type=note_type,
                    keywords=parsed.get('keywords', []),
                    problem_context=parsed.get('problem', ''),
                    solution=parsed.get('solution', ''),
                    related_files=related_files or [],
                    source_event_ids=event_ids,
                )
                self.clear_pending_events()
                return note
        except Exception as e:
            logger.error(f'Error creating note from LLM: {e}')

        return None

    def create_hindsight_note(
        self,
        error_message: str,
        events: list[Event] | None = None,
        related_files: list[str] | None = None,
    ) -> Note | None:
        """Create a hindsight note from a failed attempt.

        CCA F2: Hindsight notes document failures for learning.

        Args:
            error_message: The error that occurred
            events: Events leading to the failure
            related_files: Files involved in the failure

        Returns:
            The created Note, or None if LLM not available
        """
        events = events or self._pending_events

        if self.llm is None:
            # Create simple hindsight note without LLM
            return self.create_note(
                title=f'Failed attempt: {error_message[:50]}',
                content=f'Error: {error_message}',
                note_type=NoteType.FAILURE,
                keywords=['error', 'failure', 'hindsight'],
                error_message=error_message,
                related_files=related_files or [],
                source_event_ids=[e.id for e in events if e.id >= 0],
            )

        # Use LLM for better analysis
        events_text = '\n'.join(str(e) for e in events[-20:])
        event_ids = [e.id for e in events if e.id >= 0]

        prompt = HINDSIGHT_NOTE_PROMPT.format(
            context=self._context,
            events=events_text,
            error=error_message,
        )

        try:
            response = self._call_llm(prompt)
            parsed = self._parse_hindsight_response(response)

            if parsed:
                content = f"## Error Analysis\n\n{parsed.get('error_analysis', '')}\n\n"
                content += f"## Lessons Learned\n\n{parsed.get('lessons', '')}"

                note = self.create_note(
                    title=parsed['title'],
                    content=content,
                    note_type=NoteType.FAILURE,
                    keywords=parsed.get('keywords', []),
                    problem_context=parsed.get('problem', ''),
                    error_message=error_message,
                    related_files=related_files or [],
                    source_event_ids=event_ids,
                )
                self.clear_pending_events()
                return note
        except Exception as e:
            logger.error(f'Error creating hindsight note: {e}')

        return None

    def recall_notes(
        self,
        query: str,
        note_type: NoteType | None = None,
        limit: int = 5,
        use_llm_ranking: bool = False,
    ) -> list[Note]:
        """Retrieve relevant notes based on query.

        Args:
            query: The query to match against keywords
            note_type: Optional type filter
            limit: Maximum notes to return
            use_llm_ranking: Use LLM to re-rank results

        Returns:
            List of relevant notes
        """
        # Get keyword-matched notes
        matches = self.note_store.search_by_keywords(query, note_type, limit=limit * 2)
        notes = [note for note, _ in matches]

        if not notes:
            return []

        if use_llm_ranking and self.llm and len(notes) > 1:
            # Re-rank with LLM
            notes = self._llm_rank_notes(query, notes)

        return notes[:limit]

    def recall_for_context(
        self,
        context: str,
        related_files: list[str] | None = None,
        limit: int = 5,
    ) -> list[Note]:
        """Retrieve notes relevant to the current context.

        Combines keyword matching with file-based retrieval.

        Args:
            context: Current context/query
            related_files: Files being worked on
            limit: Maximum notes to return

        Returns:
            List of relevant notes
        """
        all_notes: dict[str, Note] = {}

        # Get keyword matches
        keyword_matches = self.note_store.search_by_keywords(context, limit=limit)
        for note, _ in keyword_matches:
            all_notes[note.note_id] = note

        # Get file-related notes
        if related_files:
            file_notes = self.note_store.get_related_notes(related_files, limit=limit)
            for note in file_notes:
                if note.note_id not in all_notes:
                    all_notes[note.note_id] = note

        # Sort by creation time and limit
        result = sorted(
            all_notes.values(),
            key=lambda n: n.created_at,
            reverse=True,
        )
        return result[:limit]

    def recall_failure_notes(
        self,
        error_pattern: str | None = None,
        limit: int = 5,
    ) -> list[Note]:
        """Retrieve failure (hindsight) notes.

        Args:
            error_pattern: Optional error message pattern to match
            limit: Maximum notes to return

        Returns:
            List of failure notes
        """
        return self.note_store.get_failure_notes(error_pattern, limit)

    def should_create_note(self) -> bool:
        """Determine if a note should be created from pending events.

        Heuristic-based decision for automatic note creation.
        """
        if len(self._pending_events) < self.auto_note_threshold:
            return False

        # Look for indicators of a completed solution
        # (This is a simple heuristic - can be enhanced with LLM)
        events_text = ' '.join(str(e) for e in self._pending_events[-10:])

        solution_indicators = [
            'successfully',
            'completed',
            'fixed',
            'resolved',
            'works',
            'passed',
        ]

        return any(indicator in events_text.lower() for indicator in solution_indicators)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        if self.llm is None:
            raise ValueError('LLM not configured')

        from openhands.core.message import Message, TextContent

        messages = [Message(role='user', content=[TextContent(text=prompt)])]
        response = self.llm.completion(messages=messages)

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content or ''
        return ''

    def _llm_rank_notes(self, query: str, notes: list[Note]) -> list[Note]:
        """Use LLM to rank notes by relevance."""
        if not notes or self.llm is None:
            return notes

        notes_text = '\n\n'.join(
            f'[{note.note_id}] {note.title}\nKeywords: {", ".join(note.keywords)}'
            for note in notes
        )

        prompt = RECALL_RELEVANCE_PROMPT.format(
            context=query,
            notes=notes_text,
        )

        try:
            response = self._call_llm(prompt)
            # Parse ranked note IDs
            match = re.search(r'<relevant_notes>(.*?)</relevant_notes>', response, re.DOTALL)
            if match:
                ranked_ids = [id.strip() for id in match.group(1).split(',')]
                note_map = {n.note_id: n for n in notes}
                ranked_notes = [note_map[id] for id in ranked_ids if id in note_map]
                # Add any notes not in the ranking
                for note in notes:
                    if note not in ranked_notes:
                        ranked_notes.append(note)
                return ranked_notes
        except Exception as e:
            logger.warning(f'Error ranking notes with LLM: {e}')

        return notes

    def _parse_note_response(self, response: str) -> dict | None:
        """Parse LLM response for note creation."""
        result = {}

        patterns = {
            'title': r'<title>(.*?)</title>',
            'keywords': r'<keywords>(.*?)</keywords>',
            'problem': r'<problem>(.*?)</problem>',
            'solution': r'<solution>(.*?)</solution>',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL)
            if match:
                value = match.group(1).strip()
                if key == 'keywords':
                    result[key] = [k.strip() for k in value.split(',')]
                else:
                    result[key] = value

        if 'title' not in result:
            return None

        return result

    def _parse_hindsight_response(self, response: str) -> dict | None:
        """Parse LLM response for hindsight note creation."""
        result = {}

        patterns = {
            'title': r'<title>(.*?)</title>',
            'keywords': r'<keywords>(.*?)</keywords>',
            'problem': r'<problem>(.*?)</problem>',
            'error_analysis': r'<error_analysis>(.*?)</error_analysis>',
            'lessons': r'<lessons>(.*?)</lessons>',
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL)
            if match:
                value = match.group(1).strip()
                if key == 'keywords':
                    result[key] = [k.strip() for k in value.split(',')]
                else:
                    result[key] = value

        if 'title' not in result:
            return None

        return result

    def get_stats(self) -> dict:
        """Get statistics about stored notes."""
        return {
            'total': self.note_store.count(),
            'by_type': {
                nt.value: self.note_store.count(nt)
                for nt in NoteType
            },
            'pending_events': len(self._pending_events),
        }
