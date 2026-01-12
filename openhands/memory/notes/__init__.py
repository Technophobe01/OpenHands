"""Note-Taking System for cross-session learning.

Implementation of CCA (Confucius Code Agent) F2: Note-taking agent that
creates persistent markdown notes for knowledge capture and retrieval.

Reference: arXiv:2512.10398
"""

from openhands.memory.notes.note import Note, NoteType
from openhands.memory.notes.note_store import NoteStore
from openhands.memory.notes.note_taker import NoteTaker

__all__ = [
    'Note',
    'NoteType',
    'NoteStore',
    'NoteTaker',
]
