"""Configuration for the note-taking system.

CCA F2: Configuration options for the note-taking agent that creates
persistent markdown notes for cross-session learning.
"""

from pydantic import BaseModel, ConfigDict, Field


class NoteConfig(BaseModel):
    """Configuration for the note-taking system (CCA F2).

    Controls how notes are created, stored, and retrieved for cross-session learning.
    """

    model_config = ConfigDict(extra='forbid')

    enabled: bool = Field(
        default=True,
        description='Whether note-taking is enabled',
    )

    notes_dir: str | None = Field(
        default=None,
        description='Directory to store notes. Defaults to ~/.openhands/notes/',
    )

    auto_recall_enabled: bool = Field(
        default=True,
        description='Whether to automatically recall relevant notes during context retrieval',
    )

    auto_note_enabled: bool = Field(
        default=False,
        description='Whether to automatically create notes from successful solutions',
    )

    auto_hindsight_enabled: bool = Field(
        default=True,
        description='Whether to automatically create hindsight notes from failures',
    )

    max_notes_per_recall: int = Field(
        default=5,
        ge=1,
        le=20,
        description='Maximum number of notes to include in recall responses',
    )

    auto_note_threshold: int = Field(
        default=5,
        ge=1,
        description='Minimum number of events before considering automatic note creation',
    )

    keyword_match_threshold: int = Field(
        default=1,
        ge=1,
        description='Minimum number of keyword matches required to recall a note',
    )
