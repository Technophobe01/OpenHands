"""Run context for extensions.

CCA F3: Context object passed to extension callbacks containing
session state, configuration, and utilities.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openhands.controller.state.state import State
    from openhands.events.event import Event


@dataclass
class RunContext:
    """Context passed to extension callbacks.

    CCA F3: Provides extensions with access to session state, history,
    and utilities needed for their operation.

    Attributes:
        session_id: The current session identifier
        iteration: The current iteration/step number
        state: The current agent state (may be None in some contexts)
        history: List of events in the current session
        working_dir: The current working directory
        extra_data: Additional task-specific data
        metadata: Extension-specific metadata storage
    """

    session_id: str
    iteration: int = 0
    state: 'State | None' = None
    history: list['Event'] = field(default_factory=list)
    working_dir: str = ''
    extra_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_recent_events(self, count: int = 10) -> list['Event']:
        """Get the most recent events from history.

        Args:
            count: Maximum number of events to return

        Returns:
            List of most recent events
        """
        if not self.history:
            return []
        return self.history[-count:]

    def get_events_by_type(self, event_type: type) -> list['Event']:
        """Get all events of a specific type from history.

        Args:
            event_type: The event class to filter by

        Returns:
            List of events matching the type
        """
        return [e for e in self.history if isinstance(e, event_type)]

    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata that persists across extension callbacks.

        Args:
            key: The metadata key
            value: The value to store
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve stored metadata.

        Args:
            key: The metadata key
            default: Default value if key not found

        Returns:
            The stored value or default
        """
        return self.metadata.get(key, default)
