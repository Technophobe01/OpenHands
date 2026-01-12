"""Extension manager for agent controller integration.

CCA F3: Manages extensions and provides integration points for the
agent controller pipeline.
"""

from typing import TYPE_CHECKING

from openhands.core.logger import openhands_logger as logger
from openhands.extensions.context import RunContext
from openhands.extensions.registry import ExtensionRegistry

if TYPE_CHECKING:
    from openhands.controller.state.state import State
    from openhands.core.message import Message
    from openhands.events.action.action import Action
    from openhands.events.observation.observation import Observation


class ExtensionManager:
    """Manager that bridges extensions with the agent controller.

    CCA F3: Provides methods to invoke extension hooks at appropriate
    points in the agent execution pipeline.

    Example:
        manager = ExtensionManager()
        manager.register(LoggingExtension())
        manager.register(CodeFormatterExtension())

        # In agent step
        messages = manager.process_input_messages(messages, state)
        llm_output = manager.process_llm_output(llm_output, state)
        action = manager.process_tag("execute_bash", content, state)
        observation = manager.process_action(action, state)
    """

    def __init__(self, registry: ExtensionRegistry | None = None):
        """Initialize the extension manager.

        Args:
            registry: Optional extension registry to use. Creates a new one if None.
        """
        self._registry = registry or ExtensionRegistry()
        self._session_id: str | None = None
        self._initialized = False

    @property
    def registry(self) -> ExtensionRegistry:
        """Get the extension registry."""
        return self._registry

    def _create_context(self, state: 'State | None' = None) -> RunContext:
        """Create a RunContext from the current state.

        Args:
            state: The current agent state

        Returns:
            A RunContext for extension callbacks
        """
        session_id = self._session_id or ''
        if state:
            session_id = state.session_id or session_id

        return RunContext(
            session_id=session_id,
            iteration=state.iteration_flag.current_value if state else 0,
            state=state,
            history=list(state.history) if state else [],
            extra_data=dict(state.extra_data) if state and state.extra_data else {},
        )

    def initialize(self, session_id: str, state: 'State | None' = None) -> None:
        """Initialize extensions for a session.

        Args:
            session_id: The session identifier
            state: Optional initial state
        """
        if self._initialized:
            return

        self._session_id = session_id
        ctx = self._create_context(state)
        self._registry.initialize_all(ctx)
        self._initialized = True
        logger.debug(f'Extension manager initialized for session {session_id}')

    def finalize(self, state: 'State | None' = None) -> None:
        """Finalize extensions at session end.

        Args:
            state: Optional final state
        """
        if not self._initialized:
            return

        ctx = self._create_context(state)
        self._registry.finalize_all(ctx)
        self._initialized = False
        logger.debug(f'Extension manager finalized for session {self._session_id}')

    def process_input_messages(
        self, messages: list['Message'], state: 'State | None' = None
    ) -> list['Message']:
        """Process messages before sending to LLM.

        CCA F3: Invokes on_input_messages on all enabled extensions.

        Args:
            messages: The messages to process
            state: The current agent state

        Returns:
            Processed messages
        """
        ctx = self._create_context(state)
        return self._registry.invoke_on_input_messages(messages, ctx)

    def process_llm_output(
        self, output: str, state: 'State | None' = None
    ) -> str:
        """Process LLM output before parsing.

        CCA F3: Invokes on_llm_output on all enabled extensions.

        Args:
            output: The raw LLM output
            state: The current agent state

        Returns:
            Processed output
        """
        ctx = self._create_context(state)
        return self._registry.invoke_on_llm_output(output, ctx)

    def process_tag(
        self, tag: str, content: str, state: 'State | None' = None
    ) -> 'Action | None':
        """Process an XML tag to potentially create an action.

        CCA F3: Invokes on_tag on enabled extensions.

        Args:
            tag: The XML tag name
            content: The tag content
            state: The current agent state

        Returns:
            An Action if an extension handles the tag, None otherwise
        """
        ctx = self._create_context(state)
        return self._registry.invoke_on_tag(tag, content, ctx)

    def process_action(
        self, action: 'Action', state: 'State | None' = None
    ) -> 'Observation | None':
        """Process an action for potential extension handling.

        CCA F3: Invokes on_action on enabled extensions.

        Args:
            action: The action to process
            state: The current agent state

        Returns:
            An Observation if an extension handles the action, None otherwise
        """
        ctx = self._create_context(state)
        return self._registry.invoke_on_action(action, ctx)

    def process_observation(
        self,
        observation: 'Observation',
        action: 'Action',
        state: 'State | None' = None,
    ) -> 'Observation':
        """Process an observation after action execution.

        CCA F3: Invokes on_observation on all enabled extensions.

        Args:
            observation: The observation to process
            action: The action that produced the observation
            state: The current agent state

        Returns:
            Processed observation
        """
        ctx = self._create_context(state)
        return self._registry.invoke_on_observation(observation, action, ctx)

    def __len__(self) -> int:
        return len(self._registry)

    def __repr__(self) -> str:
        return f'ExtensionManager(registry={self._registry!r}, initialized={self._initialized})'


# Convenience function to create a manager with common extensions
def create_default_manager() -> ExtensionManager:
    """Create an extension manager with default/built-in extensions.

    Returns:
        An ExtensionManager with common extensions registered
    """
    manager = ExtensionManager()
    # Note: Built-in extensions would be registered here once implemented
    # For now, return an empty manager
    return manager
