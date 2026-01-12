"""Base extension class for CCA F3 extension system.

CCA F3: Modular extensions with typed callbacks that allow customization
of agent behavior at different points in the execution pipeline.
"""

from abc import ABC
from enum import Enum
from typing import TYPE_CHECKING

from openhands.extensions.context import RunContext

if TYPE_CHECKING:
    from openhands.core.message import Message
    from openhands.events.action.action import Action
    from openhands.events.observation.observation import Observation


class ExtensionCategory(str, Enum):
    """Categories of extensions based on their primary function.

    CCA F3: Extensions are categorized to help organize and understand
    their role in the agent pipeline.
    """

    PERCEPTION = 'perception'
    """Extensions that process input and shape how the agent perceives context."""

    REASONING = 'reasoning'
    """Extensions that modify or enhance the agent's reasoning process."""

    ACTION = 'action'
    """Extensions that execute actions or transform action outputs."""

    UTILITY = 'utility'
    """General utility extensions that don't fit other categories."""


class ExtensionPriority(int, Enum):
    """Priority levels for extension execution order.

    Extensions with higher priority (lower number) execute first.
    """

    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


class Extension(ABC):
    """Base class for CCA F3 modular extensions.

    Extensions provide hooks into the agent execution pipeline, allowing
    customization of behavior at key points:

    - on_input_messages: Shape messages before LLM call
    - on_llm_output: Process LLM output before parsing
    - on_tag: Parse XML-style tags into actions
    - on_action: Execute or transform actions
    - on_observation: Process observations after action execution

    Example:
        class LoggingExtension(Extension):
            name = "logging"
            category = ExtensionCategory.UTILITY

            def on_input_messages(self, messages, ctx):
                logger.info(f"Sending {len(messages)} messages to LLM")
                return messages

            def on_action(self, action, ctx):
                logger.info(f"Executing action: {action}")
                return None  # Let other extensions handle execution
    """

    # Extension identification
    name: str = ''
    """Unique name for this extension."""

    description: str = ''
    """Human-readable description of what this extension does."""

    category: ExtensionCategory = ExtensionCategory.UTILITY
    """Category that best describes this extension's function."""

    priority: ExtensionPriority = ExtensionPriority.NORMAL
    """Execution priority (lower number = executed first)."""

    # Extension capabilities
    enabled: bool = True
    """Whether this extension is currently active."""

    def __init__(self, **kwargs):
        """Initialize the extension.

        Subclasses can override to accept configuration parameters.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def on_input_messages(
        self, messages: list['Message'], ctx: RunContext
    ) -> list['Message']:
        """Hook to shape messages before LLM call.

        CCA F3: Pre-LLM prompt shaping hook. Extensions can modify,
        filter, or augment the messages sent to the LLM.

        Args:
            messages: The messages about to be sent to the LLM
            ctx: The run context with session state

        Returns:
            Modified list of messages (must return messages)
        """
        return messages

    def on_llm_output(self, output: str, ctx: RunContext) -> str:
        """Hook to process LLM output before parsing.

        CCA F3: Post-LLM output processing hook. Extensions can clean up,
        transform, or augment the raw LLM response.

        Args:
            output: The raw output string from the LLM
            ctx: The run context with session state

        Returns:
            Modified output string (must return output)
        """
        return output

    def on_tag(
        self, tag: str, content: str, ctx: RunContext
    ) -> 'Action | None':
        """Hook to parse XML-style tags into actions.

        CCA F3: Tag parsing hook. Extensions can define custom XML tags
        that get converted into actions.

        Args:
            tag: The XML tag name (e.g., "execute_bash")
            content: The content inside the tag
            ctx: The run context with session state

        Returns:
            An Action if this extension handles the tag, None otherwise
        """
        return None

    def on_action(
        self, action: 'Action', ctx: RunContext
    ) -> 'Observation | None':
        """Hook to execute or transform actions.

        CCA F3: Action execution hook. Extensions can intercept actions
        to execute them, transform them, or add side effects.

        Args:
            action: The action to potentially handle
            ctx: The run context with session state

        Returns:
            An Observation if this extension handles the action, None otherwise.
            Returning None allows other extensions or the default handler to process.
        """
        return None

    def on_observation(
        self, observation: 'Observation', action: 'Action', ctx: RunContext
    ) -> 'Observation':
        """Hook to process observations after action execution.

        CCA F3: Post-action observation processing hook. Extensions can
        transform observations or add additional context.

        Args:
            observation: The observation from action execution
            action: The action that produced this observation
            ctx: The run context with session state

        Returns:
            Modified observation (must return observation)
        """
        return observation

    def on_initialize(self, ctx: RunContext) -> None:
        """Hook called when the extension is initialized for a session.

        Override to perform setup operations like loading state or
        connecting to external services.

        Args:
            ctx: The run context with session state
        """
        pass

    def on_finalize(self, ctx: RunContext) -> None:
        """Hook called when a session is ending.

        Override to perform cleanup operations like saving state or
        closing connections.

        Args:
            ctx: The run context with session state
        """
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(name={self.name!r}, category={self.category.value}, enabled={self.enabled})'


class CompositeExtension(Extension):
    """An extension that composes multiple extensions.

    CCA F3: Allows grouping related extensions together and treating
    them as a single unit.

    Example:
        code_quality = CompositeExtension(
            name="code_quality",
            extensions=[LintingExtension(), FormattingExtension()]
        )
    """

    def __init__(
        self,
        extensions: list[Extension] | None = None,
        **kwargs,
    ):
        """Initialize the composite extension.

        Args:
            extensions: List of extensions to compose
            **kwargs: Additional arguments passed to base Extension
        """
        super().__init__(**kwargs)
        self._extensions: list[Extension] = extensions or []
        # Sort by priority
        self._extensions.sort(key=lambda e: e.priority.value)

    def add_extension(self, extension: Extension) -> None:
        """Add an extension to the composition.

        Args:
            extension: The extension to add
        """
        self._extensions.append(extension)
        self._extensions.sort(key=lambda e: e.priority.value)

    def remove_extension(self, name: str) -> bool:
        """Remove an extension by name.

        Args:
            name: The name of the extension to remove

        Returns:
            True if extension was found and removed
        """
        for i, ext in enumerate(self._extensions):
            if ext.name == name:
                self._extensions.pop(i)
                return True
        return False

    def get_extensions(self) -> list[Extension]:
        """Get all composed extensions.

        Returns:
            List of extensions in priority order
        """
        return list(self._extensions)

    def on_input_messages(
        self, messages: list['Message'], ctx: RunContext
    ) -> list['Message']:
        """Chain on_input_messages through all composed extensions."""
        result = messages
        for ext in self._extensions:
            if ext.enabled:
                result = ext.on_input_messages(result, ctx)
        return result

    def on_llm_output(self, output: str, ctx: RunContext) -> str:
        """Chain on_llm_output through all composed extensions."""
        result = output
        for ext in self._extensions:
            if ext.enabled:
                result = ext.on_llm_output(result, ctx)
        return result

    def on_tag(
        self, tag: str, content: str, ctx: RunContext
    ) -> 'Action | None':
        """Try each composed extension until one handles the tag."""
        for ext in self._extensions:
            if ext.enabled:
                action = ext.on_tag(tag, content, ctx)
                if action is not None:
                    return action
        return None

    def on_action(
        self, action: 'Action', ctx: RunContext
    ) -> 'Observation | None':
        """Try each composed extension until one handles the action."""
        for ext in self._extensions:
            if ext.enabled:
                observation = ext.on_action(action, ctx)
                if observation is not None:
                    return observation
        return None

    def on_observation(
        self, observation: 'Observation', action: 'Action', ctx: RunContext
    ) -> 'Observation':
        """Chain on_observation through all composed extensions."""
        result = observation
        for ext in self._extensions:
            if ext.enabled:
                result = ext.on_observation(result, action, ctx)
        return result

    def on_initialize(self, ctx: RunContext) -> None:
        """Initialize all composed extensions."""
        for ext in self._extensions:
            if ext.enabled:
                ext.on_initialize(ctx)

    def on_finalize(self, ctx: RunContext) -> None:
        """Finalize all composed extensions."""
        for ext in self._extensions:
            if ext.enabled:
                ext.on_finalize(ctx)
