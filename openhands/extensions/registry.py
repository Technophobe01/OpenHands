"""Extension registry for managing and orchestrating extensions.

CCA F3: Central registry for extension management, providing methods to
register, configure, and invoke extensions at appropriate pipeline points.
"""

from typing import TYPE_CHECKING, Callable

from openhands.core.logger import openhands_logger as logger
from openhands.extensions.base import (
    Extension,
    ExtensionCategory,
    ExtensionPriority,
)
from openhands.extensions.context import RunContext

if TYPE_CHECKING:
    from openhands.core.message import Message
    from openhands.events.action.action import Action
    from openhands.events.observation.observation import Observation


class ExtensionRegistry:
    """Central registry for managing extensions.

    CCA F3: The registry handles:
    - Extension registration and lookup
    - Priority-based ordering
    - Pipeline hook invocation
    - Extension lifecycle management

    Example:
        registry = ExtensionRegistry()
        registry.register(LoggingExtension())
        registry.register(CodeFormatterExtension())

        # Invoke hooks
        messages = registry.invoke_on_input_messages(messages, ctx)
        output = registry.invoke_on_llm_output(output, ctx)
        action = registry.invoke_on_tag("execute_bash", content, ctx)
    """

    def __init__(self):
        """Initialize an empty extension registry."""
        self._extensions: dict[str, Extension] = {}
        self._by_category: dict[ExtensionCategory, list[Extension]] = {
            cat: [] for cat in ExtensionCategory
        }
        self._initialized: set[str] = set()

    def register(self, extension: Extension) -> None:
        """Register an extension with the registry.

        Args:
            extension: The extension to register

        Raises:
            ValueError: If an extension with the same name is already registered
        """
        if extension.name in self._extensions:
            raise ValueError(
                f"Extension with name '{extension.name}' is already registered"
            )

        self._extensions[extension.name] = extension
        self._by_category[extension.category].append(extension)
        # Sort by priority
        self._by_category[extension.category].sort(key=lambda e: e.priority.value)

        logger.debug(f'Registered extension: {extension.name} ({extension.category.value})')

    def unregister(self, name: str) -> bool:
        """Unregister an extension by name.

        Args:
            name: The name of the extension to unregister

        Returns:
            True if the extension was found and removed
        """
        if name not in self._extensions:
            return False

        extension = self._extensions.pop(name)
        self._by_category[extension.category].remove(extension)
        self._initialized.discard(name)

        logger.debug(f'Unregistered extension: {name}')
        return True

    def get(self, name: str) -> Extension | None:
        """Get an extension by name.

        Args:
            name: The extension name

        Returns:
            The extension if found, None otherwise
        """
        return self._extensions.get(name)

    def get_by_category(self, category: ExtensionCategory) -> list[Extension]:
        """Get all extensions in a category.

        Args:
            category: The category to filter by

        Returns:
            List of extensions in the category, sorted by priority
        """
        return list(self._by_category[category])

    def get_all(self) -> list[Extension]:
        """Get all registered extensions.

        Returns:
            List of all extensions, sorted by priority
        """
        all_extensions = list(self._extensions.values())
        all_extensions.sort(key=lambda e: e.priority.value)
        return all_extensions

    def get_enabled(self) -> list[Extension]:
        """Get all enabled extensions.

        Returns:
            List of enabled extensions, sorted by priority
        """
        return [e for e in self.get_all() if e.enabled]

    def enable(self, name: str) -> bool:
        """Enable an extension by name.

        Args:
            name: The extension name

        Returns:
            True if the extension was found and enabled
        """
        ext = self.get(name)
        if ext:
            ext.enabled = True
            logger.debug(f'Enabled extension: {name}')
            return True
        return False

    def disable(self, name: str) -> bool:
        """Disable an extension by name.

        Args:
            name: The extension name

        Returns:
            True if the extension was found and disabled
        """
        ext = self.get(name)
        if ext:
            ext.enabled = False
            logger.debug(f'Disabled extension: {name}')
            return True
        return False

    def initialize_all(self, ctx: RunContext) -> None:
        """Initialize all registered extensions for a session.

        Args:
            ctx: The run context for initialization
        """
        for ext in self.get_enabled():
            if ext.name not in self._initialized:
                try:
                    ext.on_initialize(ctx)
                    self._initialized.add(ext.name)
                    logger.debug(f'Initialized extension: {ext.name}')
                except Exception as e:
                    logger.warning(f'Failed to initialize extension {ext.name}: {e}')

    def finalize_all(self, ctx: RunContext) -> None:
        """Finalize all registered extensions at session end.

        Args:
            ctx: The run context for finalization
        """
        for ext in self.get_enabled():
            if ext.name in self._initialized:
                try:
                    ext.on_finalize(ctx)
                    self._initialized.discard(ext.name)
                    logger.debug(f'Finalized extension: {ext.name}')
                except Exception as e:
                    logger.warning(f'Failed to finalize extension {ext.name}: {e}')

    # Pipeline hook invocation methods

    def invoke_on_input_messages(
        self, messages: list['Message'], ctx: RunContext
    ) -> list['Message']:
        """Invoke on_input_messages on all enabled extensions.

        Extensions are called in priority order, each receiving the
        output of the previous extension.

        Args:
            messages: The initial messages
            ctx: The run context

        Returns:
            The transformed messages
        """
        result = messages
        for ext in self.get_enabled():
            try:
                result = ext.on_input_messages(result, ctx)
            except Exception as e:
                logger.warning(f'Extension {ext.name} failed in on_input_messages: {e}')
        return result

    def invoke_on_llm_output(self, output: str, ctx: RunContext) -> str:
        """Invoke on_llm_output on all enabled extensions.

        Extensions are called in priority order, each receiving the
        output of the previous extension.

        Args:
            output: The initial LLM output
            ctx: The run context

        Returns:
            The transformed output
        """
        result = output
        for ext in self.get_enabled():
            try:
                result = ext.on_llm_output(result, ctx)
            except Exception as e:
                logger.warning(f'Extension {ext.name} failed in on_llm_output: {e}')
        return result

    def invoke_on_tag(
        self, tag: str, content: str, ctx: RunContext
    ) -> 'Action | None':
        """Invoke on_tag on enabled extensions until one handles it.

        Extensions are called in priority order. The first extension
        to return an Action handles the tag.

        Args:
            tag: The XML tag name
            content: The tag content
            ctx: The run context

        Returns:
            An Action if an extension handles the tag, None otherwise
        """
        for ext in self.get_enabled():
            try:
                action = ext.on_tag(tag, content, ctx)
                if action is not None:
                    logger.debug(f'Extension {ext.name} handled tag: {tag}')
                    return action
            except Exception as e:
                logger.warning(f'Extension {ext.name} failed in on_tag: {e}')
        return None

    def invoke_on_action(
        self, action: 'Action', ctx: RunContext
    ) -> 'Observation | None':
        """Invoke on_action on enabled extensions until one handles it.

        Extensions are called in priority order. The first extension
        to return an Observation handles the action.

        Args:
            action: The action to execute
            ctx: The run context

        Returns:
            An Observation if an extension handles the action, None otherwise
        """
        for ext in self.get_enabled():
            try:
                observation = ext.on_action(action, ctx)
                if observation is not None:
                    logger.debug(f'Extension {ext.name} handled action: {type(action).__name__}')
                    return observation
            except Exception as e:
                logger.warning(f'Extension {ext.name} failed in on_action: {e}')
        return None

    def invoke_on_observation(
        self, observation: 'Observation', action: 'Action', ctx: RunContext
    ) -> 'Observation':
        """Invoke on_observation on all enabled extensions.

        Extensions are called in priority order, each receiving the
        output of the previous extension.

        Args:
            observation: The initial observation
            action: The action that produced the observation
            ctx: The run context

        Returns:
            The transformed observation
        """
        result = observation
        for ext in self.get_enabled():
            try:
                result = ext.on_observation(result, action, ctx)
            except Exception as e:
                logger.warning(f'Extension {ext.name} failed in on_observation: {e}')
        return result

    def __len__(self) -> int:
        return len(self._extensions)

    def __contains__(self, name: str) -> bool:
        return name in self._extensions

    def __repr__(self) -> str:
        return f'ExtensionRegistry(extensions={list(self._extensions.keys())})'


# Global registry instance for convenience
_global_registry: ExtensionRegistry | None = None


def get_global_registry() -> ExtensionRegistry:
    """Get the global extension registry.

    Creates the registry on first access.

    Returns:
        The global ExtensionRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ExtensionRegistry()
    return _global_registry


def register_extension(extension: Extension) -> None:
    """Register an extension with the global registry.

    Convenience function for quick registration.

    Args:
        extension: The extension to register
    """
    get_global_registry().register(extension)


def extension(
    name: str | None = None,
    category: ExtensionCategory = ExtensionCategory.UTILITY,
    priority: ExtensionPriority = ExtensionPriority.NORMAL,
    auto_register: bool = False,
) -> Callable[[type], type]:
    """Decorator to configure an extension class.

    Example:
        @extension(name="my_ext", category=ExtensionCategory.ACTION)
        class MyExtension(Extension):
            def on_action(self, action, ctx):
                ...

    Args:
        name: Extension name (defaults to class name)
        category: Extension category
        priority: Extension priority
        auto_register: Whether to auto-register with global registry

    Returns:
        Class decorator
    """
    def decorator(cls: type) -> type:
        cls.name = name or cls.__name__
        cls.category = category
        cls.priority = priority

        if auto_register:
            register_extension(cls())

        return cls

    return decorator
