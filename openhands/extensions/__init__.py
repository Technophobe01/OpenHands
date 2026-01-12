"""CCA F3: Extension system for modular agent behavior.

This module provides the extension framework that allows customization
of agent behavior at different points in the execution pipeline.

Extensions can hook into:
- Pre-LLM: Shape messages before sending to the LLM
- Post-LLM: Process raw LLM output
- Tag parsing: Convert XML tags to actions
- Action execution: Execute or transform actions
- Observation processing: Transform action results

Example:
    from openhands.extensions import Extension, ExtensionCategory, ExtensionRegistry

    class MyExtension(Extension):
        name = "my_extension"
        category = ExtensionCategory.UTILITY

        def on_input_messages(self, messages, ctx):
            # Add a system message
            return messages

    registry = ExtensionRegistry()
    registry.register(MyExtension())

    # In agent controller
    messages = registry.invoke_on_input_messages(messages, ctx)
"""

from openhands.extensions.base import (
    CompositeExtension,
    Extension,
    ExtensionCategory,
    ExtensionPriority,
)
from openhands.extensions.context import RunContext
from openhands.extensions.manager import (
    ExtensionManager,
    create_default_manager,
)
from openhands.extensions.registry import (
    ExtensionRegistry,
    extension,
    get_global_registry,
    register_extension,
)

__all__ = [
    # Base classes
    'Extension',
    'CompositeExtension',
    'ExtensionCategory',
    'ExtensionPriority',
    # Context
    'RunContext',
    # Manager
    'ExtensionManager',
    'create_default_manager',
    # Registry
    'ExtensionRegistry',
    'get_global_registry',
    'register_extension',
    'extension',
]
