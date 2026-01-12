"""Tests for the ExtensionRegistry."""

import pytest

from openhands.extensions.base import (
    Extension,
    ExtensionCategory,
    ExtensionPriority,
)
from openhands.extensions.context import RunContext
from openhands.extensions.registry import (
    ExtensionRegistry,
    extension,
    get_global_registry,
    register_extension,
)


class TestExtensionRegistry:
    """Tests for ExtensionRegistry."""

    def test_register_extension(self):
        """Test registering an extension."""
        registry = ExtensionRegistry()
        ext = Extension(name='test_ext')

        registry.register(ext)

        assert 'test_ext' in registry
        assert len(registry) == 1

    def test_register_duplicate_raises(self):
        """Test that registering duplicate name raises ValueError."""
        registry = ExtensionRegistry()
        ext1 = Extension(name='test_ext')
        ext2 = Extension(name='test_ext')

        registry.register(ext1)

        with pytest.raises(ValueError):
            registry.register(ext2)

    def test_unregister_extension(self):
        """Test unregistering an extension."""
        registry = ExtensionRegistry()
        ext = Extension(name='test_ext')
        registry.register(ext)

        assert registry.unregister('test_ext') is True
        assert 'test_ext' not in registry
        assert len(registry) == 0

    def test_unregister_nonexistent(self):
        """Test unregistering nonexistent extension returns False."""
        registry = ExtensionRegistry()
        assert registry.unregister('nonexistent') is False

    def test_get_extension(self):
        """Test getting an extension by name."""
        registry = ExtensionRegistry()
        ext = Extension(name='test_ext')
        registry.register(ext)

        assert registry.get('test_ext') is ext
        assert registry.get('nonexistent') is None

    def test_get_by_category(self):
        """Test getting extensions by category."""
        registry = ExtensionRegistry()

        action_ext = Extension(name='action')
        action_ext.category = ExtensionCategory.ACTION

        perception_ext = Extension(name='perception')
        perception_ext.category = ExtensionCategory.PERCEPTION

        registry.register(action_ext)
        registry.register(perception_ext)

        action_exts = registry.get_by_category(ExtensionCategory.ACTION)
        assert len(action_exts) == 1
        assert action_exts[0].name == 'action'

        perception_exts = registry.get_by_category(ExtensionCategory.PERCEPTION)
        assert len(perception_exts) == 1
        assert perception_exts[0].name == 'perception'

    def test_get_all(self):
        """Test getting all extensions."""
        registry = ExtensionRegistry()

        ext1 = Extension(name='ext1')
        ext2 = Extension(name='ext2')

        registry.register(ext1)
        registry.register(ext2)

        all_exts = registry.get_all()
        assert len(all_exts) == 2

    def test_get_enabled(self):
        """Test getting only enabled extensions."""
        registry = ExtensionRegistry()

        ext1 = Extension(name='enabled')
        ext2 = Extension(name='disabled')
        ext2.enabled = False

        registry.register(ext1)
        registry.register(ext2)

        enabled = registry.get_enabled()
        assert len(enabled) == 1
        assert enabled[0].name == 'enabled'

    def test_enable_disable(self):
        """Test enabling and disabling extensions."""
        registry = ExtensionRegistry()
        ext = Extension(name='test')
        ext.enabled = False
        registry.register(ext)

        assert registry.enable('test') is True
        assert ext.enabled is True

        assert registry.disable('test') is True
        assert ext.enabled is False

        assert registry.enable('nonexistent') is False
        assert registry.disable('nonexistent') is False

    def test_priority_ordering(self):
        """Test that extensions are returned in priority order."""
        registry = ExtensionRegistry()

        low = Extension(name='low')
        low.priority = ExtensionPriority.LOW

        high = Extension(name='high')
        high.priority = ExtensionPriority.HIGH

        normal = Extension(name='normal')
        normal.priority = ExtensionPriority.NORMAL

        registry.register(low)
        registry.register(high)
        registry.register(normal)

        all_exts = registry.get_all()
        assert all_exts[0].name == 'high'
        assert all_exts[1].name == 'normal'
        assert all_exts[2].name == 'low'

    def test_initialize_all(self):
        """Test initializing all extensions."""
        registry = ExtensionRegistry()
        initialized = []

        class TrackingExtension(Extension):
            def on_initialize(self, ctx):
                initialized.append(self.name)

        ext1 = TrackingExtension(name='ext1')
        ext2 = TrackingExtension(name='ext2')

        registry.register(ext1)
        registry.register(ext2)

        ctx = RunContext(session_id='test')
        registry.initialize_all(ctx)

        assert 'ext1' in initialized
        assert 'ext2' in initialized

    def test_initialize_skips_disabled(self):
        """Test that initialization skips disabled extensions."""
        registry = ExtensionRegistry()
        initialized = []

        class TrackingExtension(Extension):
            def on_initialize(self, ctx):
                initialized.append(self.name)

        ext1 = TrackingExtension(name='ext1')
        ext2 = TrackingExtension(name='ext2')
        ext2.enabled = False

        registry.register(ext1)
        registry.register(ext2)

        ctx = RunContext(session_id='test')
        registry.initialize_all(ctx)

        assert 'ext1' in initialized
        assert 'ext2' not in initialized

    def test_finalize_all(self):
        """Test finalizing all extensions."""
        registry = ExtensionRegistry()
        finalized = []

        class TrackingExtension(Extension):
            def on_initialize(self, ctx):
                pass

            def on_finalize(self, ctx):
                finalized.append(self.name)

        ext1 = TrackingExtension(name='ext1')
        ext2 = TrackingExtension(name='ext2')

        registry.register(ext1)
        registry.register(ext2)

        ctx = RunContext(session_id='test')
        registry.initialize_all(ctx)
        registry.finalize_all(ctx)

        assert 'ext1' in finalized
        assert 'ext2' in finalized


class TestRegistryInvocations:
    """Tests for registry hook invocations."""

    def test_invoke_on_input_messages(self):
        """Test invoking on_input_messages."""
        registry = ExtensionRegistry()

        class AddMessageExtension(Extension):
            def on_input_messages(self, messages, ctx):
                return messages + [{'role': 'system', 'content': self.name}]

        registry.register(AddMessageExtension(name='ext1'))
        registry.register(AddMessageExtension(name='ext2'))

        ctx = RunContext(session_id='test')
        result = registry.invoke_on_input_messages([], ctx)

        assert len(result) == 2

    def test_invoke_on_llm_output(self):
        """Test invoking on_llm_output."""
        registry = ExtensionRegistry()

        class AppendExtension(Extension):
            def on_llm_output(self, output, ctx):
                return output + f'_{self.name}'

        registry.register(AppendExtension(name='A'))
        registry.register(AppendExtension(name='B'))

        ctx = RunContext(session_id='test')
        result = registry.invoke_on_llm_output('start', ctx)

        assert result == 'start_A_B'

    def test_invoke_on_tag(self):
        """Test invoking on_tag."""
        registry = ExtensionRegistry()

        class MockAction:
            def __init__(self, name):
                self.name = name

        class TagExtension(Extension):
            def on_tag(self, tag, content, ctx):
                if tag == 'my_tag':
                    return MockAction(self.name)
                return None

        registry.register(TagExtension(name='handler'))

        ctx = RunContext(session_id='test')

        result = registry.invoke_on_tag('my_tag', '', ctx)
        assert result is not None
        assert result.name == 'handler'

        result = registry.invoke_on_tag('unknown_tag', '', ctx)
        assert result is None

    def test_invoke_on_action(self):
        """Test invoking on_action."""
        registry = ExtensionRegistry()

        class MockAction:
            pass

        class MockObservation:
            def __init__(self, name):
                self.name = name

        class ActionExtension(Extension):
            def on_action(self, action, ctx):
                if isinstance(action, MockAction):
                    return MockObservation(self.name)
                return None

        registry.register(ActionExtension(name='handler'))

        ctx = RunContext(session_id='test')

        result = registry.invoke_on_action(MockAction(), ctx)
        assert result is not None
        assert result.name == 'handler'

    def test_invoke_handles_exceptions(self):
        """Test that invoke methods handle extension exceptions gracefully."""
        registry = ExtensionRegistry()

        class FailingExtension(Extension):
            def on_llm_output(self, output, ctx):
                raise ValueError('Test error')

        class WorkingExtension(Extension):
            def on_llm_output(self, output, ctx):
                return output + '_worked'

        registry.register(FailingExtension(name='failing'))
        registry.register(WorkingExtension(name='working'))

        ctx = RunContext(session_id='test')
        # Should not raise, should continue to next extension
        result = registry.invoke_on_llm_output('start', ctx)
        assert '_worked' in result


class TestExtensionDecorator:
    """Tests for the @extension decorator."""

    def test_decorator_sets_attributes(self):
        """Test that decorator sets extension attributes."""
        @extension(
            name='decorated',
            category=ExtensionCategory.ACTION,
            priority=ExtensionPriority.HIGH,
        )
        class DecoratedExtension(Extension):
            pass

        assert DecoratedExtension.name == 'decorated'
        assert DecoratedExtension.category == ExtensionCategory.ACTION
        assert DecoratedExtension.priority == ExtensionPriority.HIGH

    def test_decorator_default_name(self):
        """Test that decorator uses class name if name not provided."""
        @extension(category=ExtensionCategory.PERCEPTION)
        class MyCustomExtension(Extension):
            pass

        assert MyCustomExtension.name == 'MyCustomExtension'


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_global_registry(self):
        """Test getting the global registry."""
        registry1 = get_global_registry()
        registry2 = get_global_registry()

        # Should return same instance
        assert registry1 is registry2

    def test_repr(self):
        """Test registry repr."""
        registry = ExtensionRegistry()
        registry.register(Extension(name='test'))

        repr_str = repr(registry)
        assert 'ExtensionRegistry' in repr_str
        assert 'test' in repr_str
