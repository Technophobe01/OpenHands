"""Tests for the Extension base class and CompositeExtension."""

import pytest

from openhands.extensions.base import (
    CompositeExtension,
    Extension,
    ExtensionCategory,
    ExtensionPriority,
)
from openhands.extensions.context import RunContext


class TestExtension:
    """Tests for the Extension base class."""

    def test_default_values(self):
        """Test that Extension has sensible defaults."""
        ext = Extension()
        assert ext.name == ''
        assert ext.description == ''
        assert ext.category == ExtensionCategory.UTILITY
        assert ext.priority == ExtensionPriority.NORMAL
        assert ext.enabled is True

    def test_kwargs_initialization(self):
        """Test that Extension can be initialized with kwargs."""
        ext = Extension(name='test', enabled=False)
        assert ext.name == 'test'
        assert ext.enabled is False

    def test_on_input_messages_passthrough(self):
        """Test that default on_input_messages returns messages unchanged."""
        ext = Extension()
        ctx = RunContext(session_id='test')
        messages = [{'role': 'user', 'content': 'hello'}]
        result = ext.on_input_messages(messages, ctx)
        assert result == messages

    def test_on_llm_output_passthrough(self):
        """Test that default on_llm_output returns output unchanged."""
        ext = Extension()
        ctx = RunContext(session_id='test')
        output = 'test output'
        result = ext.on_llm_output(output, ctx)
        assert result == output

    def test_on_tag_returns_none(self):
        """Test that default on_tag returns None."""
        ext = Extension()
        ctx = RunContext(session_id='test')
        result = ext.on_tag('test_tag', 'content', ctx)
        assert result is None

    def test_on_action_returns_none(self):
        """Test that default on_action returns None."""
        ext = Extension()
        ctx = RunContext(session_id='test')
        # Mock action
        class MockAction:
            pass
        result = ext.on_action(MockAction(), ctx)
        assert result is None

    def test_repr(self):
        """Test Extension repr."""
        ext = Extension(name='test_ext')
        ext.category = ExtensionCategory.ACTION
        repr_str = repr(ext)
        assert 'test_ext' in repr_str
        assert 'action' in repr_str


class TestExtensionCategory:
    """Tests for ExtensionCategory enum."""

    def test_categories_exist(self):
        """Test that all expected categories exist."""
        assert ExtensionCategory.PERCEPTION.value == 'perception'
        assert ExtensionCategory.REASONING.value == 'reasoning'
        assert ExtensionCategory.ACTION.value == 'action'
        assert ExtensionCategory.UTILITY.value == 'utility'


class TestExtensionPriority:
    """Tests for ExtensionPriority enum."""

    def test_priorities_ordered(self):
        """Test that priorities are properly ordered."""
        assert ExtensionPriority.HIGHEST.value < ExtensionPriority.HIGH.value
        assert ExtensionPriority.HIGH.value < ExtensionPriority.NORMAL.value
        assert ExtensionPriority.NORMAL.value < ExtensionPriority.LOW.value
        assert ExtensionPriority.LOW.value < ExtensionPriority.LOWEST.value


class TestCompositeExtension:
    """Tests for the CompositeExtension class."""

    def test_empty_composite(self):
        """Test that empty composite works."""
        comp = CompositeExtension(name='empty')
        ctx = RunContext(session_id='test')

        messages = [{'role': 'user'}]
        assert comp.on_input_messages(messages, ctx) == messages

        assert comp.on_llm_output('test', ctx) == 'test'
        assert comp.on_tag('tag', 'content', ctx) is None

    def test_add_extension(self):
        """Test adding extensions to composite."""
        comp = CompositeExtension(name='comp')
        ext1 = Extension(name='ext1')
        ext2 = Extension(name='ext2')

        comp.add_extension(ext1)
        comp.add_extension(ext2)

        assert len(comp.get_extensions()) == 2
        assert comp.get_extensions()[0].name == 'ext1'

    def test_remove_extension(self):
        """Test removing extensions from composite."""
        ext1 = Extension(name='ext1')
        comp = CompositeExtension(name='comp', extensions=[ext1])

        assert comp.remove_extension('ext1') is True
        assert len(comp.get_extensions()) == 0
        assert comp.remove_extension('nonexistent') is False

    def test_priority_ordering(self):
        """Test that extensions are ordered by priority."""
        ext_low = Extension(name='low')
        ext_low.priority = ExtensionPriority.LOW

        ext_high = Extension(name='high')
        ext_high.priority = ExtensionPriority.HIGH

        comp = CompositeExtension(name='comp', extensions=[ext_low, ext_high])

        extensions = comp.get_extensions()
        assert extensions[0].name == 'high'
        assert extensions[1].name == 'low'

    def test_chained_on_input_messages(self):
        """Test that on_input_messages chains through all extensions."""
        class AddMessageExtension(Extension):
            def on_input_messages(self, messages, ctx):
                return messages + [{'role': 'system', 'content': self.name}]

        ext1 = AddMessageExtension(name='ext1')
        ext2 = AddMessageExtension(name='ext2')
        comp = CompositeExtension(name='comp', extensions=[ext1, ext2])

        ctx = RunContext(session_id='test')
        result = comp.on_input_messages([], ctx)

        assert len(result) == 2
        assert result[0]['content'] == 'ext1'
        assert result[1]['content'] == 'ext2'

    def test_chained_on_llm_output(self):
        """Test that on_llm_output chains through all extensions."""
        class AppendExtension(Extension):
            def on_llm_output(self, output, ctx):
                return output + f'_{self.name}'

        ext1 = AppendExtension(name='A')
        ext2 = AppendExtension(name='B')
        comp = CompositeExtension(name='comp', extensions=[ext1, ext2])

        ctx = RunContext(session_id='test')
        result = comp.on_llm_output('start', ctx)

        assert result == 'start_A_B'

    def test_on_tag_first_match_wins(self):
        """Test that first extension to return action wins."""
        class MockAction:
            def __init__(self, name):
                self.name = name

        class TagExtension(Extension):
            def on_tag(self, tag, content, ctx):
                if tag == self.name:
                    return MockAction(self.name)
                return None

        ext1 = TagExtension(name='tag1')
        ext2 = TagExtension(name='tag2')
        comp = CompositeExtension(name='comp', extensions=[ext1, ext2])

        ctx = RunContext(session_id='test')

        result = comp.on_tag('tag1', '', ctx)
        assert result.name == 'tag1'

        result = comp.on_tag('tag2', '', ctx)
        assert result.name == 'tag2'

        result = comp.on_tag('unknown', '', ctx)
        assert result is None

    def test_disabled_extension_skipped(self):
        """Test that disabled extensions are skipped."""
        class AppendExtension(Extension):
            def on_llm_output(self, output, ctx):
                return output + f'_{self.name}'

        ext1 = AppendExtension(name='A')
        ext2 = AppendExtension(name='B')
        ext2.enabled = False

        comp = CompositeExtension(name='comp', extensions=[ext1, ext2])

        ctx = RunContext(session_id='test')
        result = comp.on_llm_output('start', ctx)

        assert result == 'start_A'

    def test_initialize_calls_all(self):
        """Test that on_initialize is called on all extensions."""
        initialized = []

        class TrackingExtension(Extension):
            def on_initialize(self, ctx):
                initialized.append(self.name)

        ext1 = TrackingExtension(name='ext1')
        ext2 = TrackingExtension(name='ext2')
        comp = CompositeExtension(name='comp', extensions=[ext1, ext2])

        ctx = RunContext(session_id='test')
        comp.on_initialize(ctx)

        assert 'ext1' in initialized
        assert 'ext2' in initialized

    def test_finalize_calls_all(self):
        """Test that on_finalize is called on all extensions."""
        finalized = []

        class TrackingExtension(Extension):
            def on_finalize(self, ctx):
                finalized.append(self.name)

        ext1 = TrackingExtension(name='ext1')
        ext2 = TrackingExtension(name='ext2')
        comp = CompositeExtension(name='comp', extensions=[ext1, ext2])

        ctx = RunContext(session_id='test')
        comp.on_finalize(ctx)

        assert 'ext1' in finalized
        assert 'ext2' in finalized
