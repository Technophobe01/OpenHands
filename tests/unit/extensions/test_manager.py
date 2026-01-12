"""Tests for the ExtensionManager."""

import pytest

from openhands.extensions.base import Extension, ExtensionCategory
from openhands.extensions.context import RunContext
from openhands.extensions.manager import ExtensionManager, create_default_manager
from openhands.extensions.registry import ExtensionRegistry


class MockState:
    """Mock state for testing."""

    def __init__(self, session_id='test-session', iteration=0):
        self.session_id = session_id
        self.iteration_flag = MockIterationFlag(iteration)
        self.history = []
        self.extra_data = {}


class MockIterationFlag:
    """Mock iteration flag."""

    def __init__(self, current_value=0):
        self.current_value = current_value


class TestExtensionManager:
    """Tests for ExtensionManager."""

    def test_create_manager(self):
        """Test creating a manager."""
        manager = ExtensionManager()
        assert manager.registry is not None
        assert len(manager) == 0

    def test_create_manager_with_registry(self):
        """Test creating manager with existing registry."""
        registry = ExtensionRegistry()
        registry.register(Extension(name='test'))

        manager = ExtensionManager(registry=registry)
        assert len(manager) == 1

    def test_initialize_and_finalize(self):
        """Test initialization and finalization."""
        initialized = []
        finalized = []

        class TrackingExtension(Extension):
            def on_initialize(self, ctx):
                initialized.append(self.name)

            def on_finalize(self, ctx):
                finalized.append(self.name)

        manager = ExtensionManager()
        manager.registry.register(TrackingExtension(name='tracker'))

        state = MockState()
        manager.initialize('test-session', state)

        assert 'tracker' in initialized
        assert manager._initialized is True

        manager.finalize(state)
        assert 'tracker' in finalized
        assert manager._initialized is False

    def test_initialize_only_once(self):
        """Test that initialization only happens once."""
        init_count = [0]

        class CountingExtension(Extension):
            def on_initialize(self, ctx):
                init_count[0] += 1

        manager = ExtensionManager()
        manager.registry.register(CountingExtension(name='counter'))

        manager.initialize('test-session')
        manager.initialize('test-session')
        manager.initialize('test-session')

        assert init_count[0] == 1

    def test_process_input_messages(self):
        """Test processing input messages."""
        class AddMessageExtension(Extension):
            def on_input_messages(self, messages, ctx):
                return messages + [{'role': 'system', 'content': 'added'}]

        manager = ExtensionManager()
        manager.registry.register(AddMessageExtension(name='adder'))

        state = MockState()
        result = manager.process_input_messages([], state)

        assert len(result) == 1
        assert result[0]['content'] == 'added'

    def test_process_llm_output(self):
        """Test processing LLM output."""
        class TransformExtension(Extension):
            def on_llm_output(self, output, ctx):
                return output.upper()

        manager = ExtensionManager()
        manager.registry.register(TransformExtension(name='upper'))

        state = MockState()
        result = manager.process_llm_output('hello', state)

        assert result == 'HELLO'

    def test_process_tag(self):
        """Test processing tags."""
        class MockAction:
            def __init__(self, content):
                self.content = content

        class TagExtension(Extension):
            def on_tag(self, tag, content, ctx):
                if tag == 'custom':
                    return MockAction(content)
                return None

        manager = ExtensionManager()
        manager.registry.register(TagExtension(name='tagger'))

        state = MockState()

        result = manager.process_tag('custom', 'test content', state)
        assert result is not None
        assert result.content == 'test content'

        result = manager.process_tag('unknown', 'test', state)
        assert result is None

    def test_process_action(self):
        """Test processing actions."""
        class MockAction:
            pass

        class MockObservation:
            def __init__(self, content):
                self.content = content

        class ActionExtension(Extension):
            def on_action(self, action, ctx):
                if isinstance(action, MockAction):
                    return MockObservation('handled')
                return None

        manager = ExtensionManager()
        manager.registry.register(ActionExtension(name='handler'))

        state = MockState()

        result = manager.process_action(MockAction(), state)
        assert result is not None
        assert result.content == 'handled'

    def test_process_observation(self):
        """Test processing observations."""
        class MockAction:
            pass

        class MockObservation:
            def __init__(self, content):
                self.content = content

        class ObservationExtension(Extension):
            def on_observation(self, observation, action, ctx):
                observation.content = f'{observation.content}+transformed'
                return observation

        manager = ExtensionManager()
        manager.registry.register(ObservationExtension(name='transformer'))

        state = MockState()
        obs = MockObservation('original')
        action = MockAction()

        result = manager.process_observation(obs, action, state)
        assert result.content == 'original+transformed'

    def test_context_creation(self):
        """Test that context is created with correct values."""
        captured_ctx = [None]

        class ContextCapture(Extension):
            def on_input_messages(self, messages, ctx):
                captured_ctx[0] = ctx
                return messages

        manager = ExtensionManager()
        manager.registry.register(ContextCapture(name='capture'))

        state = MockState(session_id='my-session', iteration=42)
        state.history = [{'type': 'event1'}, {'type': 'event2'}]
        state.extra_data = {'key': 'value'}

        manager.process_input_messages([], state)

        ctx = captured_ctx[0]
        assert ctx is not None
        assert ctx.session_id == 'my-session'
        assert ctx.iteration == 42
        assert len(ctx.history) == 2
        assert ctx.extra_data == {'key': 'value'}

    def test_process_without_state(self):
        """Test that processing works without state."""
        class SimpleExtension(Extension):
            def on_input_messages(self, messages, ctx):
                return messages + [{'from': 'extension'}]

        manager = ExtensionManager()
        manager.registry.register(SimpleExtension(name='simple'))

        # Should not raise
        result = manager.process_input_messages([])
        assert len(result) == 1

    def test_repr(self):
        """Test manager repr."""
        manager = ExtensionManager()
        repr_str = repr(manager)
        assert 'ExtensionManager' in repr_str


class TestCreateDefaultManager:
    """Tests for create_default_manager."""

    def test_creates_manager(self):
        """Test that create_default_manager creates a manager."""
        manager = create_default_manager()
        assert isinstance(manager, ExtensionManager)
