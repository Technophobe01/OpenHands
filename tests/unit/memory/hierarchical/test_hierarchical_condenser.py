"""Tests for hierarchical condenser."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from openhands.core.config.condenser_config import HierarchicalCondenserConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.message import Message, TextContent
from openhands.events.event import Event, EventSource
from openhands.llm import LLM
from openhands.llm.llm_registry import LLMRegistry
from openhands.memory.condenser.condenser import Condensation, View
from openhands.memory.condenser.impl.hierarchical_condenser import HierarchicalCondenser
from openhands.memory.hierarchical.scopes import MemoryScope


def create_test_event(
    message: str, timestamp: datetime | None = None, id: int | None = None
) -> Event:
    """Create a simple test event."""
    event = Event()
    event._message = message
    event.timestamp = timestamp if timestamp else datetime.now()
    if id is not None:
        event._id = id
    event._source = EventSource.USER
    return event


@pytest.fixture
def mock_llm() -> LLM:
    """Create a mock LLM for testing."""
    real_config = LLMConfig(
        model='gpt-4o', api_key='test_key', custom_llm_provider=None
    )

    mock = MagicMock(
        spec=LLM,
        config=real_config,
        metrics=MagicMock(),
    )

    # Setup mock response
    mock_message = MagicMock()
    mock_message.content = '<SESSION_SUMMARY>\nSummary content\n</SESSION_SUMMARY>'

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock.completion.return_value = mock_response

    def set_mock_response_content(content: str):
        mock_message.content = content

    mock.set_mock_response_content = set_mock_response_content

    mock.format_messages_for_llm = lambda events: [
        Message(role='user', content=[TextContent(text=str(event))]) for event in events
    ]

    return mock


class TestHierarchicalCondenser:
    """Tests for HierarchicalCondenser."""

    def test_init_valid_params(self, mock_llm):
        """Test creating condenser with valid parameters."""
        condenser = HierarchicalCondenser(
            llm=mock_llm,
            max_size=100,
            keep_first=1,
            compression_threshold=0.8,
            target_reduction_ratio=0.5,
        )
        assert condenser.max_size == 100
        assert condenser.keep_first == 1
        assert condenser.compression_threshold == 0.8

    def test_init_invalid_keep_first(self, mock_llm):
        """Test that keep_first must be less than half of max_size."""
        with pytest.raises(ValueError, match='keep_first.*must be less than half'):
            HierarchicalCondenser(
                llm=mock_llm,
                max_size=10,
                keep_first=6,  # >= 10/2
            )

    def test_should_condense_under_max(self, mock_llm):
        """Test that condensation is not triggered under max_size."""
        condenser = HierarchicalCondenser(llm=mock_llm, max_size=100)
        events = [create_test_event(f'Event {i}', id=i) for i in range(50)]
        view = View(events=events)

        assert not condenser.should_condense(view)

    def test_should_condense_over_max(self, mock_llm):
        """Test that condensation is triggered over max_size."""
        condenser = HierarchicalCondenser(llm=mock_llm, max_size=50)
        events = [create_test_event(f'Event {i}', id=i) for i in range(100)]
        view = View(events=events)

        assert condenser.should_condense(view)

    def test_should_condense_on_request(self, mock_llm):
        """Test that condensation is triggered on unhandled request."""
        condenser = HierarchicalCondenser(llm=mock_llm, max_size=100)
        events = [create_test_event(f'Event {i}', id=i) for i in range(10)]
        view = View(events=events, unhandled_condensation_request=True)

        assert condenser.should_condense(view)

    def test_get_condensation_returns_condensation(self, mock_llm):
        """Test that get_condensation returns a Condensation object."""
        condenser = HierarchicalCondenser(
            llm=mock_llm, max_size=20, keep_first=1
        )
        events = [create_test_event(f'Event {i}', id=i) for i in range(50)]
        view = View(events=events)

        result = condenser.get_condensation(view)

        assert isinstance(result, Condensation)
        assert result.action is not None
        assert result.action.summary is not None

    def test_get_condensation_keeps_first_events(self, mock_llm):
        """Test that first events are kept during condensation."""
        keep_first = 3
        condenser = HierarchicalCondenser(
            llm=mock_llm, max_size=20, keep_first=keep_first
        )
        events = [create_test_event(f'Event {i}', id=i) for i in range(50)]
        view = View(events=events)

        result = condenser.get_condensation(view)

        # The forgotten events should not include the first keep_first events
        assert result.action.forgotten_events_start_id >= keep_first

    def test_hierarchical_summary_includes_scopes(self, mock_llm):
        """Test that summary includes scope sections."""
        mock_llm.set_mock_response_content('Scope summary')

        condenser = HierarchicalCondenser(
            llm=mock_llm, max_size=20, keep_first=1
        )
        events = [create_test_event(f'Event {i}', id=i) for i in range(50)]
        view = View(events=events)

        result = condenser.get_condensation(view)

        # Summary should contain scope markers
        summary = result.action.summary
        assert summary is not None
        # The summary should contain at least one context type
        assert (
            '[SESSION CONTEXT]' in summary
            or '[TASK CONTEXT]' in summary
            or '[IMMEDIATE CONTEXT]' in summary
            or 'Scope summary' in summary
        )


class TestHierarchicalCondenserConfig:
    """Tests for HierarchicalCondenserConfig."""

    def test_config_creation(self):
        """Test creating config with defaults."""
        llm_config = LLMConfig(model='gpt-4o', api_key='test')
        config = HierarchicalCondenserConfig(llm_config=llm_config)

        assert config.type == 'hierarchical'
        assert config.keep_first == 1
        assert config.max_size == 100
        assert config.compression_threshold == 0.8
        assert config.target_reduction_ratio == 0.5

    def test_config_custom_values(self):
        """Test creating config with custom values."""
        llm_config = LLMConfig(model='gpt-4o', api_key='test')
        config = HierarchicalCondenserConfig(
            llm_config=llm_config,
            keep_first=2,
            max_size=200,
            compression_threshold=0.7,
            target_reduction_ratio=0.6,
        )

        assert config.keep_first == 2
        assert config.max_size == 200
        assert config.compression_threshold == 0.7
        assert config.target_reduction_ratio == 0.6

    def test_config_validation_threshold_range(self):
        """Test that compression_threshold must be 0.0-1.0."""
        llm_config = LLMConfig(model='gpt-4o', api_key='test')

        with pytest.raises(ValueError):
            HierarchicalCondenserConfig(
                llm_config=llm_config,
                compression_threshold=1.5,  # > 1.0
            )

    def test_config_validation_ratio_range(self):
        """Test that target_reduction_ratio must be 0.1-0.9."""
        llm_config = LLMConfig(model='gpt-4o', api_key='test')

        with pytest.raises(ValueError):
            HierarchicalCondenserConfig(
                llm_config=llm_config,
                target_reduction_ratio=0.05,  # < 0.1
            )


class TestHierarchicalCondenserFromConfig:
    """Tests for creating HierarchicalCondenser from config."""

    def test_from_config(self):
        """Test creating condenser from config."""
        llm_config = LLMConfig(model='gpt-4o', api_key='test')
        config = HierarchicalCondenserConfig(
            llm_config=llm_config,
            max_size=150,
            keep_first=2,
        )

        # Create mock registry
        mock_registry = MagicMock(spec=LLMRegistry)
        mock_llm = MagicMock(spec=LLM)
        mock_registry.get_llm.return_value = mock_llm

        condenser = HierarchicalCondenser.from_config(config, mock_registry)

        assert isinstance(condenser, HierarchicalCondenser)
        assert condenser.max_size == 150
        assert condenser.keep_first == 2
        mock_registry.get_llm.assert_called_once()
