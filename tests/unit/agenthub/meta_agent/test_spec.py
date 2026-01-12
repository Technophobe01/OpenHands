"""Tests for meta-agent specification types."""

import pytest

from openhands.agenthub.meta_agent.spec import (
    AgentSpec,
    EvaluationMetrics,
    ImprovementSuggestion,
    SynthesisResult,
    TaskAnalysis,
    TaskComplexity,
    TaskDomain,
)


class TestTaskComplexity:
    """Tests for TaskComplexity enum."""

    def test_complexity_values(self):
        """Test complexity enum values."""
        assert TaskComplexity.SIMPLE.value == 'simple'
        assert TaskComplexity.MODERATE.value == 'moderate'
        assert TaskComplexity.COMPLEX.value == 'complex'
        assert TaskComplexity.EXPERT.value == 'expert'

    def test_complexity_is_string_enum(self):
        """Test that complexity is a string enum."""
        assert isinstance(TaskComplexity.SIMPLE, str)
        assert TaskComplexity.SIMPLE == 'simple'


class TestTaskDomain:
    """Tests for TaskDomain enum."""

    def test_domain_values(self):
        """Test domain enum values."""
        assert TaskDomain.CODING.value == 'coding'
        assert TaskDomain.BROWSING.value == 'browsing'
        assert TaskDomain.ANALYSIS.value == 'analysis'
        assert TaskDomain.SYSTEM.value == 'system'
        assert TaskDomain.DATA.value == 'data'
        assert TaskDomain.MIXED.value == 'mixed'


class TestTaskAnalysis:
    """Tests for TaskAnalysis dataclass."""

    def test_default_values(self):
        """Test default values."""
        analysis = TaskAnalysis(description='Test task')

        assert analysis.description == 'Test task'
        assert analysis.complexity == TaskComplexity.MODERATE
        assert analysis.domains == [TaskDomain.CODING]
        assert analysis.estimated_steps == 10
        assert analysis.requires_file_editing is True
        assert analysis.requires_browsing is False
        assert analysis.requires_jupyter is False
        assert analysis.requires_planning is False
        assert analysis.keywords == []
        assert analysis.related_files == []
        assert analysis.constraints == {}

    def test_custom_values(self):
        """Test custom values."""
        analysis = TaskAnalysis(
            description='Complex web task',
            complexity=TaskComplexity.COMPLEX,
            domains=[TaskDomain.CODING, TaskDomain.BROWSING],
            estimated_steps=50,
            requires_file_editing=True,
            requires_browsing=True,
            requires_planning=True,
            keywords=['api', 'authentication'],
        )

        assert analysis.complexity == TaskComplexity.COMPLEX
        assert TaskDomain.BROWSING in analysis.domains
        assert analysis.requires_browsing is True


class TestAgentSpec:
    """Tests for AgentSpec dataclass."""

    def test_default_values(self):
        """Test default values."""
        spec = AgentSpec()

        assert spec.agent_type == 'CodeActAgent'
        assert spec.enable_browsing is True
        assert spec.enable_editor is True
        assert spec.enable_jupyter is True
        assert spec.enable_cmd is True
        assert spec.enable_think is True
        assert spec.enable_plan_mode is False
        assert spec.enable_stuck_detection is True
        assert spec.enable_history_truncation is True
        assert spec.llm_model is None
        assert spec.temperature is None
        assert spec.condenser_type == 'observation_masking'
        assert spec.condenser_keep_first == 1
        assert spec.condenser_max_size == 100
        assert spec.extensions == []
        assert spec.disabled_microagents == []

    def test_to_agent_config_dict(self):
        """Test conversion to agent config dict."""
        spec = AgentSpec(
            enable_browsing=True,
            enable_editor=True,
            enable_jupyter=False,
            enable_plan_mode=True,
            disabled_microagents=['agent1'],
        )

        config = spec.to_agent_config_dict()

        assert config['enable_browsing'] is True
        assert config['enable_editor'] is True
        assert config['enable_jupyter'] is False
        assert config['enable_plan_mode'] is True
        assert config['disabled_microagents'] == ['agent1']

    def test_to_agent_config_dict_with_prompt(self):
        """Test conversion includes system prompt when set."""
        spec = AgentSpec(system_prompt_filename='custom_prompt.md')

        config = spec.to_agent_config_dict()

        assert config['system_prompt_filename'] == 'custom_prompt.md'


class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""

    def test_default_values(self):
        """Test default values."""
        metrics = EvaluationMetrics()

        assert metrics.success_rate == 0.0
        assert metrics.avg_steps == 0.0
        assert metrics.avg_cost == 0.0
        assert metrics.avg_time_seconds == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.stuck_rate == 0.0
        assert metrics.tasks_evaluated == 0
        assert metrics.task_results == []

    def test_is_better_than_empty(self):
        """Test comparison with empty metrics."""
        metrics1 = EvaluationMetrics(tasks_evaluated=1, success_rate=0.8)
        metrics2 = EvaluationMetrics()

        assert metrics1.is_better_than(metrics2) is True
        assert metrics2.is_better_than(metrics1) is False

    def test_is_better_than_success_rate(self):
        """Test comparison based on success rate."""
        metrics1 = EvaluationMetrics(
            tasks_evaluated=10,
            success_rate=0.9,
            error_rate=0.1,
        )
        metrics2 = EvaluationMetrics(
            tasks_evaluated=10,
            success_rate=0.7,
            error_rate=0.2,
        )

        assert metrics1.is_better_than(metrics2) is True
        assert metrics2.is_better_than(metrics1) is False

    def test_is_better_than_considers_cost(self):
        """Test that comparison considers cost."""
        metrics1 = EvaluationMetrics(
            tasks_evaluated=10,
            success_rate=0.8,
            avg_cost=0.01,
        )
        metrics2 = EvaluationMetrics(
            tasks_evaluated=10,
            success_rate=0.8,
            avg_cost=0.1,  # Higher cost
        )

        assert metrics1.is_better_than(metrics2) is True

    def test_is_better_than_considers_error_rate(self):
        """Test that comparison considers error rate."""
        metrics1 = EvaluationMetrics(
            tasks_evaluated=10,
            success_rate=0.8,
            error_rate=0.1,
        )
        metrics2 = EvaluationMetrics(
            tasks_evaluated=10,
            success_rate=0.8,
            error_rate=0.4,  # Higher error rate
        )

        assert metrics1.is_better_than(metrics2) is True


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        spec = AgentSpec()
        analysis = TaskAnalysis(description='Test')

        result = SynthesisResult(spec=spec, task_analysis=analysis)

        assert result.spec == spec
        assert result.task_analysis == analysis
        assert result.confidence == 0.8
        assert result.reasoning == ''
        assert result.alternatives == []

    def test_custom_values(self):
        """Test custom values."""
        spec = AgentSpec()
        analysis = TaskAnalysis(description='Test')
        alt_spec = AgentSpec(agent_type='BrowsingAgent')

        result = SynthesisResult(
            spec=spec,
            task_analysis=analysis,
            confidence=0.95,
            reasoning='High confidence due to simple task',
            alternatives=[alt_spec],
        )

        assert result.confidence == 0.95
        assert len(result.alternatives) == 1


class TestImprovementSuggestion:
    """Tests for ImprovementSuggestion dataclass."""

    def test_default_confidence(self):
        """Test default confidence."""
        suggestion = ImprovementSuggestion(
            field='enable_plan_mode',
            current_value=False,
            suggested_value=True,
            reasoning='High step count',
            expected_improvement='Reduce steps',
        )

        assert suggestion.confidence == 0.5

    def test_custom_values(self):
        """Test custom values."""
        suggestion = ImprovementSuggestion(
            field='condenser_max_size',
            current_value=100,
            suggested_value=150,
            reasoning='Low success rate may need more context',
            expected_improvement='Better task completion',
            confidence=0.7,
        )

        assert suggestion.field == 'condenser_max_size'
        assert suggestion.current_value == 100
        assert suggestion.suggested_value == 150
        assert suggestion.confidence == 0.7
