"""Tests for the main MetaAgent class."""

import pytest
from unittest.mock import MagicMock, patch

from openhands.agenthub.meta_agent.evaluator import EvaluationTask
from openhands.agenthub.meta_agent.meta_agent import MetaAgent
from openhands.agenthub.meta_agent.spec import (
    AgentSpec,
    EvaluationMetrics,
    SynthesisResult,
    TaskAnalysis,
    TaskComplexity,
)


class TestMetaAgent:
    """Tests for MetaAgent."""

    @pytest.fixture
    def mock_llm_registry(self):
        """Create mock LLM registry."""
        return MagicMock()

    @pytest.fixture
    def meta_agent(self, mock_llm_registry):
        """Create MetaAgent with mock registry."""
        return MetaAgent(llm_registry=mock_llm_registry)

    @pytest.fixture
    def meta_agent_no_registry(self):
        """Create MetaAgent without registry."""
        return MetaAgent()

    def test_init_with_registry(self, meta_agent, mock_llm_registry):
        """Test initialization with LLM registry."""
        assert meta_agent.llm_registry == mock_llm_registry
        assert meta_agent.synthesizer is not None
        assert meta_agent.evaluator is not None

    def test_init_without_registry(self, meta_agent_no_registry):
        """Test initialization without LLM registry."""
        assert meta_agent_no_registry.llm_registry is None
        assert meta_agent_no_registry.synthesizer is not None
        assert meta_agent_no_registry.evaluator is None

    def test_analyze_task(self, meta_agent):
        """Test task analysis."""
        analysis = meta_agent.analyze_task('Fix a bug in the login function')

        assert isinstance(analysis, TaskAnalysis)
        assert analysis.description == 'Fix a bug in the login function'
        assert analysis.complexity is not None

    def test_analyze_task_simple(self, meta_agent):
        """Test analysis of simple task."""
        analysis = meta_agent.analyze_task('Simple typo fix')

        assert analysis.complexity == TaskComplexity.SIMPLE

    def test_analyze_task_complex(self, meta_agent):
        """Test analysis of complex task."""
        analysis = meta_agent.analyze_task(
            'Refactor the entire authentication module and implement OAuth2'
        )

        assert analysis.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXPERT)

    def test_synthesize_agent(self, meta_agent):
        """Test agent synthesis."""
        result = meta_agent.synthesize_agent('Implement a new API endpoint')

        assert isinstance(result, SynthesisResult)
        assert result.spec is not None
        assert result.task_analysis is not None
        assert 0 <= result.confidence <= 1

    def test_synthesize_agent_returns_codeact(self, meta_agent):
        """Test that synthesis returns CodeActAgent by default."""
        result = meta_agent.synthesize_agent('Fix bugs in the code')

        assert result.spec.agent_type == 'CodeActAgent'

    def test_synthesize_agent_readonly_for_analysis(self, meta_agent):
        """Test that synthesis returns ReadOnlyAgent for analysis tasks."""
        result = meta_agent.synthesize_agent('Analyze the code and explain it')

        assert result.spec.agent_type == 'ReadOnlyAgent'

    def test_evaluate_spec_requires_registry(self, meta_agent_no_registry):
        """Test that evaluate_spec requires registry."""
        spec = AgentSpec()
        tasks = [EvaluationTask(task_id='t1', description='Test')]

        with pytest.raises(ValueError, match='LLM registry required'):
            meta_agent_no_registry.evaluate_spec(spec, tasks)

    @patch('openhands.agenthub.meta_agent.meta_agent.AgentEvaluator')
    def test_evaluate_spec(self, mock_evaluator_cls, meta_agent):
        """Test spec evaluation."""
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = EvaluationMetrics(
            success_rate=0.9,
            tasks_evaluated=5,
        )
        meta_agent.evaluator = mock_evaluator

        spec = AgentSpec()
        tasks = [EvaluationTask(task_id='t1', description='Test')]
        metrics = meta_agent.evaluate_spec(spec, tasks)

        assert metrics.success_rate == 0.9
        mock_evaluator.evaluate.assert_called_once()

    def test_optimize_agent_requires_registry(self, meta_agent_no_registry):
        """Test that optimize_agent requires registry."""
        tasks = [EvaluationTask(task_id='t1', description='Test')]

        with pytest.raises(ValueError, match='LLM registry required'):
            meta_agent_no_registry.optimize_agent(
                description='Test',
                evaluation_tasks=tasks,
            )

    def test_suggest_improvements_high_error_rate(self, meta_agent):
        """Test improvement suggestions for high error rate."""
        spec = AgentSpec(enable_stuck_detection=False)
        metrics = EvaluationMetrics(
            error_rate=0.5,
            success_rate=0.5,
            tasks_evaluated=10,
        )

        suggestions = meta_agent.suggest_improvements(spec, metrics)

        assert len(suggestions) > 0
        issues = [s['issue'] for s in suggestions]
        assert 'High error rate' in issues

    def test_suggest_improvements_high_steps(self, meta_agent):
        """Test improvement suggestions for high step count."""
        spec = AgentSpec(enable_plan_mode=False)
        metrics = EvaluationMetrics(
            avg_steps=50,
            success_rate=0.8,
            tasks_evaluated=10,
        )

        suggestions = meta_agent.suggest_improvements(spec, metrics)

        issues = [s['issue'] for s in suggestions]
        assert 'High step count' in issues

    def test_suggest_improvements_low_success(self, meta_agent):
        """Test improvement suggestions for low success rate."""
        spec = AgentSpec(enable_browsing=False)
        metrics = EvaluationMetrics(
            success_rate=0.3,
            tasks_evaluated=10,
        )

        suggestions = meta_agent.suggest_improvements(spec, metrics)

        issues = [s['issue'] for s in suggestions]
        assert 'Low success rate' in issues

    def test_suggest_improvements_high_cost(self, meta_agent):
        """Test improvement suggestions for high cost."""
        spec = AgentSpec(condenser_max_size=200)
        metrics = EvaluationMetrics(
            avg_cost=0.5,
            success_rate=0.8,
            tasks_evaluated=10,
        )

        suggestions = meta_agent.suggest_improvements(spec, metrics)

        issues = [s['issue'] for s in suggestions]
        assert 'High cost per task' in issues

    def test_suggest_improvements_no_issues(self, meta_agent):
        """Test no suggestions when metrics are good."""
        spec = AgentSpec(
            enable_plan_mode=True,
            enable_stuck_detection=True,
            enable_browsing=True,
        )
        metrics = EvaluationMetrics(
            success_rate=0.95,
            error_rate=0.05,
            avg_steps=15,
            avg_cost=0.01,
            tasks_evaluated=10,
        )

        suggestions = meta_agent.suggest_improvements(spec, metrics)

        # Should have few or no suggestions with good metrics
        assert len(suggestions) == 0

    def test_get_recommended_spec(self, meta_agent):
        """Test getting recommended spec."""
        spec = meta_agent.get_recommended_spec('Fix a bug')

        assert isinstance(spec, AgentSpec)

    def test_get_recommended_spec_fast(self, meta_agent):
        """Test recommended spec with fast trait."""
        spec = meta_agent.get_recommended_spec(
            'Fix a bug',
            preferred_traits={'fast': True},
        )

        assert spec.enable_plan_mode is False
        assert spec.condenser_max_size <= 50

    def test_get_recommended_spec_accurate(self, meta_agent):
        """Test recommended spec with accurate trait."""
        spec = meta_agent.get_recommended_spec(
            'Fix a bug',
            preferred_traits={'accurate': True},
        )

        assert spec.enable_plan_mode is True
        assert spec.enable_think is True
        assert spec.enable_stuck_detection is True

    def test_get_recommended_spec_cheap(self, meta_agent):
        """Test recommended spec with cheap trait."""
        spec = meta_agent.get_recommended_spec(
            'Fix a bug',
            preferred_traits={'cheap': True},
        )

        assert spec.condenser_max_size == 50
        assert spec.enable_history_truncation is True

    def test_get_recommended_spec_thorough(self, meta_agent):
        """Test recommended spec with thorough trait."""
        spec = meta_agent.get_recommended_spec(
            'Fix a bug',
            preferred_traits={'thorough': True},
        )

        assert spec.enable_browsing is True
        assert spec.enable_editor is True
        assert spec.enable_jupyter is True
        assert spec.enable_cmd is True
        assert spec.enable_think is True

    def test_compare_specs_requires_registry(self, meta_agent_no_registry):
        """Test that compare_specs requires registry."""
        specs = [AgentSpec()]
        tasks = [EvaluationTask(task_id='t1', description='Test')]

        with pytest.raises(ValueError, match='LLM registry required'):
            meta_agent_no_registry.compare_specs(specs, tasks)

    @patch('openhands.agenthub.meta_agent.meta_agent.AgentEvaluator')
    def test_compare_specs(self, mock_evaluator_cls, meta_agent):
        """Test spec comparison."""
        mock_evaluator = MagicMock()
        # Return different metrics for different specs
        mock_evaluator.evaluate.side_effect = [
            EvaluationMetrics(success_rate=0.7, tasks_evaluated=5),
            EvaluationMetrics(success_rate=0.9, tasks_evaluated=5),
        ]
        meta_agent.evaluator = mock_evaluator

        spec1 = AgentSpec(name='spec1')
        spec2 = AgentSpec(name='spec2')
        tasks = [EvaluationTask(task_id='t1', description='Test')]

        results = meta_agent.compare_specs([spec1, spec2], tasks)

        assert len(results) == 2
        # Should be sorted by performance (best first)
        assert results[0][1].success_rate == 0.9
        assert results[1][1].success_rate == 0.7

    def test_create_agent_requires_registry(self, meta_agent_no_registry):
        """Test that create_agent requires registry."""
        with pytest.raises(ValueError, match='LLM registry required'):
            meta_agent_no_registry.create_agent('Test task')

    def test_repr(self, meta_agent):
        """Test string representation."""
        repr_str = repr(meta_agent)

        assert 'MetaAgent' in repr_str
        assert 'llm=' in repr_str
        assert 'evaluator=' in repr_str

    def test_repr_no_registry(self, meta_agent_no_registry):
        """Test string representation without registry."""
        repr_str = repr(meta_agent_no_registry)

        assert 'MetaAgent' in repr_str
        assert 'evaluator=False' in repr_str


class TestMetaAgentIntegration:
    """Integration tests for MetaAgent."""

    @pytest.fixture
    def meta_agent(self):
        """Create MetaAgent without registry for synthesis tests."""
        return MetaAgent()

    def test_full_synthesis_flow(self, meta_agent):
        """Test complete synthesis flow."""
        # Analyze task
        analysis = meta_agent.analyze_task(
            'Implement user authentication with JWT tokens'
        )

        assert analysis.complexity is not None
        assert analysis.requires_file_editing is True

        # Synthesize agent
        result = meta_agent.synthesize_agent(
            'Implement user authentication with JWT tokens'
        )

        assert result.spec.agent_type == 'CodeActAgent'
        assert result.confidence > 0.5

    def test_synthesis_for_different_task_types(self, meta_agent):
        """Test synthesis produces different configs for different tasks."""
        # Simple task
        simple_result = meta_agent.synthesize_agent('Fix typo')

        # Complex task
        complex_result = meta_agent.synthesize_agent(
            'Refactor the entire codebase architecture'
        )

        # Should have different configurations
        assert simple_result.spec.enable_plan_mode != complex_result.spec.enable_plan_mode or \
               simple_result.spec.condenser_max_size != complex_result.spec.condenser_max_size

    def test_synthesis_for_browsing_task(self, meta_agent):
        """Test synthesis enables browsing for web tasks."""
        result = meta_agent.synthesize_agent(
            'Fetch documentation from the website'
        )

        assert result.spec.enable_browsing is True

    def test_synthesis_for_data_task(self, meta_agent):
        """Test synthesis enables jupyter for data tasks."""
        result = meta_agent.synthesize_agent(
            'Analyze the dataset using pandas'
        )

        assert result.spec.enable_jupyter is True
