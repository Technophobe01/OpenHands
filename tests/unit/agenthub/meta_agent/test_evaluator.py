"""Tests for agent evaluation and build-test-improve loop."""

import pytest
from unittest.mock import MagicMock, patch

from openhands.agenthub.meta_agent.evaluator import (
    AgentEvaluator,
    BuildTestImproveLoop,
    EvaluationTask,
    TaskResult,
)
from openhands.agenthub.meta_agent.spec import (
    AgentSpec,
    EvaluationMetrics,
    SynthesisResult,
    TaskAnalysis,
    TaskComplexity,
)
from openhands.agenthub.meta_agent.synthesis import AgentSynthesizer


class TestEvaluationTask:
    """Tests for EvaluationTask dataclass."""

    def test_default_values(self):
        """Test default values."""
        task = EvaluationTask(
            task_id='test1',
            description='Test task',
        )

        assert task.task_id == 'test1'
        assert task.description == 'Test task'
        assert task.expected_outcome is None
        assert task.validation_fn is None
        assert task.timeout_seconds == 300.0
        assert task.max_iterations == 50

    def test_custom_values(self):
        """Test custom values."""
        def validator(result):
            return True

        task = EvaluationTask(
            task_id='custom',
            description='Custom task',
            expected_outcome='Success',
            validation_fn=validator,
            timeout_seconds=600.0,
            max_iterations=100,
        )

        assert task.expected_outcome == 'Success'
        assert task.validation_fn is validator
        assert task.timeout_seconds == 600.0


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_default_values(self):
        """Test default values."""
        result = TaskResult(
            task_id='test1',
            success=True,
            steps=10,
            cost=0.01,
            time_seconds=5.0,
        )

        assert result.task_id == 'test1'
        assert result.success is True
        assert result.steps == 10
        assert result.cost == 0.01
        assert result.time_seconds == 5.0
        assert result.error is None
        assert result.final_state is None
        assert result.outputs == {}

    def test_with_error(self):
        """Test result with error."""
        result = TaskResult(
            task_id='failed',
            success=False,
            steps=5,
            cost=0.005,
            time_seconds=2.0,
            error='Agent got stuck',
        )

        assert result.success is False
        assert result.error == 'Agent got stuck'


class TestAgentEvaluator:
    """Tests for AgentEvaluator."""

    @pytest.fixture
    def mock_llm_registry(self):
        """Create mock LLM registry."""
        registry = MagicMock()
        return registry

    @pytest.fixture
    def evaluator(self, mock_llm_registry):
        """Create evaluator with mock registry."""
        return AgentEvaluator(mock_llm_registry)

    def test_init(self, evaluator, mock_llm_registry):
        """Test evaluator initialization."""
        assert evaluator.llm_registry == mock_llm_registry

    def test_evaluate_empty_tasks(self, evaluator):
        """Test evaluation with no tasks."""
        spec = AgentSpec()
        metrics = evaluator.evaluate(spec, [])

        assert metrics.tasks_evaluated == 0
        assert metrics.success_rate == 0.0

    @patch.object(AgentEvaluator, '_run_task')
    def test_evaluate_single_task(self, mock_run_task, evaluator):
        """Test evaluation of single task."""
        mock_run_task.return_value = TaskResult(
            task_id='test1',
            success=True,
            steps=10,
            cost=0.01,
            time_seconds=5.0,
        )

        spec = AgentSpec()
        tasks = [EvaluationTask(task_id='test1', description='Test')]
        metrics = evaluator.evaluate(spec, tasks)

        assert metrics.tasks_evaluated == 1
        assert metrics.success_rate == 1.0
        assert metrics.avg_steps == 10
        assert metrics.avg_cost == 0.01

    @patch.object(AgentEvaluator, '_run_task')
    def test_evaluate_multiple_tasks(self, mock_run_task, evaluator):
        """Test evaluation of multiple tasks."""
        mock_run_task.side_effect = [
            TaskResult(task_id='test1', success=True, steps=10, cost=0.01, time_seconds=5.0),
            TaskResult(task_id='test2', success=False, steps=20, cost=0.02, time_seconds=10.0, error='Failed'),
            TaskResult(task_id='test3', success=True, steps=15, cost=0.015, time_seconds=7.5),
        ]

        spec = AgentSpec()
        tasks = [
            EvaluationTask(task_id='test1', description='Test 1'),
            EvaluationTask(task_id='test2', description='Test 2'),
            EvaluationTask(task_id='test3', description='Test 3'),
        ]
        metrics = evaluator.evaluate(spec, tasks)

        assert metrics.tasks_evaluated == 3
        assert metrics.success_rate == pytest.approx(2/3, rel=0.01)
        assert metrics.avg_steps == pytest.approx(15.0, rel=0.01)
        assert metrics.error_rate == pytest.approx(1/3, rel=0.01)

    def test_aggregate_results_empty(self, evaluator):
        """Test aggregation of empty results."""
        metrics = evaluator._aggregate_results([])

        assert metrics.tasks_evaluated == 0
        assert metrics.success_rate == 0.0

    def test_aggregate_results_all_success(self, evaluator):
        """Test aggregation of all successful results."""
        results = [
            TaskResult(task_id='t1', success=True, steps=10, cost=0.01, time_seconds=5.0),
            TaskResult(task_id='t2', success=True, steps=20, cost=0.02, time_seconds=10.0),
        ]

        metrics = evaluator._aggregate_results(results)

        assert metrics.success_rate == 1.0
        assert metrics.error_rate == 0.0
        assert metrics.avg_steps == 15.0
        assert metrics.avg_cost == 0.015

    def test_aggregate_results_mixed(self, evaluator):
        """Test aggregation of mixed results."""
        results = [
            TaskResult(task_id='t1', success=True, steps=10, cost=0.01, time_seconds=5.0),
            TaskResult(task_id='t2', success=False, steps=5, cost=0.005, time_seconds=2.5, error='Error'),
        ]

        metrics = evaluator._aggregate_results(results)

        assert metrics.success_rate == 0.5
        assert metrics.error_rate == 0.5
        assert len(metrics.task_results) == 2


class TestBuildTestImproveLoop:
    """Tests for BuildTestImproveLoop."""

    @pytest.fixture
    def mock_synthesizer(self):
        """Create mock synthesizer."""
        synthesizer = MagicMock(spec=AgentSynthesizer)
        synthesizer.synthesize.return_value = SynthesisResult(
            spec=AgentSpec(),
            task_analysis=TaskAnalysis(description='Test'),
            confidence=0.8,
        )
        return synthesizer

    @pytest.fixture
    def mock_evaluator(self):
        """Create mock evaluator."""
        evaluator = MagicMock(spec=AgentEvaluator)
        evaluator.evaluate.return_value = EvaluationMetrics(
            success_rate=0.8,
            avg_steps=15,
            avg_cost=0.01,
            tasks_evaluated=5,
        )
        return evaluator

    @pytest.fixture
    def loop(self, mock_synthesizer, mock_evaluator):
        """Create loop with mocks."""
        return BuildTestImproveLoop(mock_synthesizer, mock_evaluator)

    def test_init(self, loop, mock_synthesizer, mock_evaluator):
        """Test loop initialization."""
        assert loop.synthesizer == mock_synthesizer
        assert loop.evaluator == mock_evaluator

    def test_run_single_iteration(self, loop, mock_synthesizer, mock_evaluator):
        """Test run with single iteration (no improvement)."""
        tasks = [EvaluationTask(task_id='t1', description='Test')]

        # First evaluation returns good results
        mock_evaluator.evaluate.return_value = EvaluationMetrics(
            success_rate=0.95,
            avg_steps=10,
            tasks_evaluated=1,
        )

        result, metrics, history = loop.run(
            initial_description='Test task',
            tasks=tasks,
            max_iterations=1,
        )

        assert result.spec is not None
        assert metrics.success_rate == 0.95
        assert len(history) >= 1

    def test_run_with_improvements(self, loop, mock_synthesizer, mock_evaluator):
        """Test run with improving iterations."""
        tasks = [EvaluationTask(task_id='t1', description='Test')]

        # Simulate improving metrics
        call_count = [0]
        def improving_metrics(*args):
            call_count[0] += 1
            return EvaluationMetrics(
                success_rate=0.5 + (call_count[0] * 0.1),
                error_rate=0.5 - (call_count[0] * 0.1),
                avg_steps=20 - call_count[0],
                tasks_evaluated=1,
            )

        mock_evaluator.evaluate.side_effect = improving_metrics

        result, metrics, history = loop.run(
            initial_description='Test task',
            tasks=tasks,
            max_iterations=3,
        )

        # Should have run multiple iterations
        assert len(history) >= 1

    def test_analyze_and_suggest_high_error_rate(self, loop):
        """Test suggestions for high error rate."""
        result = SynthesisResult(
            spec=AgentSpec(enable_stuck_detection=False),
            task_analysis=TaskAnalysis(description='Test'),
            confidence=0.8,
        )
        metrics = EvaluationMetrics(
            error_rate=0.5,
            tasks_evaluated=10,
        )

        suggestions = loop._analyze_and_suggest(result, metrics)

        # Should suggest enabling stuck detection
        assert len(suggestions) > 0
        field_names = [s.field for s in suggestions]
        assert 'enable_stuck_detection' in field_names

    def test_analyze_and_suggest_high_steps(self, loop):
        """Test suggestions for high step count."""
        result = SynthesisResult(
            spec=AgentSpec(enable_plan_mode=False),
            task_analysis=TaskAnalysis(description='Test'),
            confidence=0.8,
        )
        metrics = EvaluationMetrics(
            avg_steps=50,
            tasks_evaluated=10,
        )

        suggestions = loop._analyze_and_suggest(result, metrics)

        # Should suggest enabling plan mode
        field_names = [s.field for s in suggestions]
        assert 'enable_plan_mode' in field_names

    def test_analyze_and_suggest_low_success(self, loop):
        """Test suggestions for low success rate."""
        result = SynthesisResult(
            spec=AgentSpec(condenser_max_size=50),
            task_analysis=TaskAnalysis(description='Test'),
            confidence=0.8,
        )
        metrics = EvaluationMetrics(
            success_rate=0.3,
            tasks_evaluated=10,
        )

        suggestions = loop._analyze_and_suggest(result, metrics)

        # Should suggest increasing condenser size
        assert len(suggestions) > 0
        field_names = [s.field for s in suggestions]
        assert 'condenser_max_size' in field_names

    def test_apply_improvements(self, loop):
        """Test applying improvements to spec."""
        from openhands.agenthub.meta_agent.spec import ImprovementSuggestion

        spec = AgentSpec(
            enable_plan_mode=False,
            condenser_max_size=100,
        )

        suggestions = [
            ImprovementSuggestion(
                field='enable_plan_mode',
                current_value=False,
                suggested_value=True,
                reasoning='Test',
                expected_improvement='Test',
            ),
            ImprovementSuggestion(
                field='condenser_max_size',
                current_value=100,
                suggested_value=150,
                reasoning='Test',
                expected_improvement='Test',
            ),
        ]

        new_spec = loop._apply_improvements(spec, suggestions)

        assert new_spec.enable_plan_mode is True
        assert new_spec.condenser_max_size == 150
        # Original should be unchanged
        assert spec.enable_plan_mode is False

    def test_apply_improvements_invalid_field(self, loop):
        """Test that invalid fields are ignored."""
        from openhands.agenthub.meta_agent.spec import ImprovementSuggestion

        spec = AgentSpec()

        suggestions = [
            ImprovementSuggestion(
                field='nonexistent_field',
                current_value=None,
                suggested_value='value',
                reasoning='Test',
                expected_improvement='Test',
            ),
        ]

        new_spec = loop._apply_improvements(spec, suggestions)

        # Should not raise, just skip invalid field
        assert new_spec is not None
