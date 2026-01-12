"""Agent evaluation and improvement system.

CCA F4: Implements the build-test-improve loop for optimizing
agent configurations based on evaluation metrics.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from openhands.agenthub.meta_agent.spec import (
    AgentSpec,
    EvaluationMetrics,
    ImprovementSuggestion,
    SynthesisResult,
)
from openhands.core.logger import openhands_logger as logger

if TYPE_CHECKING:
    from openhands.controller.agent import Agent
    from openhands.controller.state.state import State
    from openhands.core.config import AgentConfig
    from openhands.llm.llm_registry import LLMRegistry


@dataclass
class EvaluationTask:
    """A task for evaluating agent configurations."""

    task_id: str
    """Unique identifier for the task."""

    description: str
    """Task description/prompt."""

    expected_outcome: str | None = None
    """Expected outcome for validation."""

    validation_fn: Callable[[Any], bool] | None = None
    """Custom validation function."""

    timeout_seconds: float = 300.0
    """Maximum time allowed for task completion."""

    max_iterations: int = 50
    """Maximum iterations allowed."""


@dataclass
class TaskResult:
    """Result of running a single evaluation task."""

    task_id: str
    """ID of the task that was run."""

    success: bool
    """Whether the task completed successfully."""

    steps: int
    """Number of steps taken."""

    cost: float
    """Cost in USD or tokens."""

    time_seconds: float
    """Time taken in seconds."""

    error: str | None = None
    """Error message if failed."""

    final_state: Any = None
    """Final agent state."""

    outputs: dict[str, Any] = field(default_factory=dict)
    """Task outputs."""


class AgentEvaluator:
    """Evaluates agent configurations against test tasks.

    CCA F4: Provides infrastructure for testing synthesized agent
    configurations and measuring their performance.

    Example:
        evaluator = AgentEvaluator(llm_registry)
        tasks = [EvaluationTask(task_id="test1", description="Fix bug in foo.py")]
        metrics = evaluator.evaluate(spec, tasks)
    """

    def __init__(self, llm_registry: 'LLMRegistry'):
        """Initialize the evaluator.

        Args:
            llm_registry: Registry for creating LLMs
        """
        self.llm_registry = llm_registry

    def evaluate(
        self,
        spec: AgentSpec,
        tasks: list[EvaluationTask],
    ) -> EvaluationMetrics:
        """Evaluate an agent spec against a set of tasks.

        Args:
            spec: The agent specification to evaluate
            tasks: List of tasks to run

        Returns:
            Aggregated evaluation metrics
        """
        results: list[TaskResult] = []

        for task in tasks:
            logger.info(f'Evaluating task: {task.task_id}')
            result = self._run_task(spec, task)
            results.append(result)
            logger.info(
                f'Task {task.task_id}: '
                f'success={result.success}, '
                f'steps={result.steps}, '
                f'time={result.time_seconds:.1f}s'
            )

        return self._aggregate_results(results)

    def _run_task(self, spec: AgentSpec, task: EvaluationTask) -> TaskResult:
        """Run a single evaluation task.

        Args:
            spec: Agent specification
            task: Task to run

        Returns:
            TaskResult with execution details
        """
        start_time = time.time()

        try:
            # Create agent from spec
            from openhands.controller.agent import Agent
            from openhands.core.config import AgentConfig

            # Build config from spec
            config_dict = spec.to_agent_config_dict()
            config = AgentConfig(**config_dict)

            # Get agent class
            agent_cls = Agent.get_cls(spec.agent_type)
            agent = agent_cls(config=config, llm_registry=self.llm_registry)

            # Run evaluation
            # Note: In a real implementation, this would use the full
            # AgentController loop. For now, we simulate evaluation.
            result = self._simulate_evaluation(agent, task, start_time)

        except Exception as e:
            logger.warning(f'Task {task.task_id} failed with error: {e}')
            result = TaskResult(
                task_id=task.task_id,
                success=False,
                steps=0,
                cost=0.0,
                time_seconds=time.time() - start_time,
                error=str(e),
            )

        return result

    def _simulate_evaluation(
        self,
        agent: 'Agent',
        task: EvaluationTask,
        start_time: float,
    ) -> TaskResult:
        """Simulate task evaluation.

        In a full implementation, this would:
        1. Create an event stream and controller
        2. Run the agent step loop
        3. Track metrics and validate outcomes

        For now, returns a placeholder result.
        """
        # Placeholder - real implementation would run agent loop
        elapsed = time.time() - start_time

        # Estimate cost based on task description length
        estimated_tokens = len(task.description.split()) * 100
        estimated_cost = estimated_tokens * 0.00001  # Rough estimate

        return TaskResult(
            task_id=task.task_id,
            success=True,  # Placeholder
            steps=10,  # Placeholder
            cost=estimated_cost,
            time_seconds=elapsed,
            outputs={},
        )

    def _aggregate_results(self, results: list[TaskResult]) -> EvaluationMetrics:
        """Aggregate individual task results into metrics."""
        if not results:
            return EvaluationMetrics()

        total = len(results)
        successful = sum(1 for r in results if r.success)
        errored = sum(1 for r in results if r.error is not None)

        total_steps = sum(r.steps for r in results)
        total_cost = sum(r.cost for r in results)
        total_time = sum(r.time_seconds for r in results)

        return EvaluationMetrics(
            success_rate=successful / total if total > 0 else 0.0,
            avg_steps=total_steps / total if total > 0 else 0.0,
            avg_cost=total_cost / total if total > 0 else 0.0,
            avg_time_seconds=total_time / total if total > 0 else 0.0,
            error_rate=errored / total if total > 0 else 0.0,
            stuck_rate=0.0,  # Would need stuck detection tracking
            tasks_evaluated=total,
            task_results=[
                {
                    'task_id': r.task_id,
                    'success': r.success,
                    'steps': r.steps,
                    'cost': r.cost,
                    'time': r.time_seconds,
                    'error': r.error,
                }
                for r in results
            ],
        )


class BuildTestImproveLoop:
    """Implements the build-test-improve optimization loop.

    CCA F4: Iteratively improves agent configurations by:
    1. Building (synthesizing) a configuration
    2. Testing it against evaluation tasks
    3. Analyzing results and suggesting improvements
    4. Applying improvements and repeating

    Example:
        loop = BuildTestImproveLoop(synthesizer, evaluator)
        result = loop.run(
            initial_description="Fix bugs in the codebase",
            tasks=evaluation_tasks,
            max_iterations=5,
        )
    """

    def __init__(
        self,
        synthesizer: 'AgentSynthesizer',
        evaluator: AgentEvaluator,
    ):
        """Initialize the loop.

        Args:
            synthesizer: For generating configurations
            evaluator: For testing configurations
        """
        from openhands.agenthub.meta_agent.synthesis import AgentSynthesizer

        self.synthesizer = synthesizer
        self.evaluator = evaluator

    def run(
        self,
        initial_description: str,
        tasks: list[EvaluationTask],
        max_iterations: int = 5,
        improvement_threshold: float = 0.05,
    ) -> tuple[SynthesisResult, EvaluationMetrics, list[dict]]:
        """Run the build-test-improve loop.

        Args:
            initial_description: Task description for initial synthesis
            tasks: Evaluation tasks
            max_iterations: Maximum improvement iterations
            improvement_threshold: Minimum improvement to continue

        Returns:
            Tuple of (best_result, best_metrics, iteration_history)
        """
        # Initial synthesis
        best_result = self.synthesizer.synthesize(initial_description)
        best_metrics = self.evaluator.evaluate(best_result.spec, tasks)

        history = [
            {
                'iteration': 0,
                'spec': best_result.spec,
                'metrics': best_metrics,
                'improvements_applied': [],
            }
        ]

        logger.info(
            f'Initial evaluation: success_rate={best_metrics.success_rate:.2%}, '
            f'avg_steps={best_metrics.avg_steps:.1f}'
        )

        for iteration in range(1, max_iterations + 1):
            # Analyze and suggest improvements
            suggestions = self._analyze_and_suggest(best_result, best_metrics)

            if not suggestions:
                logger.info(f'No improvements suggested at iteration {iteration}')
                break

            # Apply improvements
            improved_spec = self._apply_improvements(best_result.spec, suggestions)
            improved_result = SynthesisResult(
                spec=improved_spec,
                task_analysis=best_result.task_analysis,
                confidence=best_result.confidence,
                reasoning=f'Iteration {iteration}: ' + '; '.join(
                    s.reasoning for s in suggestions
                ),
            )

            # Evaluate improved spec
            improved_metrics = self.evaluator.evaluate(improved_spec, tasks)

            # Check if improved
            if improved_metrics.is_better_than(best_metrics):
                improvement = (
                    improved_metrics.success_rate - best_metrics.success_rate
                )
                logger.info(
                    f'Iteration {iteration}: improvement +{improvement:.2%} '
                    f'(success_rate={improved_metrics.success_rate:.2%})'
                )

                if improvement < improvement_threshold:
                    logger.info('Improvement below threshold, stopping')
                    break

                best_result = improved_result
                best_metrics = improved_metrics
            else:
                logger.info(f'Iteration {iteration}: no improvement, reverting')

            history.append(
                {
                    'iteration': iteration,
                    'spec': improved_spec,
                    'metrics': improved_metrics,
                    'improvements_applied': [
                        {'field': s.field, 'value': s.suggested_value}
                        for s in suggestions
                    ],
                }
            )

        return best_result, best_metrics, history

    def _analyze_and_suggest(
        self,
        result: SynthesisResult,
        metrics: EvaluationMetrics,
    ) -> list[ImprovementSuggestion]:
        """Analyze metrics and suggest improvements."""
        suggestions = []

        # High error rate -> enable stuck detection, increase history
        if metrics.error_rate > 0.3:
            if not result.spec.enable_stuck_detection:
                suggestions.append(
                    ImprovementSuggestion(
                        field='enable_stuck_detection',
                        current_value=False,
                        suggested_value=True,
                        reasoning='High error rate suggests stuck agents',
                        expected_improvement='Reduce error rate',
                        confidence=0.7,
                    )
                )

        # High step count -> enable plan mode
        if metrics.avg_steps > 30 and not result.spec.enable_plan_mode:
            suggestions.append(
                ImprovementSuggestion(
                    field='enable_plan_mode',
                    current_value=False,
                    suggested_value=True,
                    reasoning='High step count suggests need for planning',
                    expected_improvement='Reduce steps through better planning',
                    confidence=0.6,
                )
            )

        # Low success but complex task -> increase condenser size
        if metrics.success_rate < 0.5:
            if result.spec.condenser_max_size < 150:
                suggestions.append(
                    ImprovementSuggestion(
                        field='condenser_max_size',
                        current_value=result.spec.condenser_max_size,
                        suggested_value=min(result.spec.condenser_max_size + 50, 200),
                        reasoning='Low success rate may benefit from more context',
                        expected_improvement='Better task completion with more history',
                        confidence=0.5,
                    )
                )

        return suggestions

    def _apply_improvements(
        self,
        spec: AgentSpec,
        suggestions: list[ImprovementSuggestion],
    ) -> AgentSpec:
        """Apply improvement suggestions to a spec."""
        # Create a copy of the spec
        import copy
        new_spec = copy.deepcopy(spec)

        for suggestion in suggestions:
            if hasattr(new_spec, suggestion.field):
                setattr(new_spec, suggestion.field, suggestion.suggested_value)
                logger.debug(
                    f'Applied improvement: {suggestion.field}='
                    f'{suggestion.suggested_value}'
                )

        return new_spec
