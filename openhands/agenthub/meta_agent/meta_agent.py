"""Meta-Agent implementation.

CCA F4: A meta-agent that synthesizes optimal agent configurations
for given tasks, implementing the build-test-improve loop for
continuous optimization.

The MetaAgent serves as a factory and optimizer for other agents,
analyzing task requirements and generating tailored configurations.
"""

from typing import TYPE_CHECKING, Any

from openhands.agenthub.meta_agent.evaluator import (
    AgentEvaluator,
    BuildTestImproveLoop,
    EvaluationTask,
)
from openhands.agenthub.meta_agent.spec import (
    AgentSpec,
    EvaluationMetrics,
    SynthesisResult,
    TaskAnalysis,
)
from openhands.agenthub.meta_agent.synthesis import AgentSynthesizer
from openhands.core.logger import openhands_logger as logger

if TYPE_CHECKING:
    from openhands.controller.agent import Agent
    from openhands.controller.state.state import State
    from openhands.core.config import AgentConfig
    from openhands.llm.llm import LLM
    from openhands.llm.llm_registry import LLMRegistry


class MetaAgent:
    """Meta-agent for synthesizing and optimizing agent configurations.

    CCA F4: This agent doesn't execute tasks directly. Instead, it:
    1. Analyzes task descriptions to determine requirements
    2. Synthesizes optimal agent configurations
    3. Optionally runs build-test-improve loops for optimization

    Example:
        meta = MetaAgent(llm_registry)

        # Quick synthesis
        result = meta.synthesize_agent("Fix bugs in the authentication module")
        agent_config = AgentConfig(**result.spec.to_agent_config_dict())

        # With optimization
        result, metrics = meta.optimize_agent(
            description="Implement a new API endpoint",
            evaluation_tasks=[...],
            max_iterations=5,
        )
    """

    def __init__(
        self,
        llm_registry: 'LLMRegistry | None' = None,
        llm: 'LLM | None' = None,
    ):
        """Initialize the meta-agent.

        Args:
            llm_registry: Registry for creating LLMs (for evaluation)
            llm: Optional LLM for advanced synthesis analysis
        """
        self.llm_registry = llm_registry
        self.synthesizer = AgentSynthesizer(llm=llm)
        self.evaluator: AgentEvaluator | None = None

        if llm_registry is not None:
            self.evaluator = AgentEvaluator(llm_registry)

    def analyze_task(self, description: str) -> TaskAnalysis:
        """Analyze a task description without full synthesis.

        Args:
            description: Task description

        Returns:
            TaskAnalysis with requirements and complexity assessment
        """
        return self.synthesizer.analyze_task(description)

    def synthesize_agent(self, description: str) -> SynthesisResult:
        """Synthesize an optimal agent configuration for a task.

        This is the main entry point for quick agent synthesis without
        evaluation or optimization.

        Args:
            description: Task description

        Returns:
            SynthesisResult with spec and metadata
        """
        return self.synthesizer.synthesize(description)

    async def synthesize_agent_with_llm(self, description: str) -> SynthesisResult:
        """Synthesize using LLM for advanced analysis.

        Uses the LLM to refine heuristic-based synthesis with
        more sophisticated task understanding.

        Args:
            description: Task description

        Returns:
            SynthesisResult with LLM-refined spec
        """
        return await self.synthesizer.synthesize_with_llm(description)

    def create_agent(
        self,
        description: str,
        agent_config_overrides: dict[str, Any] | None = None,
    ) -> tuple['Agent', SynthesisResult]:
        """Synthesize and instantiate an agent for a task.

        Args:
            description: Task description
            agent_config_overrides: Optional overrides to apply to synthesized config

        Returns:
            Tuple of (instantiated Agent, SynthesisResult)

        Raises:
            ValueError: If llm_registry is not available
        """
        if self.llm_registry is None:
            raise ValueError('LLM registry required to create agents')

        # Synthesize configuration
        result = self.synthesize_agent(description)

        # Build config dict
        config_dict = result.spec.to_agent_config_dict()
        if agent_config_overrides:
            config_dict.update(agent_config_overrides)

        # Import here to avoid circular imports
        from openhands.controller.agent import Agent
        from openhands.core.config import AgentConfig

        config = AgentConfig(**config_dict)
        agent_cls = Agent.get_cls(result.spec.agent_type)
        agent = agent_cls(config=config, llm_registry=self.llm_registry)

        return agent, result

    def evaluate_spec(
        self,
        spec: AgentSpec,
        tasks: list[EvaluationTask],
    ) -> EvaluationMetrics:
        """Evaluate an agent specification against test tasks.

        Args:
            spec: Agent specification to evaluate
            tasks: List of evaluation tasks

        Returns:
            Aggregated evaluation metrics

        Raises:
            ValueError: If evaluator is not available
        """
        if self.evaluator is None:
            raise ValueError('LLM registry required for evaluation')

        return self.evaluator.evaluate(spec, tasks)

    def optimize_agent(
        self,
        description: str,
        evaluation_tasks: list[EvaluationTask],
        max_iterations: int = 5,
        improvement_threshold: float = 0.05,
    ) -> tuple[SynthesisResult, EvaluationMetrics, list[dict]]:
        """Synthesize and optimize an agent configuration.

        Runs the build-test-improve loop to iteratively refine
        the agent configuration based on evaluation results.

        Args:
            description: Initial task description for synthesis
            evaluation_tasks: Tasks to use for evaluation
            max_iterations: Maximum optimization iterations
            improvement_threshold: Minimum improvement to continue

        Returns:
            Tuple of (best_result, best_metrics, iteration_history)

        Raises:
            ValueError: If evaluator is not available
        """
        if self.evaluator is None:
            raise ValueError('LLM registry required for optimization')

        loop = BuildTestImproveLoop(self.synthesizer, self.evaluator)
        return loop.run(
            initial_description=description,
            tasks=evaluation_tasks,
            max_iterations=max_iterations,
            improvement_threshold=improvement_threshold,
        )

    def suggest_improvements(
        self,
        spec: AgentSpec,
        metrics: EvaluationMetrics,
    ) -> list[dict[str, Any]]:
        """Analyze metrics and suggest configuration improvements.

        Provides targeted suggestions for improving agent performance
        based on observed metrics.

        Args:
            spec: Current agent specification
            metrics: Evaluation metrics to analyze

        Returns:
            List of improvement suggestions with reasoning
        """
        suggestions = []

        # High error rate analysis
        if metrics.error_rate > 0.3:
            suggestions.append({
                'issue': 'High error rate',
                'current_value': f'{metrics.error_rate:.1%}',
                'recommendations': [
                    {'field': 'enable_stuck_detection', 'value': True}
                    if not spec.enable_stuck_detection
                    else None,
                    {'field': 'condenser_max_size', 'value': min(spec.condenser_max_size + 50, 200)}
                    if spec.condenser_max_size < 200
                    else None,
                ],
            })

        # High step count analysis
        if metrics.avg_steps > 30:
            suggestions.append({
                'issue': 'High step count',
                'current_value': f'{metrics.avg_steps:.1f} steps',
                'recommendations': [
                    {'field': 'enable_plan_mode', 'value': True}
                    if not spec.enable_plan_mode
                    else None,
                    {'field': 'enable_think', 'value': True}
                    if not spec.enable_think
                    else None,
                ],
            })

        # Low success rate analysis
        if metrics.success_rate < 0.5:
            suggestions.append({
                'issue': 'Low success rate',
                'current_value': f'{metrics.success_rate:.1%}',
                'recommendations': [
                    {'field': 'enable_browsing', 'value': True}
                    if not spec.enable_browsing
                    else None,
                    {'field': 'enable_jupyter', 'value': True}
                    if not spec.enable_jupyter
                    else None,
                ],
            })

        # High cost analysis
        if metrics.avg_cost > 0.1:  # > $0.10 per task
            suggestions.append({
                'issue': 'High cost per task',
                'current_value': f'${metrics.avg_cost:.4f}',
                'recommendations': [
                    {'field': 'condenser_max_size', 'value': max(spec.condenser_max_size - 25, 50)}
                    if spec.condenser_max_size > 50
                    else None,
                    {'field': 'enable_history_truncation', 'value': True}
                    if not spec.enable_history_truncation
                    else None,
                ],
            })

        # Filter out None recommendations
        for suggestion in suggestions:
            suggestion['recommendations'] = [
                r for r in suggestion['recommendations'] if r is not None
            ]

        return [s for s in suggestions if s['recommendations']]

    def compare_specs(
        self,
        specs: list[AgentSpec],
        tasks: list[EvaluationTask],
    ) -> list[tuple[AgentSpec, EvaluationMetrics]]:
        """Compare multiple agent specifications.

        Evaluates each spec and returns results sorted by performance.

        Args:
            specs: List of agent specifications to compare
            tasks: Evaluation tasks

        Returns:
            List of (spec, metrics) tuples sorted by performance (best first)

        Raises:
            ValueError: If evaluator is not available
        """
        if self.evaluator is None:
            raise ValueError('LLM registry required for comparison')

        results = []
        for spec in specs:
            logger.info(f'Evaluating spec: {spec.name or spec.agent_type}')
            metrics = self.evaluator.evaluate(spec, tasks)
            results.append((spec, metrics))

        # Sort by performance (using is_better_than comparison)
        results.sort(key=lambda x: (
            x[1].success_rate * 100
            - x[1].error_rate * 50
            - x[1].avg_cost * 10
        ), reverse=True)

        return results

    def get_recommended_spec(
        self,
        description: str,
        preferred_traits: dict[str, Any] | None = None,
    ) -> AgentSpec:
        """Get a recommended spec with optional trait preferences.

        Args:
            description: Task description
            preferred_traits: Optional preferences like:
                - 'fast': Prioritize speed over accuracy
                - 'accurate': Prioritize accuracy over speed
                - 'cheap': Minimize cost
                - 'thorough': Enable all relevant tools

        Returns:
            Recommended AgentSpec
        """
        result = self.synthesize_agent(description)
        spec = result.spec

        if preferred_traits:
            if preferred_traits.get('fast'):
                spec.enable_plan_mode = False
                spec.condenser_max_size = min(spec.condenser_max_size, 50)

            if preferred_traits.get('accurate'):
                spec.enable_plan_mode = True
                spec.enable_think = True
                spec.enable_stuck_detection = True
                spec.condenser_max_size = max(spec.condenser_max_size, 150)

            if preferred_traits.get('cheap'):
                spec.condenser_max_size = 50
                spec.enable_history_truncation = True

            if preferred_traits.get('thorough'):
                spec.enable_browsing = True
                spec.enable_editor = True
                spec.enable_jupyter = True
                spec.enable_cmd = True
                spec.enable_think = True

        return spec

    def __repr__(self) -> str:
        """String representation."""
        llm_available = self.synthesizer.llm is not None
        eval_available = self.evaluator is not None
        return (
            f'MetaAgent(llm={llm_available}, evaluator={eval_available})'
        )
