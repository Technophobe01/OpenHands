"""Meta-agent specification format.

CCA F4: Defines the specification format for meta-agent configurations,
including task analysis results and agent synthesis parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskComplexity(str, Enum):
    """Complexity levels for tasks."""

    SIMPLE = 'simple'
    """Single-step tasks with clear requirements."""

    MODERATE = 'moderate'
    """Multi-step tasks with some ambiguity."""

    COMPLEX = 'complex'
    """Long-horizon tasks requiring planning."""

    EXPERT = 'expert'
    """Highly complex tasks requiring specialized knowledge."""


class TaskDomain(str, Enum):
    """Domain categories for tasks."""

    CODING = 'coding'
    """Code writing, debugging, refactoring."""

    BROWSING = 'browsing'
    """Web navigation and information retrieval."""

    ANALYSIS = 'analysis'
    """Code analysis, review, documentation."""

    SYSTEM = 'system'
    """System administration, DevOps."""

    DATA = 'data'
    """Data processing, jupyter notebooks."""

    MIXED = 'mixed'
    """Combination of multiple domains."""


@dataclass
class TaskAnalysis:
    """Analysis of a task for agent synthesis.

    CCA F4: Captures the requirements and characteristics of a task
    to inform agent configuration synthesis.
    """

    description: str
    """Original task description."""

    complexity: TaskComplexity = TaskComplexity.MODERATE
    """Estimated complexity level."""

    domains: list[TaskDomain] = field(default_factory=lambda: [TaskDomain.CODING])
    """Domains involved in the task."""

    estimated_steps: int = 10
    """Estimated number of steps to complete."""

    requires_file_editing: bool = True
    """Whether the task requires file modifications."""

    requires_browsing: bool = False
    """Whether the task requires web browsing."""

    requires_jupyter: bool = False
    """Whether the task requires Jupyter notebooks."""

    requires_planning: bool = False
    """Whether the task benefits from explicit planning."""

    keywords: list[str] = field(default_factory=list)
    """Key terms extracted from the task."""

    related_files: list[str] = field(default_factory=list)
    """Files likely to be involved."""

    constraints: dict[str, Any] = field(default_factory=dict)
    """Additional constraints (time limits, restrictions, etc.)."""


@dataclass
class AgentSpec:
    """Specification for an agent configuration.

    CCA F4: Defines the parameters for synthesizing an agent,
    including which tools and capabilities to enable.
    """

    agent_type: str = 'CodeActAgent'
    """Base agent class to use."""

    # Tool enablement
    enable_browsing: bool = True
    """Enable web browsing tools."""

    enable_editor: bool = True
    """Enable file editing tools."""

    enable_jupyter: bool = True
    """Enable Jupyter/IPython tools."""

    enable_cmd: bool = True
    """Enable bash command execution."""

    enable_think: bool = True
    """Enable think tool for reasoning."""

    enable_plan_mode: bool = False
    """Enable plan mode for complex tasks."""

    enable_stuck_detection: bool = True
    """Enable stuck loop detection."""

    enable_history_truncation: bool = True
    """Enable automatic history truncation."""

    # LLM configuration
    llm_model: str | None = None
    """Specific LLM model to use (None = use default)."""

    temperature: float | None = None
    """Temperature for LLM sampling."""

    max_tokens: int | None = None
    """Maximum tokens for LLM responses."""

    # Condenser configuration
    condenser_type: str = 'observation_masking'
    """Type of history condenser to use."""

    condenser_keep_first: int = 1
    """Number of initial events to preserve."""

    condenser_max_size: int = 100
    """Maximum condensed history size."""

    # Extensions to enable
    extensions: list[str] = field(default_factory=list)
    """Names of extensions to activate."""

    # Microagent configuration
    disabled_microagents: list[str] = field(default_factory=list)
    """Microagents to disable."""

    # Custom prompt
    system_prompt_filename: str | None = None
    """Custom system prompt template."""

    # Metadata
    name: str = ''
    """Human-readable name for this spec."""

    description: str = ''
    """Description of this agent configuration."""

    def to_agent_config_dict(self) -> dict[str, Any]:
        """Convert to AgentConfig constructor arguments.

        Returns:
            Dictionary suitable for AgentConfig(**dict)
        """
        config = {
            'enable_browsing': self.enable_browsing,
            'enable_editor': self.enable_editor,
            'enable_jupyter': self.enable_jupyter,
            'enable_cmd': self.enable_cmd,
            'enable_think': self.enable_think,
            'enable_plan_mode': self.enable_plan_mode,
            'enable_stuck_detection': self.enable_stuck_detection,
            'enable_history_truncation': self.enable_history_truncation,
            'disabled_microagents': self.disabled_microagents,
        }

        if self.system_prompt_filename:
            config['system_prompt_filename'] = self.system_prompt_filename

        return config


@dataclass
class SynthesisResult:
    """Result of agent synthesis.

    CCA F4: Contains the synthesized agent specification along with
    metadata about the synthesis process.
    """

    spec: AgentSpec
    """The synthesized agent specification."""

    task_analysis: TaskAnalysis
    """Analysis used to generate the spec."""

    confidence: float = 0.8
    """Confidence score (0-1) in the synthesis."""

    reasoning: str = ''
    """Explanation of synthesis decisions."""

    alternatives: list[AgentSpec] = field(default_factory=list)
    """Alternative specs considered."""


@dataclass
class EvaluationMetrics:
    """Metrics from evaluating an agent configuration.

    CCA F4: Captures performance metrics for the build-test-improve loop.
    """

    success_rate: float = 0.0
    """Percentage of tasks completed successfully."""

    avg_steps: float = 0.0
    """Average number of steps per task."""

    avg_cost: float = 0.0
    """Average cost (in tokens or USD) per task."""

    avg_time_seconds: float = 0.0
    """Average time per task in seconds."""

    error_rate: float = 0.0
    """Percentage of tasks that errored."""

    stuck_rate: float = 0.0
    """Percentage of tasks where agent got stuck."""

    tasks_evaluated: int = 0
    """Number of tasks used for evaluation."""

    task_results: list[dict[str, Any]] = field(default_factory=list)
    """Detailed results per task."""

    def is_better_than(self, other: 'EvaluationMetrics') -> bool:
        """Compare if this result is better than another.

        Uses a weighted combination of metrics.
        """
        if self.tasks_evaluated == 0:
            return False
        if other.tasks_evaluated == 0:
            return True

        # Higher success rate is better
        # Lower cost/time/steps is better
        self_score = (
            self.success_rate * 100
            - self.error_rate * 50
            - self.stuck_rate * 30
            - (self.avg_cost * 10)
        )
        other_score = (
            other.success_rate * 100
            - other.error_rate * 50
            - other.stuck_rate * 30
            - (other.avg_cost * 10)
        )
        return self_score > other_score


@dataclass
class ImprovementSuggestion:
    """Suggestion for improving an agent configuration.

    CCA F4: Generated by analyzing evaluation results.
    """

    field: str
    """Configuration field to modify."""

    current_value: Any
    """Current value of the field."""

    suggested_value: Any
    """Suggested new value."""

    reasoning: str
    """Why this change is suggested."""

    expected_improvement: str
    """Expected impact of the change."""

    confidence: float = 0.5
    """Confidence in this suggestion (0-1)."""
