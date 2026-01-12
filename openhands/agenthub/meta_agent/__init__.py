"""CCA F4: Meta-Agent for agent configuration synthesis.

This module provides the meta-agent system that analyzes tasks and
synthesizes optimal agent configurations. Key features:

1. Task Analysis: Analyzes task descriptions to determine complexity,
   domains, and requirements.

2. Agent Synthesis: Generates optimal agent configurations based on
   task analysis, selecting appropriate tools and settings.

3. Build-Test-Improve Loop: Iteratively optimizes configurations by
   evaluating against test tasks and applying improvements.

Example:
    from openhands.agenthub.meta_agent import MetaAgent, AgentSpec

    # Create meta-agent
    meta = MetaAgent(llm_registry)

    # Synthesize an agent configuration
    result = meta.synthesize_agent("Fix authentication bugs in login.py")
    print(f"Recommended: {result.spec.agent_type}")
    print(f"Confidence: {result.confidence}")

    # Create agent directly
    agent, result = meta.create_agent("Implement user registration")

    # Optimize with evaluation
    best_result, metrics, history = meta.optimize_agent(
        description="Refactor the API module",
        evaluation_tasks=[...],
    )
"""

from openhands.agenthub.meta_agent.evaluator import (
    AgentEvaluator,
    BuildTestImproveLoop,
    EvaluationTask,
    TaskResult,
)
from openhands.agenthub.meta_agent.meta_agent import MetaAgent
from openhands.agenthub.meta_agent.spec import (
    AgentSpec,
    EvaluationMetrics,
    ImprovementSuggestion,
    SynthesisResult,
    TaskAnalysis,
    TaskComplexity,
    TaskDomain,
)
from openhands.agenthub.meta_agent.synthesis import AgentSynthesizer

__all__ = [
    # Main class
    'MetaAgent',
    # Synthesis
    'AgentSynthesizer',
    'SynthesisResult',
    # Specifications
    'AgentSpec',
    'TaskAnalysis',
    'TaskComplexity',
    'TaskDomain',
    # Evaluation
    'AgentEvaluator',
    'BuildTestImproveLoop',
    'EvaluationTask',
    'TaskResult',
    'EvaluationMetrics',
    'ImprovementSuggestion',
]
