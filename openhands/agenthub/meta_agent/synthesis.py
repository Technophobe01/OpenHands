"""Agent configuration synthesis.

CCA F4: Synthesizes agent configurations based on task analysis,
automatically selecting tools, models, and settings.
"""

import re
from typing import TYPE_CHECKING

from openhands.agenthub.meta_agent.spec import (
    AgentSpec,
    SynthesisResult,
    TaskAnalysis,
    TaskComplexity,
    TaskDomain,
)
from openhands.core.logger import openhands_logger as logger

if TYPE_CHECKING:
    from openhands.llm.llm import LLM


# Keyword patterns for task analysis
BROWSING_KEYWORDS = [
    'web', 'browser', 'url', 'http', 'website', 'page', 'link',
    'download', 'fetch', 'scrape', 'online', 'internet',
]

JUPYTER_KEYWORDS = [
    'jupyter', 'notebook', 'ipython', 'data', 'pandas', 'numpy',
    'plot', 'visualization', 'analysis', 'csv', 'dataframe',
]

SYSTEM_KEYWORDS = [
    'install', 'docker', 'kubernetes', 'deploy', 'server', 'config',
    'environment', 'package', 'dependency', 'build', 'compile',
]

COMPLEX_INDICATORS = [
    'refactor', 'architecture', 'design', 'system', 'migrate',
    'implement', 'create', 'build', 'develop', 'feature',
]


class AgentSynthesizer:
    """Synthesizes agent configurations from task descriptions.

    CCA F4: Analyzes tasks and generates optimal agent configurations
    based on task requirements and characteristics.

    Example:
        synthesizer = AgentSynthesizer()
        result = synthesizer.synthesize("Fix the bug in login.py")
        agent_config = AgentConfig(**result.spec.to_agent_config_dict())
    """

    def __init__(self, llm: 'LLM | None' = None):
        """Initialize the synthesizer.

        Args:
            llm: Optional LLM for advanced analysis. If None, uses heuristics.
        """
        self.llm = llm

    def analyze_task(self, description: str) -> TaskAnalysis:
        """Analyze a task description to determine requirements.

        Args:
            description: The task description

        Returns:
            TaskAnalysis with determined requirements
        """
        description_lower = description.lower()

        # Determine domains
        domains = self._detect_domains(description_lower)

        # Determine complexity
        complexity = self._estimate_complexity(description, domains)

        # Extract keywords
        keywords = self._extract_keywords(description)

        # Determine tool requirements
        requires_browsing = any(kw in description_lower for kw in BROWSING_KEYWORDS)
        requires_jupyter = any(kw in description_lower for kw in JUPYTER_KEYWORDS)
        requires_file_editing = self._needs_file_editing(description_lower)
        requires_planning = complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXPERT)

        # Estimate steps
        estimated_steps = self._estimate_steps(complexity)

        return TaskAnalysis(
            description=description,
            complexity=complexity,
            domains=domains,
            estimated_steps=estimated_steps,
            requires_file_editing=requires_file_editing,
            requires_browsing=requires_browsing,
            requires_jupyter=requires_jupyter,
            requires_planning=requires_planning,
            keywords=keywords,
        )

    def synthesize(self, description: str) -> SynthesisResult:
        """Synthesize an agent configuration for a task.

        Args:
            description: The task description

        Returns:
            SynthesisResult with the synthesized configuration
        """
        # Analyze the task
        analysis = self.analyze_task(description)

        # Generate spec based on analysis
        spec = self._generate_spec(analysis)

        # Generate reasoning
        reasoning = self._generate_reasoning(analysis, spec)

        # Calculate confidence
        confidence = self._calculate_confidence(analysis)

        logger.info(
            f'Synthesized agent spec: {spec.name}, '
            f'complexity={analysis.complexity.value}, '
            f'confidence={confidence:.2f}'
        )

        return SynthesisResult(
            spec=spec,
            task_analysis=analysis,
            confidence=confidence,
            reasoning=reasoning,
        )

    async def synthesize_with_llm(self, description: str) -> SynthesisResult:
        """Synthesize using LLM for advanced analysis.

        Args:
            description: The task description

        Returns:
            SynthesisResult with LLM-assisted synthesis
        """
        if self.llm is None:
            logger.warning('No LLM available, falling back to heuristic synthesis')
            return self.synthesize(description)

        # Get heuristic analysis first
        heuristic_result = self.synthesize(description)

        # Use LLM to refine
        prompt = self._build_synthesis_prompt(description, heuristic_result)

        try:
            response = self.llm.completion(
                messages=[{'role': 'user', 'content': prompt}],
            )
            # Parse LLM response to refine spec
            refined_spec = self._parse_llm_response(
                response.choices[0].message.content,
                heuristic_result.spec,
            )
            heuristic_result.spec = refined_spec
            heuristic_result.confidence = min(heuristic_result.confidence + 0.1, 1.0)
        except Exception as e:
            logger.warning(f'LLM synthesis failed, using heuristics: {e}')

        return heuristic_result

    def _detect_domains(self, description: str) -> list[TaskDomain]:
        """Detect which domains a task involves."""
        domains = []

        # Check for coding indicators
        coding_indicators = ['code', 'function', 'class', 'bug', 'fix', 'implement', 'test']
        if any(ind in description for ind in coding_indicators):
            domains.append(TaskDomain.CODING)

        # Check for browsing
        if any(kw in description for kw in BROWSING_KEYWORDS):
            domains.append(TaskDomain.BROWSING)

        # Check for data/analysis
        if any(kw in description for kw in JUPYTER_KEYWORDS):
            domains.append(TaskDomain.DATA)

        # Check for system tasks
        if any(kw in description for kw in SYSTEM_KEYWORDS):
            domains.append(TaskDomain.SYSTEM)

        # Check for analysis tasks
        analysis_keywords = ['analyze', 'review', 'explain', 'document', 'understand']
        if any(kw in description for kw in analysis_keywords):
            domains.append(TaskDomain.ANALYSIS)

        # Default to coding if no domain detected
        if not domains:
            domains.append(TaskDomain.CODING)

        # Mark as mixed if multiple domains
        if len(domains) > 2:
            return [TaskDomain.MIXED]

        return domains

    def _estimate_complexity(
        self, description: str, domains: list[TaskDomain]
    ) -> TaskComplexity:
        """Estimate task complexity."""
        desc_lower = description.lower()
        word_count = len(description.split())

        # Simple tasks
        simple_indicators = ['simple', 'quick', 'small', 'typo', 'rename']
        if any(ind in desc_lower for ind in simple_indicators) or word_count < 20:
            return TaskComplexity.SIMPLE

        # Expert tasks
        expert_indicators = ['architecture', 'system design', 'migrate', 'refactor entire']
        if any(ind in desc_lower for ind in expert_indicators):
            return TaskComplexity.EXPERT

        # Complex tasks
        if any(ind in desc_lower for ind in COMPLEX_INDICATORS) or word_count > 100:
            return TaskComplexity.COMPLEX

        # Multiple domains suggest complexity
        if len(domains) > 1 or TaskDomain.MIXED in domains:
            return TaskComplexity.COMPLEX

        return TaskComplexity.MODERATE

    def _extract_keywords(self, description: str) -> list[str]:
        """Extract important keywords from description."""
        # Remove common words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'but', 'and', 'or', 'if',
            'because', 'until', 'while', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'you', 'your', 'it', 'its', 'we', 'they',
            'please', 'help', 'need', 'want',
        }

        # Extract words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', description.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Deduplicate while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords[:20]  # Limit to top 20

    def _needs_file_editing(self, description: str) -> bool:
        """Determine if task requires file editing."""
        readonly_indicators = ['read', 'find', 'search', 'list', 'show', 'explain', 'analyze']
        edit_indicators = ['fix', 'update', 'change', 'modify', 'add', 'remove', 'create', 'implement']

        has_readonly = any(ind in description for ind in readonly_indicators)
        has_edit = any(ind in description for ind in edit_indicators)

        # If explicitly read-only and no edit indicators
        if has_readonly and not has_edit:
            return False

        # Default to needing editing
        return True

    def _estimate_steps(self, complexity: TaskComplexity) -> int:
        """Estimate number of steps based on complexity."""
        step_estimates = {
            TaskComplexity.SIMPLE: 5,
            TaskComplexity.MODERATE: 15,
            TaskComplexity.COMPLEX: 40,
            TaskComplexity.EXPERT: 80,
        }
        return step_estimates.get(complexity, 15)

    def _generate_spec(self, analysis: TaskAnalysis) -> AgentSpec:
        """Generate an AgentSpec from task analysis."""
        # Determine agent type
        if not analysis.requires_file_editing:
            agent_type = 'ReadOnlyAgent'
        elif TaskDomain.BROWSING in analysis.domains:
            agent_type = 'VisualBrowsingAgent'
        else:
            agent_type = 'CodeActAgent'

        # Build spec
        spec = AgentSpec(
            agent_type=agent_type,
            enable_browsing=analysis.requires_browsing,
            enable_editor=analysis.requires_file_editing,
            enable_jupyter=analysis.requires_jupyter,
            enable_cmd=True,
            enable_think=True,
            enable_plan_mode=analysis.requires_planning,
            enable_stuck_detection=True,
            enable_history_truncation=True,
            name=f'synthesized_{analysis.complexity.value}',
            description=f'Auto-synthesized for: {analysis.description[:50]}...',
        )

        # Adjust condenser based on complexity
        if analysis.complexity == TaskComplexity.EXPERT:
            spec.condenser_max_size = 200
            spec.condenser_keep_first = 3
        elif analysis.complexity == TaskComplexity.COMPLEX:
            spec.condenser_max_size = 150
            spec.condenser_keep_first = 2
        elif analysis.complexity == TaskComplexity.SIMPLE:
            spec.condenser_max_size = 50
            spec.condenser_keep_first = 1

        return spec

    def _generate_reasoning(self, analysis: TaskAnalysis, spec: AgentSpec) -> str:
        """Generate reasoning for synthesis decisions."""
        reasons = []

        reasons.append(f'Task complexity: {analysis.complexity.value}')
        reasons.append(f'Domains: {", ".join(d.value for d in analysis.domains)}')
        reasons.append(f'Selected agent: {spec.agent_type}')

        if analysis.requires_planning:
            reasons.append('Plan mode enabled due to task complexity')

        if analysis.requires_browsing:
            reasons.append('Browsing enabled due to web-related keywords')

        if analysis.requires_jupyter:
            reasons.append('Jupyter enabled due to data/analysis keywords')

        if not analysis.requires_file_editing:
            reasons.append('Read-only mode: no file editing required')

        return '; '.join(reasons)

    def _calculate_confidence(self, analysis: TaskAnalysis) -> float:
        """Calculate confidence score for synthesis."""
        confidence = 0.7  # Base confidence

        # More keywords = better understanding
        if len(analysis.keywords) > 5:
            confidence += 0.1

        # Clear complexity = higher confidence
        if analysis.complexity in (TaskComplexity.SIMPLE, TaskComplexity.MODERATE):
            confidence += 0.1

        # Single domain = more certain
        if len(analysis.domains) == 1:
            confidence += 0.05

        return min(confidence, 0.95)

    def _build_synthesis_prompt(
        self, description: str, heuristic_result: SynthesisResult
    ) -> str:
        """Build prompt for LLM-assisted synthesis."""
        return f"""Analyze this task and suggest optimal agent configuration:

Task: {description}

Current analysis:
- Complexity: {heuristic_result.task_analysis.complexity.value}
- Domains: {', '.join(d.value for d in heuristic_result.task_analysis.domains)}
- Current spec: {heuristic_result.spec.agent_type}
- Plan mode: {heuristic_result.spec.enable_plan_mode}
- Browsing: {heuristic_result.spec.enable_browsing}
- Jupyter: {heuristic_result.spec.enable_jupyter}

Should any settings be adjusted? Respond with JSON:
{{"adjustments": {{"field": "value"}}, "reasoning": "..."}}
"""

    def _parse_llm_response(
        self, response: str, current_spec: AgentSpec
    ) -> AgentSpec:
        """Parse LLM response to refine spec."""
        import json

        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                adjustments = data.get('adjustments', {})

                # Apply adjustments
                for field, value in adjustments.items():
                    if hasattr(current_spec, field):
                        setattr(current_spec, field, value)

        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f'Could not parse LLM response: {e}')

        return current_spec
