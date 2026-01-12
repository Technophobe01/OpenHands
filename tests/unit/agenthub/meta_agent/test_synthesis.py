"""Tests for agent configuration synthesis."""

import pytest

from openhands.agenthub.meta_agent.spec import (
    AgentSpec,
    TaskAnalysis,
    TaskComplexity,
    TaskDomain,
)
from openhands.agenthub.meta_agent.synthesis import (
    BROWSING_KEYWORDS,
    COMPLEX_INDICATORS,
    JUPYTER_KEYWORDS,
    SYSTEM_KEYWORDS,
    AgentSynthesizer,
)


class TestAgentSynthesizer:
    """Tests for AgentSynthesizer."""

    @pytest.fixture
    def synthesizer(self):
        """Create synthesizer without LLM."""
        return AgentSynthesizer()

    def test_init_without_llm(self, synthesizer):
        """Test initialization without LLM."""
        assert synthesizer.llm is None

    def test_analyze_simple_task(self, synthesizer):
        """Test analysis of simple task."""
        analysis = synthesizer.analyze_task('Fix a simple typo in README.md')

        assert analysis.complexity == TaskComplexity.SIMPLE
        assert analysis.requires_file_editing is True

    def test_analyze_complex_task(self, synthesizer):
        """Test analysis of complex task."""
        description = (
            'Refactor the entire authentication system to use OAuth2. '
            'This requires updating the database schema, implementing new '
            'API endpoints, and integrating with external providers.'
        )
        analysis = synthesizer.analyze_task(description)

        assert analysis.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXPERT)
        assert analysis.requires_planning is True

    def test_analyze_browsing_task(self, synthesizer):
        """Test analysis of browsing task."""
        analysis = synthesizer.analyze_task(
            'Fetch the latest documentation from the website and summarize it'
        )

        assert analysis.requires_browsing is True
        assert TaskDomain.BROWSING in analysis.domains

    def test_analyze_data_task(self, synthesizer):
        """Test analysis of data/jupyter task."""
        analysis = synthesizer.analyze_task(
            'Analyze the CSV data file using pandas and create visualizations'
        )

        assert analysis.requires_jupyter is True
        assert TaskDomain.DATA in analysis.domains

    def test_analyze_system_task(self, synthesizer):
        """Test analysis of system task."""
        analysis = synthesizer.analyze_task(
            'Install Docker and configure the Kubernetes deployment'
        )

        assert TaskDomain.SYSTEM in analysis.domains

    def test_analyze_readonly_task(self, synthesizer):
        """Test analysis of read-only task."""
        analysis = synthesizer.analyze_task(
            'Find all usages of the deprecated function and list them'
        )

        assert analysis.requires_file_editing is False

    def test_analyze_extracts_keywords(self, synthesizer):
        """Test that keywords are extracted."""
        analysis = synthesizer.analyze_task(
            'Implement user authentication with JWT tokens'
        )

        assert len(analysis.keywords) > 0
        assert 'authentication' in analysis.keywords or 'jwt' in analysis.keywords

    def test_synthesize_returns_result(self, synthesizer):
        """Test synthesize returns SynthesisResult."""
        result = synthesizer.synthesize('Fix a bug in the login function')

        assert result.spec is not None
        assert result.task_analysis is not None
        assert 0 <= result.confidence <= 1

    def test_synthesize_simple_task(self, synthesizer):
        """Test synthesis for simple task."""
        result = synthesizer.synthesize('Quick fix: rename variable from x to count')

        assert result.spec.agent_type == 'CodeActAgent'
        assert result.spec.enable_plan_mode is False
        assert result.spec.condenser_max_size <= 100

    def test_synthesize_complex_task(self, synthesizer):
        """Test synthesis for complex task."""
        result = synthesizer.synthesize(
            'Implement a complete user registration system with email '
            'verification, password reset, and OAuth integration'
        )

        assert result.spec.enable_plan_mode is True
        assert result.spec.condenser_max_size >= 100

    def test_synthesize_browsing_task(self, synthesizer):
        """Test synthesis enables browsing for web tasks."""
        result = synthesizer.synthesize(
            'Fetch the API documentation from the website and implement the client'
        )

        assert result.spec.enable_browsing is True

    def test_synthesize_readonly_task(self, synthesizer):
        """Test synthesis for read-only tasks."""
        result = synthesizer.synthesize(
            'Analyze the codebase and explain the architecture'
        )

        assert result.spec.agent_type == 'ReadOnlyAgent'
        assert result.task_analysis.requires_file_editing is False

    def test_synthesize_generates_reasoning(self, synthesizer):
        """Test that reasoning is generated."""
        result = synthesizer.synthesize('Implement new feature')

        assert len(result.reasoning) > 0
        assert 'complexity' in result.reasoning.lower() or 'domain' in result.reasoning.lower()

    def test_synthesize_sets_spec_name(self, synthesizer):
        """Test that spec name is set."""
        result = synthesizer.synthesize('Test task')

        assert 'synthesized_' in result.spec.name

    def test_synthesize_sets_description(self, synthesizer):
        """Test that spec description includes task info."""
        result = synthesizer.synthesize('Implement login feature')

        assert 'Auto-synthesized' in result.spec.description


class TestComplexityEstimation:
    """Tests for complexity estimation."""

    @pytest.fixture
    def synthesizer(self):
        return AgentSynthesizer()

    def test_simple_indicators(self, synthesizer):
        """Test simple task indicators."""
        analysis = synthesizer.analyze_task('Simple fix for a typo')
        assert analysis.complexity == TaskComplexity.SIMPLE

    def test_complex_indicators(self, synthesizer):
        """Test complex task indicators."""
        analysis = synthesizer.analyze_task(
            'Refactor the entire module and implement new features'
        )
        assert analysis.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EXPERT)

    def test_expert_indicators(self, synthesizer):
        """Test expert task indicators."""
        analysis = synthesizer.analyze_task(
            'Design the system architecture and migrate the entire database'
        )
        assert analysis.complexity == TaskComplexity.EXPERT

    def test_word_count_affects_complexity(self, synthesizer):
        """Test that word count affects complexity."""
        short_analysis = synthesizer.analyze_task('Fix bug')
        long_analysis = synthesizer.analyze_task(
            ' '.join(['word'] * 150)  # Very long description
        )

        assert short_analysis.complexity.value <= long_analysis.complexity.value


class TestDomainDetection:
    """Tests for domain detection."""

    @pytest.fixture
    def synthesizer(self):
        return AgentSynthesizer()

    def test_coding_domain(self, synthesizer):
        """Test coding domain detection."""
        analysis = synthesizer.analyze_task('Fix the bug in the function')
        assert TaskDomain.CODING in analysis.domains

    def test_browsing_domain(self, synthesizer):
        """Test browsing domain detection."""
        for keyword in BROWSING_KEYWORDS[:3]:
            analysis = synthesizer.analyze_task(f'Do something with {keyword}')
            assert TaskDomain.BROWSING in analysis.domains, f'Failed for {keyword}'

    def test_data_domain(self, synthesizer):
        """Test data domain detection."""
        for keyword in JUPYTER_KEYWORDS[:3]:
            analysis = synthesizer.analyze_task(f'Work with {keyword}')
            assert TaskDomain.DATA in analysis.domains, f'Failed for {keyword}'

    def test_system_domain(self, synthesizer):
        """Test system domain detection."""
        for keyword in SYSTEM_KEYWORDS[:3]:
            analysis = synthesizer.analyze_task(f'Configure {keyword}')
            assert TaskDomain.SYSTEM in analysis.domains, f'Failed for {keyword}'

    def test_analysis_domain(self, synthesizer):
        """Test analysis domain detection."""
        analysis = synthesizer.analyze_task('Analyze the code and review it')
        assert TaskDomain.ANALYSIS in analysis.domains

    def test_mixed_domain(self, synthesizer):
        """Test mixed domain detection."""
        analysis = synthesizer.analyze_task(
            'Analyze the web data using jupyter notebooks and deploy to docker'
        )
        # When many domains, should collapse to MIXED
        # Or keep individual domains if <= 2
        assert len(analysis.domains) >= 1

    def test_default_to_coding(self, synthesizer):
        """Test default to coding when no domain detected."""
        analysis = synthesizer.analyze_task('Do something unspecified')
        assert TaskDomain.CODING in analysis.domains


class TestKeywordExtraction:
    """Tests for keyword extraction."""

    @pytest.fixture
    def synthesizer(self):
        return AgentSynthesizer()

    def test_extracts_meaningful_keywords(self, synthesizer):
        """Test extraction of meaningful keywords."""
        analysis = synthesizer.analyze_task(
            'Implement user authentication with JWT tokens'
        )

        # Should extract meaningful words, not stop words
        assert 'the' not in analysis.keywords
        assert 'implement' in analysis.keywords or 'user' in analysis.keywords

    def test_deduplicates_keywords(self, synthesizer):
        """Test that keywords are deduplicated."""
        analysis = synthesizer.analyze_task(
            'The user user user should authenticate'
        )

        # Should only have one 'user'
        assert analysis.keywords.count('user') <= 1

    def test_limits_keywords(self, synthesizer):
        """Test that keywords are limited."""
        analysis = synthesizer.analyze_task(' '.join([f'word{i}' for i in range(100)]))

        assert len(analysis.keywords) <= 20


class TestStepEstimation:
    """Tests for step estimation."""

    @pytest.fixture
    def synthesizer(self):
        return AgentSynthesizer()

    def test_simple_task_steps(self, synthesizer):
        """Test step estimation for simple task."""
        analysis = synthesizer.analyze_task('Simple typo fix')
        assert analysis.estimated_steps <= 10

    def test_complex_task_steps(self, synthesizer):
        """Test step estimation for complex task."""
        analysis = synthesizer.analyze_task(
            'Refactor the entire authentication system'
        )
        assert analysis.estimated_steps >= 20

    def test_expert_task_steps(self, synthesizer):
        """Test step estimation for expert task."""
        analysis = synthesizer.analyze_task(
            'Design system architecture and migrate database'
        )
        assert analysis.estimated_steps >= 50
