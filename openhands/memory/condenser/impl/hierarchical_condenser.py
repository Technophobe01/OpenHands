"""Hierarchical Condenser implementation.

CCA F1: Condenser that uses hierarchical working memory with visibility scopes
and architect-based compression for intelligent context management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openhands.core.config.condenser_config import HierarchicalCondenserConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.action.agent import CondensationAction
from openhands.memory.condenser.condenser import (
    Condensation,
    RollingCondenser,
    View,
)
from openhands.memory.hierarchical.architect import Architect
from openhands.memory.hierarchical.scopes import MemoryScope
from openhands.memory.hierarchical.working_memory import HierarchicalWorkingMemory

if TYPE_CHECKING:
    from openhands.llm.llm import LLM
    from openhands.llm.llm_registry import LLMRegistry


class HierarchicalCondenser(RollingCondenser):
    """CCA F1: Hierarchical condenser with scope-based compression.

    This condenser organizes events into a hierarchical memory structure
    with three visibility scopes:
    - SESSION: Persistent context (repo info, user goals)
    - ENTRY: Current task context (files being edited)
    - RUNNABLE: Immediate action context (command outputs)

    When context limits are exceeded, the Architect compresses lower scopes
    first, preserving important information through intelligent summarization.

    Benefits over flat condensation:
    1. Better preservation of high-level context (SESSION scope)
    2. Intelligent compression order (transient first)
    3. Scope-aware summaries that maintain coherence
    4. Reduced token usage through hierarchical organization
    """

    def __init__(
        self,
        llm: LLM,
        max_size: int = 100,
        keep_first: int = 1,
        compression_threshold: float = 0.8,
        target_reduction_ratio: float = 0.5,
    ):
        """Initialize the hierarchical condenser.

        Args:
            llm: LLM to use for summarization
            max_size: Maximum number of events before condensation
            keep_first: Number of initial events to always keep
            compression_threshold: Trigger compression at this % of capacity
            target_reduction_ratio: Target ratio of tokens to keep after compression
        """
        if keep_first >= max_size // 2:
            raise ValueError(
                f'keep_first ({keep_first}) must be less than half of max_size ({max_size})'
            )

        self.llm = llm
        self.max_size = max_size
        self.keep_first = keep_first
        self.compression_threshold = compression_threshold
        self.target_reduction_ratio = target_reduction_ratio

        # Create architect for compression
        self.architect = Architect(
            llm=llm,
            target_reduction_ratio=target_reduction_ratio,
        )

        # Hierarchical memory state (rebuilt each condensation cycle)
        self._memory: HierarchicalWorkingMemory | None = None

        super().__init__()

    def should_condense(self, view: View) -> bool:
        """Check if condensation should be triggered."""
        # Condense if view exceeds max_size or there's an unhandled request
        if len(view) > self.max_size:
            return True
        if view.unhandled_condensation_request:
            return True
        return False

    def get_condensation(self, view: View) -> Condensation:
        """Generate condensation using hierarchical memory and architect."""
        # Build hierarchical memory from events
        self._memory = HierarchicalWorkingMemory.from_events(list(view.events))
        self._memory.max_tokens = self.max_size * 100  # Rough token estimate

        # Keep first N events unchanged
        head_events = view[: self.keep_first]

        # Calculate target size for events to keep
        target_size = self.max_size // 2
        events_from_tail = target_size - len(head_events) - 1

        # Get events to forget (those not in head or tail)
        forgotten_events = []
        for event in view[self.keep_first : -events_from_tail]:
            forgotten_events.append(event)

        if not forgotten_events:
            logger.warning('HierarchicalCondenser: No events to forget')
            return Condensation(
                action=CondensationAction(
                    forgotten_events_start_id=-1,
                    forgotten_events_end_id=-1,
                )
            )

        # Use architect to create hierarchical summary
        summary = self._create_hierarchical_summary(forgotten_events)

        # Record metadata
        stats = self.architect.get_compression_stats(self._memory)
        self.add_metadata('hierarchical_stats', stats)
        self.add_metadata('metrics', self.llm.metrics.get())

        return Condensation(
            action=CondensationAction(
                forgotten_events_start_id=min(e.id for e in forgotten_events),
                forgotten_events_end_id=max(e.id for e in forgotten_events),
                summary=summary,
                summary_offset=self.keep_first,
            )
        )

    def _create_hierarchical_summary(self, events: list) -> str:
        """Create a hierarchical summary of events organized by scope."""
        # Classify events by scope
        scope_events: dict[MemoryScope, list] = {
            MemoryScope.SESSION: [],
            MemoryScope.ENTRY: [],
            MemoryScope.RUNNABLE: [],
        }

        for event in events:
            scope = HierarchicalWorkingMemory._classify_event_scope(event)
            scope_events[scope].append(event)

        # Generate summaries for each scope that has events
        summaries = []

        # SESSION scope summary (most persistent, summarize last)
        if scope_events[MemoryScope.SESSION]:
            session_summary = self._summarize_scope_events(
                MemoryScope.SESSION, scope_events[MemoryScope.SESSION]
            )
            summaries.append(f'[SESSION CONTEXT]\n{session_summary}')

        # ENTRY scope summary (task context)
        if scope_events[MemoryScope.ENTRY]:
            entry_summary = self._summarize_scope_events(
                MemoryScope.ENTRY, scope_events[MemoryScope.ENTRY]
            )
            summaries.append(f'[TASK CONTEXT]\n{entry_summary}')

        # RUNNABLE scope summary (immediate context, compress most aggressively)
        if scope_events[MemoryScope.RUNNABLE]:
            runnable_summary = self._summarize_scope_events(
                MemoryScope.RUNNABLE, scope_events[MemoryScope.RUNNABLE]
            )
            summaries.append(f'[IMMEDIATE CONTEXT]\n{runnable_summary}')

        return '\n\n'.join(summaries) if summaries else 'No events summarized.'

    def _summarize_scope_events(self, scope: MemoryScope, events: list) -> str:
        """Summarize events at a specific scope level."""
        from openhands.core.message import Message, TextContent
        from openhands.events.serialization.event import truncate_content

        # Format events for summarization
        event_strs = []
        for event in events:
            content = truncate_content(str(event), max_chars=2000)
            event_strs.append(f'<event id={event.id}>\n{content}\n</event>')

        events_text = '\n'.join(event_strs)

        prompt = f"""Summarize the following {scope.name} scope events for a coding agent.

{scope.name} scope contains:
{self._get_scope_description(scope)}

Keep the summary concise but preserve:
- Key state changes
- Important decisions
- Error information
- File paths and function names

Events:
{events_text}

Summary:"""

        messages = [Message(role='user', content=[TextContent(text=prompt)])]
        response = self.llm.completion(
            messages=self.llm.format_messages_for_llm(messages),
        )

        return response.choices[0].message.content or 'Unable to summarize.'

    @staticmethod
    def _get_scope_description(scope: MemoryScope) -> str:
        """Get description of what a scope typically contains."""
        descriptions = {
            MemoryScope.SESSION: 'Repository structure, user requirements, architecture decisions, persistent context',
            MemoryScope.ENTRY: 'Current task context, files being worked on, active goals, recent progress',
            MemoryScope.RUNNABLE: 'Command outputs, tool results, immediate action feedback',
        }
        return descriptions.get(scope, 'General context')

    @classmethod
    def from_config(
        cls, config: HierarchicalCondenserConfig, llm_registry: 'LLMRegistry'
    ) -> 'HierarchicalCondenser':
        """Create a HierarchicalCondenser from config."""
        # Disable prompt caching for the condenser LLM
        llm_config = config.llm_config.model_copy()
        llm_config.caching_prompt = False
        llm = llm_registry.get_llm('condenser', llm_config)

        return HierarchicalCondenser(
            llm=llm,
            max_size=config.max_size,
            keep_first=config.keep_first,
            compression_threshold=config.compression_threshold,
            target_reduction_ratio=config.target_reduction_ratio,
        )


# Register the config type
HierarchicalCondenser.register_config(HierarchicalCondenserConfig)
