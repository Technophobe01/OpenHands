"""Architect Planner for hierarchical memory compression.

CCA F1: The Architect analyzes the memory hierarchy and produces
structured summaries that preserve important context while reducing tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.memory.hierarchical.scopes import COMPRESSION_ORDER, MemoryScope
from openhands.memory.hierarchical.working_memory import (
    HierarchicalWorkingMemory,
    MemoryEntry,
)

if TYPE_CHECKING:
    from openhands.llm.llm import LLM


@dataclass
class CompressionPlan:
    """Plan for compressing hierarchical memory."""

    entries_to_compress: list[MemoryEntry]
    target_token_reduction: int
    scope_summaries: dict[MemoryScope, str]


@dataclass
class CompressionResult:
    """Result of architect compression."""

    compressed_entries: list[MemoryEntry]
    summaries: dict[MemoryScope, str]
    tokens_before: int
    tokens_after: int
    llm_calls: int


class Architect:
    """CCA F1: Architect planner for hierarchical memory compression.

    The Architect analyzes the memory hierarchy and determines:
    1. Which entries to compress (lower scopes first, lower importance first)
    2. How to summarize them (preserving key information)
    3. Where to insert summaries in the hierarchy

    This enables intelligent context management that preserves critical
    information while staying within token limits.
    """

    # Prompt template for scope-aware summarization
    SUMMARIZE_PROMPT = """You are an Architect maintaining hierarchical context for a coding agent.

You will receive entries from the agent's working memory at scope level: {scope_name}

Scope levels (from most to least persistent):
- SESSION: Repository structure, user requirements, architecture decisions
- ENTRY: Current task context, files being worked on
- RUNNABLE: Immediate action outputs, command results

Your task: Summarize the {scope_name} entries while preserving:
1. Key information needed for the agent's task
2. Important state changes or decisions
3. Error information that might be relevant
4. File paths and function names mentioned

Keep the summary concise but complete. Format:

<{scope_name}_SUMMARY>
[Your summary here - preserve critical details, remove redundancy]
</{scope_name}_SUMMARY>

Entries to summarize:
{entries}"""

    def __init__(
        self,
        llm: LLM,
        target_reduction_ratio: float = 0.5,
        min_entries_per_summary: int = 3,
        max_summary_tokens: int = 500,
    ):
        """Initialize the Architect.

        Args:
            llm: LLM to use for summarization
            target_reduction_ratio: Target ratio of tokens to keep (0.5 = reduce by half)
            min_entries_per_summary: Minimum entries before creating a summary
            max_summary_tokens: Maximum tokens for each scope summary
        """
        self.llm = llm
        self.target_reduction_ratio = target_reduction_ratio
        self.min_entries_per_summary = min_entries_per_summary
        self.max_summary_tokens = max_summary_tokens

    def plan_compression(
        self, memory: HierarchicalWorkingMemory
    ) -> CompressionPlan | None:
        """Create a compression plan for the memory.

        Returns None if compression is not needed or not possible.
        """
        if not memory.should_compress():
            return None

        current_tokens = memory.root.total_tokens_estimate()
        target_tokens = int(current_tokens * self.target_reduction_ratio)
        target_reduction = current_tokens - target_tokens

        # Get entries to compress in order
        entries_to_compress = memory.get_entries_to_compress()

        if len(entries_to_compress) < self.min_entries_per_summary:
            logger.debug(
                f'Not enough entries to compress: {len(entries_to_compress)} < {self.min_entries_per_summary}'
            )
            return None

        return CompressionPlan(
            entries_to_compress=entries_to_compress,
            target_token_reduction=target_reduction,
            scope_summaries={},
        )

    def compress(self, memory: HierarchicalWorkingMemory) -> CompressionResult | None:
        """Compress the memory using hierarchical summarization.

        Compresses lower scopes first, creating summaries that preserve
        important context while reducing token count.
        """
        plan = self.plan_compression(memory)
        if plan is None:
            return None

        tokens_before = memory.root.total_tokens_estimate()
        llm_calls = 0
        compressed_entries: list[MemoryEntry] = []
        scope_summaries: dict[MemoryScope, str] = {}

        # Compress by scope, starting with most transient (RUNNABLE)
        for scope in COMPRESSION_ORDER:
            scope_entries = [e for e in plan.entries_to_compress if e.scope == scope]

            if len(scope_entries) < self.min_entries_per_summary:
                continue

            # Generate summary for this scope
            summary = self._summarize_scope(scope, scope_entries)
            llm_calls += 1

            # Mark entries as compressed
            for entry in scope_entries:
                entry.compress(summary)
                compressed_entries.append(entry)

            scope_summaries[scope] = summary

            # Check if we've reduced enough
            current_tokens = memory.root.total_tokens_estimate()
            if current_tokens <= tokens_before * self.target_reduction_ratio:
                break

        tokens_after = memory.root.total_tokens_estimate()

        logger.info(
            f'Architect compression: {tokens_before} -> {tokens_after} tokens '
            f'({llm_calls} LLM calls, {len(compressed_entries)} entries compressed)'
        )

        return CompressionResult(
            compressed_entries=compressed_entries,
            summaries=scope_summaries,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            llm_calls=llm_calls,
        )

    def _summarize_scope(self, scope: MemoryScope, entries: list[MemoryEntry]) -> str:
        """Generate a summary for entries at a specific scope."""
        # Format entries for the prompt
        entries_text = '\n\n'.join(
            f'<entry id="{e.entry_id}" importance="{e.importance}">\n{e.content}\n</entry>'
            for e in entries
        )

        prompt = self.SUMMARIZE_PROMPT.format(
            scope_name=scope.name, entries=entries_text
        )

        messages = [Message(role='user', content=[TextContent(text=prompt)])]

        response = self.llm.completion(
            messages=self.llm.format_messages_for_llm(messages),
        )

        summary = response.choices[0].message.content or ''

        # Extract just the summary content from the tags
        start_tag = f'<{scope.name}_SUMMARY>'
        end_tag = f'</{scope.name}_SUMMARY>'
        if start_tag in summary and end_tag in summary:
            start = summary.index(start_tag) + len(start_tag)
            end = summary.index(end_tag)
            summary = summary[start:end].strip()

        return summary

    def get_compression_stats(
        self, memory: HierarchicalWorkingMemory
    ) -> dict[str, int | float]:
        """Get statistics about the current memory state."""
        all_entries = memory.get_all_entries()
        compressed_count = sum(1 for e in all_entries if e.compressed)

        return {
            'total_entries': len(all_entries),
            'compressed_entries': compressed_count,
            'session_entries': sum(1 for e in all_entries if e.scope == MemoryScope.SESSION),
            'entry_entries': sum(1 for e in all_entries if e.scope == MemoryScope.ENTRY),
            'runnable_entries': sum(1 for e in all_entries if e.scope == MemoryScope.RUNNABLE),
            'estimated_tokens': memory.root.total_tokens_estimate(),
            'compression_ratio': compressed_count / len(all_entries) if all_entries else 0,
        }
