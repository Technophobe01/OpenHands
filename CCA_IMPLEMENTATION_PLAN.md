# CCA (Confucius Code Agent) Enhancement Implementation Plan

This document outlines the plan to integrate CCA architecture improvements into OpenHands.

## Reference

- Paper: arXiv:2512.10398 (Confucius Code Agent)
- Target: Improve OpenHands from 72.8% to ~74.6% on SWE-Bench Verified

## Architecture Mapping

| CCA Component | OpenHands Equivalent | Enhancement Required |
|---------------|---------------------|---------------------|
| Hierarchical working memory | `memory/condenser/` | Replace with hierarchy |
| Architect planner | `AgentSummarizeAction` | Structured plan output |
| Note-taking agent | None | New module |
| Extensions with callbacks | `events/` + tools | Add typed callbacks |
| Meta-agent | None | New module |
| AX/UX separation | Partial in events | Cleaner separation |

## Directory Structure (New/Modified)

```
openhands/
├── memory/
│   ├── hierarchical/           # NEW: Hierarchical memory system
│   │   ├── __init__.py
│   │   ├── working_memory.py   # Hierarchical working memory
│   │   ├── architect.py        # Architect planner for compression
│   │   ├── scopes.py          # Visibility scopes (session/entry/runnable)
│   │   └── storage.py         # Memory persistence
│   ├── condenser/              # MODIFIED: Enhanced condenser
│   │   └── impl/
│   │       └── hierarchical_condenser.py  # NEW
│   └── notes/                  # NEW: Note-taking system
│       ├── __init__.py
│       ├── note_store.py      # Markdown note storage
│       ├── note_taker.py      # Note-taking agent
│       └── hindsight.py       # Failure/hindsight notes
├── agenthub/
│   ├── architect_agent/       # NEW: Architect for context compression
│   ├── note_taker_agent/      # NEW: Note-taking agent
│   └── meta_agent/            # NEW: Meta-agent for agent synthesis
├── extensions/                 # NEW: Extension system
│   ├── __init__.py
│   ├── base.py                # Extension base class with callbacks
│   ├── registry.py            # Extension registration
│   ├── perception/            # Perception extensions
│   ├── reasoning/             # Reasoning extensions
│   └── action/                # Action extensions
└── ax_ux/                     # NEW: AX/UX separation layer
    ├── __init__.py
    ├── agent_view.py          # Compressed view for agent (AX)
    └── user_view.py           # Rich view for user (UX)
```

## Implementation Phases

### Phase 1: Foundation & Environment Setup
**Branch:** `feature/cca-phase1-foundation`

- [ ] Set up Docker development environment
- [ ] Verify OpenHands runs locally
- [ ] Run existing test suite
- [ ] Understand codebase deeply (read key files)
- [ ] Create CCA test suite framework

**Key Files to Study:**
- `openhands/memory/memory.py`
- `openhands/memory/condenser/condenser.py`
- `openhands/controller/agent_controller.py`
- `openhands/controller/state/state.py`
- `openhands/events/event.py`

### Phase 2: Hierarchical Memory System
**Branch:** `feature/cca-phase2-hierarchical-memory`
**PR Target:** Can be merged independently

**Tasks:**
- [ ] Design hierarchical memory schema
- [ ] Implement `HierarchicalWorkingMemory` class
- [ ] Implement visibility scopes (session, entry, runnable)
- [ ] Implement `ArchitectAgent` for structured summarization
- [ ] Create `HierarchicalCondenser` implementation
- [ ] Integrate with existing `State` class
- [ ] Add configuration options
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Benchmark against baseline

**Interface Design:**
```python
class HierarchicalWorkingMemory:
    """CCA F1: Hierarchical working memory with visibility scopes."""

    def __init__(self, scopes: list[MemoryScope]):
        self.root = MemoryNode(scope=MemoryScope.SESSION)

    def add_entry(self, content: str, scope: MemoryScope, metadata: dict) -> str:
        """Add content to memory with specified visibility scope."""

    def compress(self, architect: ArchitectAgent) -> CompressionResult:
        """Trigger architect-based compression when thresholds exceeded."""

    def get_context(self, max_tokens: int) -> list[Message]:
        """Get compressed context for LLM, respecting scope visibility."""
```

### Phase 3: Note-Taking Agent
**Branch:** `feature/cca-phase3-note-taking`
**PR Target:** Can be merged independently

**Tasks:**
- [ ] Design note storage format (Markdown with frontmatter)
- [ ] Implement `NoteStore` for file-system storage
- [ ] Implement `NoteTakerAgent`
- [ ] Implement hindsight notes for failures
- [ ] Add cross-session retrieval
- [ ] Integrate with agent lifecycle
- [ ] Write tests
- [ ] Document usage

**Note Structure:**
```markdown
---
id: unique_note_id
title: Note Title
keywords: [keyword1, keyword2]
type: solution|failure|pattern|architecture
created: 2025-01-11T20:00:00Z
session_id: xxx
---

## Problem Context
[Description]

## Solution/Insight
[Details]

## Related Files
- /path/to/file.py
```

### Phase 4: Extension System
**Branch:** `feature/cca-phase4-extensions`
**PR Target:** Larger refactor, discuss with maintainers first

**Tasks:**
- [ ] Design extension callback interface
- [ ] Implement `Extension` base class
- [ ] Implement `ExtensionRegistry`
- [ ] Create callback hooks:
  - `on_input_messages` - Pre-LLM prompt shaping
  - `on_llm_output` - Post-LLM output processing
  - `on_tag` - XML tag parsing
  - `on_action` - Action execution hooks
- [ ] Refactor existing tools as extensions
- [ ] Add extension composition
- [ ] Write tests
- [ ] Migration guide for existing tools

**Interface Design:**
```python
class Extension(ABC):
    """CCA F3: Modular extension with typed callbacks."""

    name: str
    category: ExtensionCategory  # PERCEPTION | REASONING | ACTION

    def on_input_messages(self, messages: list[Message], ctx: RunContext) -> list[Message]:
        """Hook to shape messages before LLM call."""
        return messages

    def on_llm_output(self, output: str, ctx: RunContext) -> str:
        """Hook to process LLM output."""
        return output

    def on_tag(self, tag: str, content: str, ctx: RunContext) -> Action | None:
        """Hook to parse XML-style tags into actions."""
        return None

    def on_action(self, action: Action, ctx: RunContext) -> Observation:
        """Hook to execute action and return observation."""
        raise NotImplementedError
```

### Phase 5: Meta-Agent
**Branch:** `feature/cca-phase5-meta-agent`
**PR Target:** Advanced feature, may stay as fork-only initially

**Tasks:**
- [ ] Design meta-agent specification format
- [ ] Implement build-test-improve loop
- [ ] Implement agent configuration synthesis
- [ ] Implement extension wiring
- [ ] Add evaluation framework integration
- [ ] Write tests
- [ ] Document meta-agent usage

### Phase 6: AX/UX Separation
**Branch:** `feature/cca-phase6-ax-ux`
**PR Target:** Discuss with maintainers

**Tasks:**
- [ ] Design AX (agent) vs UX (user) view interfaces
- [ ] Implement `AgentView` (compressed, structured)
- [ ] Implement `UserView` (rich, streaming)
- [ ] Refactor event stream to support dual views
- [ ] Update UI to use UserView
- [ ] Update agent to use AgentView
- [ ] Write tests

## Development Environment Setup

### Docker Setup (Mac Host -> Container)

```dockerfile
# Dockerfile.dev
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install Node 22
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs

# Install Poetry
RUN pip install poetry==2.1.2

WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy source
COPY . .

EXPOSE 3000 3001 8080

CMD ["make", "run"]
```

### docker-compose.dev.yml

```yaml
version: '3.8'
services:
  openhands-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/workspace
      - /var/run/docker.sock:/var/run/docker.sock
    ports:
      - "3000:3000"
      - "3001:3001"
      - "8080:8080"
    environment:
      - LLM_API_KEY=${LLM_API_KEY}
      - LLM_MODEL=${LLM_MODEL:-anthropic/claude-sonnet-4-20250514}
    stdin_open: true
    tty: true
```

## Testing Strategy

### Unit Tests
- Each new module has corresponding test file
- Mock LLM calls for deterministic testing
- Test edge cases (empty memory, max context, etc.)

### Integration Tests
- Test hierarchical memory with real condenser
- Test note-taking with file system
- Test extension composition

### Benchmark Tests
- Run on SWE-Bench-Lite subset (50 tasks)
- Compare before/after each phase
- Track: resolve rate, tokens used, turns, latency

## PR Strategy

1. **Phase 1-2**: Can be PR'd as "Enhanced memory system"
2. **Phase 3**: Can be PR'd as "Note-taking for cross-session learning"
3. **Phase 4**: Larger discussion needed - may need RFC
4. **Phase 5-6**: May stay as fork features initially

## Configuration

Add to `config.toml`:

```toml
[memory]
# CCA hierarchical memory settings
hierarchical_enabled = true
compression_threshold = 0.8  # Trigger at 80% context usage
architect_model = "anthropic/claude-haiku-3-20240307"  # Cheaper model for compression

[notes]
enabled = true
storage_path = ".openhands/notes"
hindsight_enabled = true  # Record failure notes

[extensions]
enabled = true
# List of enabled extensions
active = ["file_edit", "bash", "code_search", "planning"]
```

## Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| SWE-Bench Verified | 72.8% | 74.6%+ |
| Avg tokens per task | ~100k | ~90k |
| Avg turns per task | ~60 | ~55 |
| Cross-session improvement | N/A | +1.4% |

## Timeline Estimate

- Phase 1: 1-2 weeks (environment, deep code study)
- Phase 2: 2-3 weeks (hierarchical memory)
- Phase 3: 1-2 weeks (note-taking)
- Phase 4: 2-3 weeks (extensions)
- Phase 5: 2 weeks (meta-agent)
- Phase 6: 1-2 weeks (AX/UX separation)

**Total: 9-14 weeks for full implementation**

## Next Steps

1. Create GitHub fork at github.com/YOUR_USERNAME/OpenHands
2. Add fork as origin: `git remote add origin git@github.com:YOUR_USERNAME/OpenHands.git`
3. Set up Docker dev environment
4. Run `make build` and verify baseline
5. Start Phase 1 deep code study
