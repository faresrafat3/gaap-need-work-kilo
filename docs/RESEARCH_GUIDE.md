# Deep Discovery Engine Guide

## Overview

The Deep Discovery Engine (DDE) is an inference-first research system that builds provable knowledge. It combines web search, source verification, content extraction, and LLM-powered synthesis to produce high-quality research findings.

### Key Features

- **Multi-Provider Search**: DuckDuckGo (free), Serper, Perplexity, Brave
- **Epistemic Trust Scoring (ETS)**: Automated source credibility evaluation
- **Citation Mapping**: Follow citations to primary sources
- **Hypothesis Verification**: Cross-validate claims across multiple sources
- **Knowledge Graph Integration**: Permanent storage with no TTL
- **Configurable Depth**: 5 depth levels from quick search to deep dive

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepDiscoveryEngine                       │
│                     (Main Orchestrator)                      │
└─────────────────────────────────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────────┐
│ WebFetcher  │─────▶│SourceAuditor│─────▶│ContentExtractor │
│  (Search)   │      │ (ETS Score) │      │   (Clean Text)  │
└─────────────┘      └─────────────┘      └─────────────────┘
       │                      │                      │
       │                      │                      │
       ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────┐
│                        DeepDive                              │
│              (Citation Mapping + Cross-Validation)           │
└─────────────────────────────────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────────┐
│ Synthesizer │      │ Hypothesis  │      │  Associative    │
│  (LLM)      │─────▶│ Verification│─────▶│    Triples      │
└─────────────┘      └─────────────┘      └─────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ KnowledgeIntegrator │
                   │   (Permanent Store) │
                   └─────────────────────┘
```

---

## Components

### 1. WebFetcher

Web search and content retrieval with multiple provider support.

#### Providers

| Provider | API Key Required | Rate Limit | Notes |
|----------|-----------------|------------|-------|
| **DuckDuckGo** | No | 2/sec | Free, default provider |
| Serper | Yes | Varies | Google search results |
| Perplexity | Yes | Varies | AI-powered search |
| Brave | Yes | Varies | Privacy-focused search |

#### Usage

```python
from gaap.research import WebFetcher, WebFetcherConfig

# DuckDuckGo (free)
fetcher = WebFetcher(WebFetcherConfig(
    provider="duckduckgo",
    max_results=10,
))

# Serper (Google)
fetcher = WebFetcher(WebFetcherConfig(
    provider="serper",
    api_key="your-serper-api-key",
    max_results=20,
))

# Search
results = await fetcher.search("Python async best practices")
for result in results:
    print(f"{result.title}: {result.url}")

# Fetch content
content = await fetcher.fetch_content(results[0].url)

# Batch fetch
contents = await fetcher.fetch_batch([r.url for r in results[:5]])
```

---

### 2. SourceAuditor (ETS Scoring)

Epistemic Trust Score evaluation for source credibility.

#### ETS Scoring Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| Domain Reputation | 50% | Official docs > Community > Blogs |
| Content Freshness | 20% | Recent content scores higher |
| Author Credibility | 30% | Identified author boosts score |
| Citation Count | Bonus | +0.02 per citation (max +0.1) |

#### ETS Levels

| Level | Score | Source Types |
|-------|-------|--------------|
| **VERIFIED** | 0.95-1.0 | Official documentation, verified GitHub repos |
| **RELIABLE** | 0.65-0.94 | Peer-reviewed papers, Stack Overflow, Wikipedia |
| **QUESTIONABLE** | 0.45-0.64 | Medium articles, tutorials, tech blogs |
| **UNRELIABLE** | 0.20-0.44 | Random blogs, AI summaries |
| **BLACKLISTED** | 0.0-0.19 | Known bad sources |

#### Default Domain Scores

```python
# VERIFIED (1.0)
"docs.python.org": 1.0
"fastapi.tiangolo.com": 1.0
"kubernetes.io": 1.0

# RELIABLE (0.7-0.9)
"github.com": 0.9
"stackoverflow.com": 0.75
"arxiv.org": 0.9
"wikipedia.org": 0.7

# QUESTIONABLE (0.4-0.6)
"medium.com": 0.5
"dev.to": 0.55
"geeksforgeeks.org": 0.5

# UNRELIABLE (0.2-0.4)
"blogspot.com": 0.3
"substack.com": 0.4
```

#### Usage

```python
from gaap.research import SourceAuditor, SourceAuditConfig

auditor = SourceAuditor(SourceAuditConfig(
    min_ets_threshold=0.3,
    check_author=True,
    check_date=True,
))

# Audit single source
result = auditor.audit(source)
print(f"ETS Score: {result.ets_score}")
print(f"ETS Level: {result.ets_level}")
print(f"Reasons: {result.reasons}")

# Audit batch with filtering
passed, filtered = auditor.audit_batch(sources)
print(f"Passed: {len(passed)}, Filtered: {len(filtered)}")

# Customize scores
auditor.set_domain_score("mydocs.company.com", 0.95)
auditor.add_blacklist_domain("spam-site.com")
auditor.add_whitelist_domain("trusted-source.com")
```

---

### 3. ContentExtractor

Clean text extraction from web pages.

#### Features

- HTML to clean text conversion
- Code block extraction
- Link extraction
- Metadata extraction (author, date)
- Main content detection (ignores nav, ads, etc.)

#### Usage

```python
from gaap.research import ContentExtractor, ContentExtractorConfig

extractor = ContentExtractor(ContentExtractorConfig(
    max_content_length=50000,
    min_content_length=100,
    extract_code_blocks=True,
    extract_links=True,
))

# Extract from URL
content = await extractor.extract("https://example.com/article")

print(content.title)           # Page title
print(content.content)         # Clean text content
print(content.author)          # Author if found
print(content.publish_date)    # Publication date
print(content.code_blocks)     # Extracted code blocks
print(content.links)           # Links found in content

# Batch extraction
contents = await extractor.extract_batch(urls, max_concurrent=5)
```

#### Extracted Content Structure

```python
@dataclass
class ExtractedContent:
    url: str
    title: str
    content: str
    author: str | None
    publish_date: date | None
    links: list[str]
    code_blocks: list[str]
    extraction_success: bool
    error: str | None
```

---

### 4. Synthesizer

LLM-powered hypothesis building and verification.

#### Features

- Claim extraction from content
- Hypothesis building from claims
- Cross-validation between sources
- Contradiction detection
- Knowledge triple extraction

#### Usage

```python
from gaap.research import Synthesizer, SynthesizerConfig

synthesizer = Synthesizer(
    llm_provider=provider,
    config=SynthesizerConfig(
        max_hypotheses=10,
        min_confidence_threshold=0.5,
        cross_validate_enabled=True,
        detect_contradictions=True,
    ),
)

# Extract claims from content
claims = await synthesizer.extract_claims(content, source)
for claim in claims:
    print(f"Claim: {claim.text}")
    print(f"Confidence: {claim.confidence}")

# Build hypothesis from claim
hypothesis = await synthesizer.build_hypothesis(claims[0], sources)
print(f"Hypothesis: {hypothesis.statement}")
print(f"Status: {hypothesis.status}")

# Verify hypothesis against sources
verified = await synthesizer.verify_hypothesis(hypothesis, all_sources)
print(f"Verified: {verified.is_verified}")
print(f"Confidence: {verified.confidence}")

# Extract knowledge triples
triples = await synthesizer.extract_triples(content, source)
for triple in triples:
    print(f"{triple.subject} -> {triple.predicate} -> {triple.object}")

# Find contradictions between hypotheses
contradictions = await synthesizer.find_contradictions(hypotheses)
for c in contradictions:
    print(f"Conflict: {c.claim1} vs {c.claim2}")
```

#### Hypothesis Verification Process

```
1. Extract claims from each source
2. Build formal hypothesis statement
3. Cross-validate against multiple sources
4. Classify each source as:
   - SUPPORTS
   - CONTRADICTS
   - NEUTRAL
5. Calculate confidence based on agreement
6. Assign final status:
   - VERIFIED (high confidence, no contradictions)
   - FALSIFIED (contradictions with high confidence sources)
   - CONFLICTED (mixed signals)
   - UNVERIFIED (insufficient evidence)
```

---

### 5. DeepDive

Citation mapping and cross-validation exploration.

#### Depth Levels

| Level | Description | Actions |
|-------|-------------|---------|
| **1** | Basic | Search + Content extraction |
| **2** | Standard | + Citation following |
| **3** | Deep | + Cross-validation + Hypothesis building |
| **4** | Extended | + Related topic exploration |
| **5** | Full | + Recursive exploration |

#### Usage

```python
from gaap.research import DeepDive, DeepDiveConfig

deep_dive = DeepDive(DeepDiveConfig(
    default_depth=3,
    max_depth=5,
    citation_follow_depth=2,
    max_sources_per_depth=20,
))

# Execute deep dive
result = await deep_dive.explore(
    query="FastAPI async best practices",
    depth=4,
)

print(f"Total sources: {len(result.sources)}")
print(f"Primary sources: {len(result.primary_sources)}")
print(f"Citations followed: {result.citations_followed}")
print(f"Cross-validation score: {result.cross_validation_score}")
```

#### Deep Dive Protocol

```
Step 1: Exploration
├── Web search for initial results
└── Extract content from top N sources

Step 2: Citation Mapping (depth >= 2)
├── Extract links from content
├── Filter valid citation links
└── Follow citations to primary sources

Step 3: Cross-Validation (depth >= 3)
├── Extract claims from each source
├── Build hypotheses
└── Verify against multiple sources

Step 4: Related Topics (depth >= 4)
├── Generate related queries
└── Search and integrate results

Step 5: Knowledge Extraction (depth = 5)
├── Extract associative triples
└── Store in knowledge graph
```

---

### 6. KnowledgeIntegrator

Permanent storage for research results (no TTL).

#### Storage Structure

```
research_findings/
├── finding_id_1/
│   ├── query
│   ├── sources[] → KnowledgeGraph nodes
│   ├── hypotheses[] → KnowledgeGraph nodes
│   └── triples[] → KnowledgeGraph edges
└── finding_id_2/
    └── ...
```

#### Features

- **No TTL**: Everything is kept permanently
- **Deduplication**: By content hash and URL
- **Knowledge Graph**: Sources, hypotheses, and entities as nodes
- **Triple Storage**: Subject-predicate-object relationships
- **Similarity Search**: Find related previous research

#### Usage

```python
from gaap.research import KnowledgeIntegrator, StorageConfig

integrator = KnowledgeIntegrator(
    knowledge_graph=kg_builder,
    sqlite_store=sqlite_store,
    config=StorageConfig(
        knowledge_graph_enabled=True,
        sqlite_cache_enabled=True,
        dedup_enabled=True,
        storage_path=".gaap/research",
    ),
)

# Store research finding
finding_id = await integrator.store_research(finding)

# Find similar existing research
existing = await integrator.find_similar("FastAPI async", threshold=0.8)
if existing:
    print(f"Found existing research: {existing.id}")

# Get research by topic
findings = await integrator.get_by_topic("async")

# Add triple to knowledge graph
await integrator.add_triple(
    subject="FastAPI",
    predicate="supports",
    object="async/await",
    source=source,
    confidence=0.95,
)
```

---

## Usage Examples

### Basic Research

```python
from gaap.research import DeepDiscoveryEngine, DDEConfig

# Create engine with default config
engine = DeepDiscoveryEngine()

# Execute research
result = await engine.research("FastAPI async best practices")

if result.success:
    print(f"Query: {result.query}")
    print(f"Sources: {len(result.finding.sources)}")
    print(f"Hypotheses: {len(result.finding.hypotheses)}")
    print(f"Triples: {len(result.finding.triples)}")
    print(f"Time: {result.total_time_ms:.0f}ms")
    
    # Access metrics
    print(f"Avg ETS Score: {result.metrics.avg_ets_score:.2f}")
    print(f"Hypotheses Verified: {result.metrics.hypotheses_verified}")
else:
    print(f"Research failed: {result.error}")
```

### Quick Search

```python
from gaap.research import DeepDiscoveryEngine

engine = DeepDiscoveryEngine()

# Quick search without deep analysis
sources = await engine.quick_search("Python async", max_results=5)
for source in sources:
    print(f"[{source.ets_score:.2f}] {source.title}")
    print(f"  {source.url}")
```

### Custom Configuration

```python
from gaap.research import (
    DeepDiscoveryEngine,
    DDEConfig,
    WebFetcherConfig,
    SourceAuditConfig,
    SynthesizerConfig,
)

# Custom configuration
config = DDEConfig(
    research_depth=4,
    max_total_sources=100,
    max_total_hypotheses=25,
    
    web_fetcher=WebFetcherConfig(
        provider="serper",
        api_key="your-serper-api-key",
        max_results=20,
    ),
    
    source_audit=SourceAuditConfig(
        min_ets_threshold=0.5,  # Higher quality threshold
        check_author=True,
        check_date=True,
    ),
    
    synthesizer=SynthesizerConfig(
        max_hypotheses=20,
        cross_validate_enabled=True,
        verification_confidence_threshold=0.8,
    ),
)

engine = DeepDiscoveryEngine(config=config, llm_provider=provider)
result = await engine.research("Microservices architecture patterns")
```

### Using Presets

```python
from gaap.research import DDEConfig, DeepDiscoveryEngine

# Quick research (depth=1)
quick_config = DDEConfig.quick()

# Standard research (depth=3)
standard_config = DDEConfig.standard()

# Deep research (depth=5)
deep_config = DDEConfig.deep()

# Academic research (high ETS standards)
academic_config = DDEConfig.academic()

# Use preset
engine = DeepDiscoveryEngine(config=DDEConfig.deep())
result = await engine.research("Quantum computing algorithms")
```

### API Integration

```python
from fastapi import FastAPI
from gaap.research import DeepDiscoveryEngine, DDEConfig

app = FastAPI()
engine = DeepDiscoveryEngine(config=DDEConfig.standard())

@app.post("/research")
async def research(query: str, depth: int = 3):
    result = await engine.research(query, depth=depth)
    return result.to_dict()

@app.get("/stats")
async def stats():
    return engine.get_stats()

@app.get("/config")
async def config():
    return engine.get_config().to_dict()
```

### With LLM Provider

```python
from gaap.research import DeepDiscoveryEngine, DDEConfig
from gaap.providers import GeminiProvider

# Create provider
provider = GeminiProvider(api_key="your-api-key")

# Create engine with provider for synthesis
engine = DeepDiscoveryEngine(
    config=DDEConfig(
        research_depth=3,
        synthesizer=SynthesizerConfig(
            cross_validate_enabled=True,
        ),
    ),
    llm_provider=provider,
)

result = await engine.research("Machine learning best practices")

# Hypotheses are now built and verified with LLM
for hypothesis in result.finding.hypotheses:
    print(f"[{hypothesis.status.name}] {hypothesis.statement}")
    print(f"  Confidence: {hypothesis.confidence:.0%}")
```

---

## Best Practices

### 1. Choose Appropriate Depth

| Use Case | Recommended Depth | Config |
|----------|-------------------|--------|
| Quick fact lookup | 1 | `DDEConfig.quick()` |
| General research | 3 | `DDEConfig.standard()` |
| Deep analysis | 4 | Custom with depth=4 |
| Academic research | 5 | `DDEConfig.academic()` |

### 2. Set Appropriate ETS Thresholds

```python
# For academic papers
config = DDEConfig(
    source_audit=SourceAuditConfig(
        min_ets_threshold=0.5,  # Only reliable+ sources
    ),
)

# For general research (more sources)
config = DDEConfig(
    source_audit=SourceAuditConfig(
        min_ets_threshold=0.3,  # Include questionable sources
    ),
)
```

### 3. Handle Large Research Efficiently

```python
# Limit sources for faster results
config = DDEConfig(
    max_total_sources=30,  # Limit to top 30 sources
    max_total_hypotheses=10,  # Limit hypotheses
    parallel_processing=True,
    max_parallel_tasks=5,  # Concurrent extraction
)
```

### 4. Cache and Reuse Results

```python
# Check existing research first
config = DDEConfig(
    check_existing_research=True,  # Check cache
)

# Force fresh research
result = await engine.research(
    query,
    force_fresh=True,
)
```

### 5. Monitor Performance

```python
# Get statistics
stats = engine.get_stats()
print(f"Research count: {stats['research_count']}")
print(f"Cache hit rate: {stats['cache_hit_rate']}")

# Component stats
for component, comp_stats in stats['components'].items():
    print(f"{component}: {comp_stats}")
```

### 6. Handle Errors Gracefully

```python
result = await engine.research(query)

if not result.success:
    print(f"Error: {result.error}")
    print(f"Execution trace:")
    for step in result.execution_trace:
        if step.error:
            print(f"  {step.step_name}: {step.error}")
```

---

## Performance Characteristics

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| Quick search (depth=1) | 2-5s | Web search only |
| Standard research (depth=3) | 10-30s | + Citation following |
| Deep research (depth=5) | 30-120s | + Cross-validation |
| Content extraction | 0.5-2s per URL | Depends on page size |
| ETS scoring | <10ms per source | Pure calculation |
| Hypothesis synthesis | 1-3s per hypothesis | LLM-dependent |

---

## Next Steps

- [API Reference](API_REFERENCE.md) - Detailed API documentation
- [Architecture Guide](ARCHITECTURE.md) - System architecture details
- [Examples](examples/) - More code examples
