# GAAP Quick Reference Card (Cheat Sheet)

> **آخر تحديث:** February 17, 2026  
> **المستوى:** جميع المستويات  
> **الغرض:** مرجع سريع في صفحة واحدة

---

## 🚀 Quick Start (30 ثانية)

```python
from gaap import GAAPEngine, GAAPRequest

engine = GAAPEngine(budget=10.0)
response = await engine.process(
    GAAPRequest(text="Write a binary search function")
)
print(response.output)
engine.shutdown()
```

---

## 📦 التثبيت (دقيقة واحدة)

```bash
# 1. Clone & Setup
git clone https://github.com/gaap-system/gaap.git
cd gaap
python -m venv .venv && source .venv/bin/activate

# 2. Install
pip install -e ".[dev]"
pip install streamlit pandas plotly

# 3. API Keys (.gaap_env)
echo "GROQ_API_KEY=gsk_..." > .gaap_env
echo "GEMINI_API_KEYS=key1,key2" >> .gaap_env

# 4. Test
gaap chat "Hello"
```

---

## 🏗️ البنية المعمارية (نظرة سريعة)

```
┌─────────────────────────────────────────────┐
│  L0: Interface (Security + Router)          │
│  ↓                                           │
│  L1: Strategic (ToT + MAD Panel)            │
│  ↓                                           │
│  L2: Tactical (Task Decomposition)          │
│  ↓                                           │
│  L3: Execution (Parallel + Quality)         │
└─────────────────────────────────────────────┘
       ↑           ↑           ↑
    Memory     Healing    Observability
```

---

## 🎯 OODA Loop (5 خطوات)

```
1. OBSERVE    → Scan environment, classify intent
2. ORIENT     → Update plan, replan if needed
3. DECIDE     → Select next task
4. ACT        → Execute with quality checks
5. LEARN      → Record lessons to memory

Max: 15 iterations | 2 retries per task
```

---

## 🛡️ Self-Healing Levels (5 مستويات)

| Level | Action | When |
|-------|--------|------|
| **L1** | RETRY | Transient errors (network, timeout) |
| **L2** | REFINE | Prompt syntax/logic errors |
| **L3** | PIVOT | Model capability limits |
| **L4** | STRATEGY SHIFT | Task too complex |
| **L5** | HUMAN ESCALATE | Unrecoverable errors |

---

## 🧠 Memory Tiers (4 طبقات)

| Tier | Name | Capacity | Purpose |
|------|------|----------|---------|
| **L1** | Working | 100 items | Current context |
| **L2** | Episodic | Unlimited | Event history |
| **L3** | Semantic | Unlimited | Patterns & rules |
| **L4** | Procedural | Unlimited | Skills & templates |

---

## 🎭 MAD Panels (6 نقاد × 2 جولات)

### L1 Architecture Panel:
```
1. SCALABILITY      → Can it scale?
2. PRAGMATISM       → Is it practical?
3. COST             → Is it cost-effective?
4. ROBUSTNESS       → Is it resilient?
5. MAINTAINABILITY  → Is it maintainable?
6. SECURITY_ARCH    → Is it secure?
```

### L3 Quality Panel:
```
1. LOGIC            → Is it correct?
2. SECURITY         → Any vulnerabilities?
3. PERFORMANCE      → Is it efficient?
4. STYLE            → Code style compliance
5. TEST_COVERAGE    → Tests included?
6. DOCUMENTATION    → Well documented?
```

---

## 🔄 Routing Strategies (5 استراتيجيات)

```python
RoutingStrategy.QUALITY_FIRST    # Best quality, cost doesn't matter
RoutingStrategy.COST_OPTIMIZED   # Cheapest option
RoutingStrategy.SPEED_FIRST      # Fastest response
RoutingStrategy.BALANCED         # Balance all factors
RoutingStrategy.SMART            # Context-aware decision
```

**Scoring Weights:**
```
Quality: 40% | Cost: 30% | Speed: 20% | Availability: 10%
```

---

## 🔒 Security Layers (7 طبقات)

```
L1: Surface Inspection      → Pattern matching
L2: Lexical Analysis        → Word-level check
L3: Syntactic Analysis      → Structure check
L4: Semantic Analysis       → Meaning check
L5: Contextual Verification → Context awareness
L6: Behavioral Analysis     → Behavior patterns
L7: Adversarial Testing     → Attack simulation
```

**Risk Levels:**
```python
SAFE → LOW → MEDIUM → HIGH → CRITICAL → BLOCKED
```

---

## 📊 Common Commands

```bash
# CLI
gaap chat "Write a function"     # Quick chat
gaap interactive                  # Interactive mode
gaap providers list               # List providers
gaap models tiers                 # Model tiers
gaap doctor                       # System diagnostics
gaap web                          # Start web UI

# Development
make check                        # Run all checks
pytest                            # Run tests
black gaap/ tests/                # Format code
mypy gaap/                        # Type check
ruff check gaap/ --fix            # Lint

# Docker
docker build -t gaap .
docker run -p 8501:8501 -p 8080:8080 gaap
docker-compose up -d
```

---

## 💻 Code Templates

### Basic Usage:
```python
from gaap import GAAPEngine, GAAPRequest, TaskPriority

engine = GAAPEngine(budget=50.0)

request = GAAPRequest(
    text="Build a REST API",
    priority=TaskPriority.HIGH,
    budget_limit=10.0
)

response = await engine.process(request)
print(f"Success: {response.success}")
print(f"Quality: {response.quality_score:.2f}")
print(f"Cost: ${response.total_cost_usd:.4f}")
print(f"Time: {response.total_time_ms:.0f}ms")

engine.shutdown()
```

### Custom Providers:
```python
from gaap.providers import GroqProvider, GeminiProvider
from gaap.routing.router import SmartRouter, RoutingStrategy

providers = [
    GroqProvider(api_key="gsk_..."),
    GeminiProvider(api_keys=["key1", "key2"]),
]

router = SmartRouter(
    providers=providers,
    strategy=RoutingStrategy.BALANCED,
    budget_limit=20.0
)

engine = GAAPEngine(providers=providers, router=router)
```

### Memory Usage:
```python
from gaap.memory.hierarchical import HierarchicalMemory, EpisodicMemory

memory = HierarchicalMemory()

# Record episode
episode = EpisodicMemory(
    task_id="task_001",
    action="code_generation",
    result="Generated API",
    success=True,
    duration_ms=3500,
    tokens_used=4500,
    cost_usd=0.15,
    model="llama-3.3-70b",
    provider="groq",
    lessons=["Use JWT for auth"]
)
memory.record_episode(episode)

# Search lessons
lessons = memory.search_lessons("authentication", top_k=3)
```

### Error Handling:
```python
from gaap.core.exceptions import (
    ProviderRateLimitError,
    ProviderTimeoutError,
    MaxRetriesExceededError
)

try:
    response = await engine.process(request)
except ProviderRateLimitError as e:
    if e.recoverable:
        await healing_system.heal(e)
    else:
        escalate(e)
except MaxRetriesExceededError:
    print("Human intervention required")
```

---

## 🐛 Quick Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| **Rate Limit** | Add more API keys or use fallback |
| **Memory Full** | `memory.clear_tier(MemoryTier.WORKING)` |
| **Slow Response** | Use `RoutingStrategy.SPEED_FIRST` |
| **Axiom Violation** | Check `KNOWN_PACKAGES` list |
| **Security Block** | Reduce firewall strictness |
| **Provider Down** | Enable healing for auto-fallback |

---

## 📈 Performance Tips

```python
# 1. Enable Caching
from gaap.cache import LRUCache
cache = LRUCache(max_size=1000, ttl=3600)

# 2. Parallel Execution
layer3 = Layer3Execution(max_parallel=5)

# 3. Reduce MAD Rounds
layer1 = Layer1Strategic(mad_rounds=2)  # Default: 3

# 4. Use Faster Models
router = SmartRouter(strategy=RoutingStrategy.SPEED_FIRST)

# 5. Clear Memory Periodically
import gc; gc.collect()
```

---

## 📊 Exception Quick Reference

| Code | Exception | Recoverable? | Action |
|------|-----------|--------------|--------|
| `GAAP_PRV_004` | RateLimit | ✅ Yes | Wait + Retry |
| `GAAP_PRV_006` | Timeout | ✅ Yes | Increase timeout |
| `GAAP_PRV_005` | Auth Fail | ❌ No | Check API key |
| `GAAP_TSK_007` | MaxRetries | ❌ No | Human escalate |
| `GAAP_SEC_002` | Injection | ❌ No | Sanitize input |
| `GAAP_AXM_002` | Axiom Viol | ⚠️ Maybe | Fix violation |

---

## 🎯 Decision Tree (Quick)

```
Request Received
    ↓
Security Scan (L0)
    ↓
Is Safe? → No → BLOCK
    ↓ Yes
Intent Classification
    ↓
Route To:
├── DIRECT (L3)    → Simple tasks
├── TACTICAL (L2)  → Needs decomposition
└── STRATEGIC (L1) → Complex planning
    ↓
Execute → Quality Check → Return
```

---

## 📚 File Structure Quick Ref

```
gaap/
├── core/           # Types, config, exceptions
├── layers/         # L0-L3 implementation
├── providers/      # LLM providers
├── routing/        # Smart router + fallback
├── security/       # Firewall + audit
├── healing/        # Self-healing system
├── memory/         # Hierarchical memory
├── context/        # Context management
├── tools/          # Tool registry
├── cli/            # CLI commands
├── web/            # Streamlit UI
└── api/            # FastAPI REST API

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── benchmarks/     # Performance tests

docs/
├── ARCHITECTURE.md
├── API_REFERENCE.md
├── DEVELOPMENT.md
└── ... (14 more files)
```

---

## 🔗 Quick Links

| Resource | Location |
|----------|----------|
| **Full Documentation** | `QWEN_CODE_DOCUMENTATION.md` |
| **API Reference** | `docs/API_REFERENCE.md` |
| **Advanced Usage** | `docs/ADVANCED_USAGE.md` |
| **Examples** | `examples/README.md` |
| **Testing Guide** | `examples/07_testing_guide.py` |
| **Evolution Plan** | `docs/evolution_plan_2026/` |
| **Cellular Analysis** | `docs/CELLULAR_LEVEL_ANALYSIS.md` |

---

## 📞 Emergency Contacts (Debug Mode)

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("gaap").setLevel(logging.DEBUG)

# Get detailed stats
stats = engine.get_stats()
print(json.dumps(stats, indent=2))

# Check system health
gaap doctor

# View logs
tail -f .gaap/logs/gaap.log
```

---

*GAAP Quick Reference Card - Last Updated: February 17, 2026*  
*Print this page for quick access!*
