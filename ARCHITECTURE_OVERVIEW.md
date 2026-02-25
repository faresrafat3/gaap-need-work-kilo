# GAAP Architecture Overview / نظرة عامة على معمارية GAAP

**Version / الإصدار**: 0.9.0  
**Last Updated / آخر تحديث**: February 2026

---

## 1. System Overview / نظرة عامة على النظام

### 1.1 What is GAAP? / ما هو GAAP؟

GAAP (Generative Agentic Architecture Platform) is an autonomous AI coding agent with a 4-layer OODA cognitive architecture. It integrates Deep Research, Self-Healing, Meta-Learning, and Swarm Intelligence into a unified cognitive system with a full Web GUI featuring real-time updates.

GAAP (منصة العمارة البرمجية التوليدية) هو وكيل ترميز ذكي مستقل مع معمارية認知 OODA ذات 4 طبقات. يدمج البحث العميق، والشفاء الذاتي، والتعلم الوصفي، وذكاء السرب في نظام معرفي موحد مع واجهة ويب كاملة تتميز بالتحديثات في الوقت الفعلي.

### 1.2 High-Level Architecture / المعمارية عالية المستوى

```mermaid
flowchart TB
    subgraph User["المستخدم / User"]
        CLI[CLI]
        API[API]
        Web[Web Interface]
    end
    
    subgraph OODA["OODA Loop / حلقة OODA"]
        subgraph Layer0["Layer 0: Interface / الطبقة 0: الواجهة"]
            Firewall[Firewall 7-Layer]
            Intent[Intent Classifier]
            Router[Smart Router]
        end
        
        subgraph Layer1["Layer 1: Strategic / الطبقة 1: الاستراتيجي"]
            ToT[Tree of Thoughts]
            MAD[MAD Panel]
            MCTS[Monte Carlo Tree Search]
        end
        
        subgraph Layer2["Layer 2: Tactical / الطبقة 2: التكتيكي"]
            Decomposer[Task Decomposer]
            DAG[Dependency DAG]
            Phase[Phase Planner]
        end
        
        subgraph Layer3["Layer 3: Execution / الطبقة 3: التنفيذ"]
            Executor[Tool Executor]
            Healer[Self-Healing]
            Auditor[Code Auditor]
        end
    end
    
    subgraph Memory["Memory System / نظام الذاكرة"]
        WM[Working Memory]
        EM[Episodic Memory]
        SM[Semantic Memory]
        PM[Procedural Memory]
    end
    
    subgraph Swarm["Swarm Intelligence / ذكاء السرب"]
        Fractals[Fractal Agents]
        Auction[Auction Mechanism]
        Reputation[Reputation System]
    end
    
    User --> Layer0
    Layer0 --> Layer1
    Layer1 --> Layer2
    Layer2 --> Layer3
    Layer3 --> Memory
    Layer1 -.-> ToT
    Layer2 -.-> DAG
    Layer3 -.-> Healer
```

### 1.3 OODA Loop / حلقة OODA

The OODA (Observe-Orient-Decide-Act) loop is the core cognitive cycle:

```mermaid
stateDiagram-v2
    [*] --> Observe
    Observe --> Orient : Input
    Orient --> Decide : Strategy
    Decide --> Act : Plan
    Act --> Learn : Result
    Learn --> Observe : Feedback
    Act --> Orient : Re-plan
    Orient --> [*]
```

| Phase / المرحلة | Layer / الطبقة | Component / المكون | Responsibility / المسؤولية |
|----------------|----------------|--------------------|---------------------------|
| **Observe** | L0 | `PromptFirewall` | Security scanning, intent classification, environment sensing / فحص الأمان، تصنيف النية، استشعار البيئة |
| **Orient** | L1 | `StrategicToT` | Deep Research (STORM) and Strategy Generation / البحث العميق وتوليد الاستراتيجية |
| **Decide** | L2 | `TacticalDecomposer` | Task breakdown and dependency graph construction / تفكيك المهام وبناء رسم بياني للتبعيات |
| **Act** | L3 | `SpecializedExecutors` | Action execution with Tool Synthesis / تنفيذ الإجراءات مع تركيب الأدوات |
| **Learn** | Meta | `Metacognition` | Reflective learning and reputation updates / التعلم التأملي وتحديثات السمعة |

---

## 2. Layer Architecture / معمارية الطبقات

### 2.1 Layer 0: Interface / الطبقة 0: الواجهة

#### Purpose / الغرض

Layer 0 is the entry point for all GAAP requests. It provides security scanning, intent classification, complexity estimation, and smart routing decisions.

الطبقة 0 هي نقطة الدخول لجميع طلبات GAAP. توفر فحص الأمان، تصنيف النية، تقدير التعقيد، وقرارات التوجيه الذكي.

#### Components / المكونات

```mermaid
flowchart LR
    Input[Input<br/>المدخلات] --> Firewall[7-Layer<br/>Firewall]
    Firewall --> Intent[Intent<br/>Classifier]
    Intent --> Complex[Complexity<br/>Estimator]
    Complex --> Router[Smart<br/>Router]
    Router --> Output[Layer 1/2/3]
```

| Component / المكون | File / الملف | Description / الوصف |
|-------------------|--------------|---------------------|
| `PromptFirewall` | `gaap/security/firewall.py` | 7-layer security scanning / فحص أمان من 7 طبقات |
| `IntentClassifier` | `gaap/layers/layer0_interface.py` | 11 intent types classification / تصنيف 11 نوع نية |
| `ComplexityEstimator` | `gaap/layers/layer0_interface.py` | Task complexity scoring / تقدير تعقيد المهمة |
| `SmartRouter` | `gaap/routing/router.py` | Dynamic routing to L1/L2/L3 / التوجيه الديناميكي |

#### Flow / التدفق

1. **Input Reception**: User request enters Layer 0 / استقبال المدخلات: يدخل طلب المستخدم الطبقة 0
2. **Security Scan**: 7-layer firewall checks for threats / فحص الأمان: يفحص جدار الحماية المكون من 7 طبقات للتهديدات
3. **Intent Classification**: Classifies into 11 intent types / تصنيف النية: يصنف إلى 11 نوع نية
4. **Complexity Estimation**: Estimates task complexity (SIMPLE/MODERATE/COMPLEX/CRITICAL) / تقدير التعقيد: يقدر تعقيد المهمة
5. **Routing Decision**: Routes to Strategic (L1), Tactical (L2), or Direct (L3) / قرار التوجيه: يوجه إلى الاستراتيجي أو التكتيكي أو المباشر

---

### 2.2 Layer 1: Strategic (Think) / الطبقة 1: الاستراتيجي (التفكير)

#### Purpose / الغرض

Layer 1 handles high-level strategic planning using Tree of Thoughts (ToT), MAD (Multi-Agent Debate) Panel, and Monte Carlo Tree Search (MCTS) for complex decision-making.

الطبقة 1 تتعامل مع التخطيط الاستراتيجي عالي المستوى باستخدام شجرة الأفكار (ToT)، ولوحة MAD (النقاش متعدد الوكلاء)، وشجرة البحث مونت كارلو (MCTS) لاتخاذ القرارات المعقدة.

#### Components / المكونات

```mermaid
flowchart TB
    subgraph Layer1["Layer 1: Strategic"]
        Input[Intent] --> ToT[Tree of Thoughts]
        ToT --> MAD[MAD Panel]
        MAD --> MCTS[MCTS]
        MCTS --> Output[Architecture Spec]
    end
    
    subgraph ToT_Components["ToT Components"]
        Depth[Depth: 5]
        Branch[Branching: 4]
        Explore[Exploration]
        Evaluate[Evaluation]
    end
    
    subgraph MAD_Components["MAD Components"]
        Security[Security Critic]
        Performance[Performance Critic]
        Scalability[Scalability Critic]
        Cost[Cost Critic]
        Maintain[Maintainability Critic]
        Compliance[Compliance Critic]
    end
```

| Component / المكون | File / الملف | Description / الوصف |
|-------------------|--------------|---------------------|
| `ToTStrategic` | `gaap/layers/layer1_strategic.py` | Tree of Thoughts with depth=5, branching=4 / شجرة الأفكار بعمق 5 وتفرع 4 |
| `MADPanel` | `gaap/mad/critic_prompts.py` | 6 architecture critics / 6 نقاد معماريين |
| `MCTSStrategic` | `gaap/layers/mcts_logic.py` | Monte Carlo Tree Search for COMPLEX tasks / شجرة البحث مونت كارلو للمهام المعقدة |
| `WisdomDistiller` | `gaap/meta_learning/wisdom_distiller.py` | Extracts patterns from history / يستخرج الأنماط من التاريخ |

#### Flow / التدفق

1. **Input Processing**: Receives structured intent from Layer 0 / معالجة المدخلات: يستقبل النية المهيكلة من الطبقة 0
2. **ToT Exploration**: Explores multiple solution paths (depth=5, branching=4) / استكشاف ToT: يستكشف مسارات الحل المتعددة
3. **MAD Debate**: 6 critics evaluate each path / نقاش MAD: يقيم 6 نقاد كل مسار
4. **MCTS Search**: For COMPLEX/CRITICAL tasks, uses MCTS for optimal path / بحث MCTS: للمهام المعقدة/الحرجة، يستخدم MCTS للمسار الأمثل
5. **Architecture Generation**: Produces ArchitectureSpec / توليد المعمارية: ينتج مواصفات المعمارية

---

### 2.3 Layer 2: Tactical (Plan) / الطبقة 2: التكتيكي (التخطيط)

#### Purpose / الغرض

Layer 2 decomposes strategic plans into executable tasks using Task Decomposition and builds a Directed Acyclic Graph (DAG) for dependency management.

الطبقة 2 تفكك الخطط الاستراتيجية إلى مهام قابلة للتنفيذ باستخدام تفكيك المهام وتبني رسم بياني غير دوري موجه (DAG) لإدارة التبعيات.

#### Components / المكونات

```mermaid
flowchart TB
    Input[Architecture Spec] --> Decomposer[Task Decomposer]
    Decomposer --> Categories[Multi-Domain Categories]
    Categories --> DAG[DAG Builder]
    DAG --> Phase[Phase Planner]
    Phase --> Output[Task Graph]
    
    subgraph Categories["Task Categories"]
        SE[Software Engineering]
        RI[Research & Intelligence]
        DT[Diagnostics]
        AN[Analysis]
    end
    
    subgraph DAG_Properties["DAG Properties"]
        Acyclic[Acyclic]
        Parallel[Parallel Execution]
        Dependencies[Dependencies]
    end
```

| Component / المكون | File / الملف | Description / الوصف |
|-------------------|--------------|---------------------|
| `TacticalDecomposer` | `gaap/layers/layer2_tactical.py` | Multi-domain task breakdown / تفكيك المهام متعدد المجالات |
| `TaskGraph` | `gaap/layers/layer2_tactical.py` | DAG construction / بناء الرسم البياني |
| `PhasePlanner` | `gaap/layers/phase_planner.py` | Phase discovery and planning / اكتشاف المراحل والتخطيط |
| `SemanticDependencies` | `gaap/layers/semantic_dependencies.py` | Semantic dependency resolution / حل التبعيات الدلالية |

#### Flow / التدفق

1. **Task Decomposition**: Breaks architecture spec into atomic tasks / تفكيك المهام: يكسر مواصفات المعمارية إلى مهام ذرية
2. **Category Classification**: Categorizes tasks (SETUP, DATABASE, API, FRONTEND, TESTING, SECURITY, etc.) / تصنيف الفئات: يصنف المهام
3. **DAG Construction**: Builds dependency graph ensuring no cycles / بناء DAG: يبني رسم التبعيات بدون دورات
4. **Phase Planning**: Groups tasks into executable phases / تخطيط المراحل: يجمع المهام في مراحل قابلة للتنفيذ
5. **Semantic Resolution**: Resolves cross-domain dependencies / الحل الدلالي: يحل التبعيات عبر المجالات

---

### 2.4 Layer 3: Execution (Act) / الطبقة 3: التنفيذ (الفعل)

#### Purpose / الغرض

Layer 3 executes tasks using specialized executors, includes self-healing capabilities for automatic error recovery, and performs code auditing.

الطبقة 3 تنفذ المهام باستخدام المنفذين المتخصصين، تشمل قدرات الشفاء الذاتي للتعافي التلقائي من الأخطاء، وتقوم بتدقيق الكود.

#### Components / المكونات

```mermaid
flowchart TB
    Input[Task Graph] --> Executor[Tool Executor]
    Executor --> Healer{Self-Healing?}
    Healer -->|Yes| Heal[Healing Loop]
    Healer -->|No| Auditor[Code Auditor]
    Heal --> Executor
    Auditor --> Output[Results]
    
    subgraph Tools["Tool System"]
        Native[Native Tools]
        MCP[MCP Tools]
        Synthesized[Synthesized Tools]
    end
    
    subgraph Healing["Healing Mechanisms"]
        Retry[Retry]
        Refine[Refine]
        Replan[Replan]
        Fallback[Fallback]
    end
```

| Component / المكون | File / الملف | Description / الوصف |
|-------------------|--------------|---------------------|
| `Layer3Execution` | `gaap/layers/layer3_execution.py` | Main execution engine / محرك التنفيذ الرئيسي |
| `NativeToolCaller` | `gaap/layers/native_function_caller.py` | Native function execution / تنفيذ الدوال الأصلية |
| `ToolSynthesizer` | `gaap/tools/synthesizer.py` | Dynamic tool synthesis / تركيب الأدوات الديناميكي |
| `SelfHealingSystem` | `gaap/healing/healer.py` | Error detection and recovery / اكتشاف الأخطاء والتعافي |
| `CodeAuditor` | `gaap/layers/code_auditor.py` | Code quality verification / التحقق من جودة الكود |

#### Flow / التدفق

1. **Task Selection**: Selects next executable task from DAG / اختيار المهمة: يختار المهمة التالية القابلة للتنفيذ من DAG
2. **Tool Selection**: Chooses appropriate tool (native/MCP/synthesized) / اختيار الأداة: يختار الأداة المناسبة
3. **Execution**: Runs the tool with parameters / التنفيذ: يشغل الأداة بالمعلمات
4. **Self-Healing**: On error, attempts retry → refine → replan → fallback / الشفاء الذاتي: عند الخطأ، يحاول إعادة المحاولة → التحسين → إعادة التخطيط → البديل
5. **Auditing**: Verifies output quality / التدقيق: يتحقق من جودة المخرجات

---

## 3. Memory System / نظام الذاكرة

GAAP implements a 4-tier hierarchical memory system inspired by human memory architecture:

```mermaid
flowchart TB
    subgraph MemoryTiers["Memory Tiers / طبقات الذاكرة"]
        L1[L1: Working Memory<br/>ذاكرة العمل]
        L2[L2: Episodic Memory<br/>ذاكرة الأحداث]
        L3[L3: Semantic Memory<br/>ذاكرة دلالية]
        L4[L4: Procedural Memory<br/>ذاكرة إجرائية]
    end
    
    Input[Experience] --> L1
    L1 --> L2 : Consolidation
    L2 --> L3 : Abstraction
    L3 --> L4 : Proceduralization
    
    L1 <--> L2 : Short-term
    L2 <--> L3 : Medium-term
    L3 <--> L4 : Long-term
```

### 3.1 Memory Tiers / طبقات الذاكرة

| Tier / الطبقة | Type / النوع | Capacity / السعة | Purpose / الغرض |
|--------------|-------------|-----------------|-----------------|
| **L1: Working** | `WorkingMemory` | 100 items | Fast access to current context / الوصول السريع للسياق الحالي |
| **L2: Episodic** | `EpisodicMemory` | Unlimited | Event history and learning / تاريخ الأحداث والتعلم |
| **L3: Semantic** | `SemanticMemory` | Unlimited | Patterns and extracted knowledge / الأنماط والمعرفة المستخرجة |
| **L4: Procedural** | `ProceduralMemory` | Unlimited | Skills and procedures / المهارات والإجراءات |

### 3.2 Dream Cycle / دورة الأحلام

The Dream Cycle consolidates episodic memories into semantic patterns during idle periods:

```mermaid
sequenceDiagram
    participant Day as Day Cycle
    participant WM as Working Memory
    participant EM as Episodic Memory
    participant SM as Semantic Memory
    participant Dream as Dream Cycle
    
    Day->>WM: Store current context
    Day->>EM: Record episodes
    EM->>Dream: Trigger consolidation
    Dream->>EM: Select important episodes
    EM->>SM: Extract patterns
    SM->>SM: Update knowledge graph
    Dream-->>Day: Ready for next day
```

**File**: `gaap/memory/hierarchical.py`

---

## 4. Meta-Learning / التعلم الوصفي

### 4.1 Confidence Scoring / تقدير الثقة

GAAP implements metacognitive confidence scoring to self-assess decision quality:

```mermaid
flowchart LR
    Input[Decision] --> Assess[Confidence Assessor]
    Assess --> Factors[Factors]
    
    subgraph Factors["Confidence Factors"]
        F1[Task Similarity]
        F2[Historical Success]
        F3[Information Quality]
        F4[Uncertainty]
    end
    
    Factors --> Score[Confidence Score]
    Score --> Output[0.0 - 1.0]
```

**File**: `gaap/core/confidence_scorer.py`

### 4.2 Wisdom Distillation / تقطير الحكمة

Extracts actionable insights from episodic memories:

```mermaid
flowchart TB
    Episodes[Episodes] --> Classifier[Category Classifier]
    Classifier --> Extractors[Pattern Extractors]
    
    subgraph Extractors["Extractors"]
        E1[Success Patterns]
        E2[Failure Patterns]
        E3[Optimization Patterns]
    end
    
    Extractors --> Heuristics[Heuristics]
    Heuristics --> Output[Applicable Lessons]
```

**File**: `gaap/meta_learning/wisdom_distiller.py`

### 4.3 Failure Learning / تعلم الفشل

Tracks and learns from execution failures:

```mermaid
flowchart LR
    Failure[Failure] --> Analysis[Root Cause Analysis]
    Analysis --> Categorize[Categorize]
    Categorize --> Extract[Extract Lesson]
    Extract --> Store[Failure Store]
    Store --> Future[Future Decisions]
```

**File**: `gaap/meta_learning/failure_store.py`

---

## 5. Swarm Intelligence / ذكاء السرب

### 5.1 Fractal Agents / الوكلاء المتشققين

Fractals are specialized sub-agents with domain expertise:

```mermaid
flowchart TB
    subgraph Fractal["Fractal Agent"]
        State[State Machine]
        Capabilities[Capabilities]
        Reputation[Reputation Score]
        Bidding[Bidding Logic]
    end
    
    State -->|IDLE| Bidding
    Bidding -->|BIDDING| State
    State -->|EXECUTING| Bidding
    State -->|COOLDOWN| Bidding
    
    Capabilities -->|evaluate| Bidding
    Reputation -->|inform| Bidding
```

**File**: `gaap/swarm/fractal.py`

### 5.2 Reputation System / نظام السمعة

Domain-aware reputation tracking:

```mermaid
flowchart TB
    Task[Task Complete] --> Event[Reputation Event]
    Event --> Update[Score Update]
    
    subgraph Update["Update Formula"]
        Prior[Prior Score]
        Evidence[New Evidence]
        Decay[Time Decay]
    end
    
    Prior -->|Bayesian| Update
    Evidence -->|Weigh| Update
    Decay -->|Apply| Update
    
    Update --> Score[New Score]
```

**File**: `gaap/swarm/reputation.py`

### 5.3 Auction Mechanism / آلية المزاد

Reputation-Based Task Auction (RBTA):

```mermaid
sequenceDiagram
    participant Orch as Orchestrator
    participant Auction as Auctioneer
    participant F1 as Fractal 1
    participant F2 as Fractal 2
    
    Orch->>Auction: Create TaskAuction
    Auction->>F1: Broadcast task
    Auction->>F2: Broadcast task
    F1->>Auction: Submit Bid (utility score)
    F2->>Auction: Submit Bid (utility score)
    Auction->>Orch: Select winner
    Orch->>F1: TaskAward
```

**File**: `gaap/swarm/auction.py`

---

## 6. Security / الأمان

### 6.1 Firewall Layers / طبقات الجدار الناري

7-layer security defense system:

```mermaid
flowchart TB
    Input[Input] --> L1[L1: Surface Inspection]
    L1 --> L2[L2: Lexical Analysis]
    L2 --> L3[L3: Syntactic Analysis]
    L3 --> L4[L4: Semantic Analysis]
    L4 --> L5[L5: Contextual Verification]
    L5 --> L6[L6: Behavioral Analysis]
    L6 --> L7[L7: Adversarial Testing]
    L7 --> Output{Safe?}
    Output -->|Yes| Pass[Pass to Layer 1]
    Output -->|No| Block[Block & Log]
```

| Layer / الطبقة | Check / الفحص |
|---------------|--------------|
| L1: Surface | Basic pattern matching / مطابقة الأنماط الأساسية |
| L2: Lexical | Token analysis / تحليل الرموز |
| L3: Syntactic | Structure validation / التحقق من البنية |
| L4: Semantic | Meaning analysis / تحليل المعنى |
| L5: Contextual | Context verification / التحقق من السياق |
| L6: Behavioral | Behavior monitoring / مراقبة السلوك |
| L7: Adversarial | Attack simulation / محاكاة الهجوم |

**File**: `gaap/security/firewall.py`

### 6.2 Sandbox Execution / التنفيذ في بيئة معزولة

Tools execute in isolated environments:

**File**: `gaap/security/sandbox.py`

### 6.3 Input Validation / التحقق من المدخلات

Strict input validation at Layer 0:

**File**: `gaap/security/preflight.py`

---

## 7. Web Interface / واجهة الويب

### 7.1 Frontend Stack / تقنيات الواجهة الأمامية

```mermaid
flowchart LR
    subgraph Frontend["Frontend / الواجهة الأمامية"]
        NextJS[Next.js 14]
        React[React 18]
        Tailwind[Tailwind CSS]
        State[Zustand]
        Query[React Query]
        Charts[Recharts]
        Graph[@xyflow React]
    end
    
    NextJS --> React
    React --> Tailwind
    React --> State
    React --> Query
    Query --> Charts
    Query --> Graph
```

**Stack**:
- **Framework**: Next.js 14
- **UI**: React 18, Tailwind CSS
- **State**: Zustand
- **Data Fetching**: React Query
- **Visualization**: Recharts, @xyflow/react
- **Animations**: Framer Motion

**File**: `frontend/package.json`

### 7.2 Backend API / واجهة API الخلفية

FastAPI-based REST API:

```mermaid
flowchart TB
    subgraph API["REST API"]
        Tasks[/api/tasks]
        Sessions[/api/sessions]
        Memory[/api/memory]
        System[/api/system]
        Providers[/api/providers]
        Research[/api/research]
    end
    
    Tasks --> DB[(SQLite)]
    Sessions --> DB
    Memory --> DB
    System --> Observability
```

**Files**:
- `gaap/api/tasks.py`
- `gaap/api/sessions.py`
- `gaap/api/memory.py`
- `gaap/api/system.py`
- `gaap/api/providers.py`
- `gaap/api/research.py`

### 7.3 WebSocket / ويب سوكت

Real-time event streaming:

```mermaid
sequenceDiagram
    participant Client
    participant WS as WebSocket
    participant Server
    
    Client->>WS: Connect
    WS->>Server: Register
    Server->>WS: Accept
    loop Events
        Server->>WS: Broadcast event
        WS->>Client: Push update
    end
```

**Channels**:
- `events`: General system events
- `ooda`: OODA loop progress
- `steering`: User steering commands

**File**: `gaap/api/websocket.py`

---

## 8. Key Files / الملفات الرئيسية

### Core Engine / محرك الأساسي

| File / الملف | Description / الوصف |
|-------------|-------------------|
| `gaap/gaap_engine.py` | Main OODA loop engine / محرك حلقة OODA الرئيسي |
| `gaap/core/types.py` | Core type definitions / تعريفات الأنواع الأساسية |
| `gaap/core/config.py` | Configuration management / إدارة التكوين |

### Layers / الطبقات

| File / الملف | Description / الوصف |
|-------------|-------------------|
| `gaap/layers/layer0_interface.py` | Interface layer / الطبقةinterface |
| `gaap/layers/layer1_strategic.py` | Strategic layer (ToT, MAD, MCTS) / الطبقة الاستراتيجية |
| `gaap/layers/layer2_tactical.py` | Tactical layer (DAG, decomposition) / الطبقة التكتيكية |
| `gaap/layers/layer3_execution.py` | Execution layer (tools, healing) / الطبقة التنفيذية |

### Memory / الذاكرة

| File / الملف | Description / الوصف |
|-------------|-------------------|
| `gaap/memory/hierarchical.py` | 4-tier memory system / نظام الذاكرة رباعي الطبقات |
| `gaap/memory/vector_backends.py` | Vector storage backends / واجهات التخزين المتجهة |

### Security / الأمان

| File / الملف | Description / الوصف |
|-------------|-------------------|
| `gaap/security/firewall.py` | 7-layer prompt firewall / جدار الحماية |
| `gaap/security/sandbox.py` | Sandboxed execution / التنفيذ المعزول |
| `gaap/security/preflight.py` | Input validation / التحقق من المدخلات |

### Swarm / السرب

| File / الملف | Description / الوصف |
|-------------|-------------------|
| `gaap/swarm/fractal.py` | Fractal agent implementation / تنفيذ الوكيل المتشقق |
| `gaap/swarm/reputation.py` | Reputation system / نظام السمعة |
| `gaap/swarm/auction.py` | Auction mechanism / آلية المزاد |
| `gaap/swarm/orchestrator.py` | Swarm orchestration / تنسيق السرب |

### Meta-Learning / التعلم الوصفي

| File / الملف | Description / الوصف |
|-------------|-------------------|
| `gaap/meta_learning/confidence.py` | Confidence scoring / تقدير الثقة |
| `gaap/meta_learning/wisdom_distiller.py` | Wisdom extraction / استخراج الحكمة |
| `gaap/meta_learning/failure_store.py` | Failure learning / تعلم الفشل |

### API / واجهة برمجة التطبيقات

| File / الملف | Description / الوصف |
|-------------|-------------------|
| `gaap/api/websocket.py` | WebSocket manager / مدير ويب سوكت |
| `gaap/api/tasks.py` | Task endpoints / نقاط نهاية المهام |
| `gaap/api/memory.py` | Memory endpoints / نقاط نهاية الذاكرة |
| `gaap/api/system.py` | System health / صحة النظام |

### Frontend / الواجهة الأمامية

| File / الملف | Description / الوصف |
|-------------|-------------------|
| `frontend/src/lib/api.ts` | API client / عميل API |
| `frontend/src/hooks/useWebSocket.ts` | WebSocket hook / خطاف ويب سوكت |
| `frontend/src/components/ooda/` | OODA visualization components / مكونات تصور OODA |

---

## Project Statistics / إحصائيات المشروع

- **Specifications Implemented**: 38 (73%)
- **Lines of Code**: ~90,000
- **Tests**: 1,500+
- **Test Coverage**: ~55%

---

**GAAP: The future of autonomous cognitive engineering**  
**GAAP: مستقبل هندسة認知 المستقلة**
