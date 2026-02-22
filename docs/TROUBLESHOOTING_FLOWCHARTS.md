# GAAP Troubleshooting Flowcharts

> **دليل استكشاف الأخطاء**  
> **آخر تحديث:** February 17, 2026  
> **النوع:** Flowcharts تفاعلية

---

## 📊 Flowchart 1: Request Failed

```
┌─────────────────────────────────────────────────────────────┐
│                    REQUEST FAILED                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Check Error Code  │
                  └───────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ GAAP_PRV_***  │  │ GAAP_TSK_***  │  │ GAAP_SEC_***  │
│ (Provider)    │  │ (Task)        │  │ (Security)    │
└───────────────┘  └───────────────┘  └───────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ PRV_004:      │  │ TSK_007:      │  │ SEC_002:      │
│ Rate Limit    │  │ Max Retries   │  │ Injection     │
└───────────────┘  └───────────────┘  └───────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ ✅ Recoverable│  │ ❌ Critical   │  │ ❌ Block      │
│ Wait + Retry  │  │ Human Escalate│  │ Sanitize Input│
│ Or Pivot      │  │               │  │               │
└───────────────┘  └───────────────┘  └───────────────┘
```

---

## 🔧 Flowchart 2: Self-Healing Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│                      ERROR DETECTED                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Is Recoverable?   │
                  └───────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                   Yes             No
                    │               │
                    ▼               ▼
          ┌─────────────────┐  ┌──────────┐
          │ Check Error Type│  │ Escalate │
          └─────────────────┘  │ to Human │
                    │          └──────────┘
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│ Transient │ │ Prompt    │ │ Model     │
│ (Network) │ │ Issue     │ │ Limit     │
└───────────┘ └───────────┘ └───────────┘
        │           │           │
        ▼           ▼           ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│ L1: RETRY │ │ L2: REFINE│ │ L3: PIVOT │
│ + Backoff │ │ Prompt    │ │ Provider  │
└───────────┘ └───────────┘ └───────────┘
        │           │           │
        └───────────┴───────────┘
                    │
                    ▼
          ┌─────────────────┐
          │ Still Failing?  │
          └─────────────────┘
                    │
            ┌───────┴───────┐
            │               │
           Yes             No
            │               │
            ▼               ▼
      ┌───────────┐   ┌──────────┐
      │ L4: SHIFT │   │ SUCCESS  │
      │ Strategy  │   └──────────┘
      └───────────┘
            │
            ▼
      ┌─────────────────┐
      │ Still Failing?  │
      └─────────────────┘
            │
      ┌─────┴─────┐
      │           │
     Yes         No
      │           │
      ▼           ▼
┌───────────┐ ┌──────────┐
│ L5: HUMAN │ │ SUCCESS  │
└───────────┘ └──────────┘
```

---

## 🛡️ Flowchart 3: Security Scan Decision

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT RECEIVED                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ L1: Surface Scan  │
                  └───────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                 Clean         Suspicious
                    │               │
                    ▼               ▼
          ┌─────────────────┐ ┌──────────┐
          │ L2: Lexical     │ │ BLOCK +  │
          │ Analysis        │ │ Log      │
          └─────────────────┘ └──────────┘
                    │
            ┌───────┴───────┐
            │               │
         Clean         Dirty
            │               │
            ▼               ▼
  ┌─────────────────┐ ┌──────────┐
  │ L3: Syntactic   │ │ QUARANTINE│
  │ Analysis        │ │ + Alert  │
  └─────────────────┘ └──────────┘
            │
    ┌───────┴───────┐
    │               │
 Valid          Invalid
    │               │
    ▼               ▼
┌─────────┐   ┌──────────┐
│ L4-L7   │   │ REJECT   │
│ Deep    │   │ + Report │
│ Analysis│   │          │
└─────────┘   └──────────┘
    │
    ▼
┌─────────────────┐
│ Final Risk Score│
└─────────────────┘
    │
    ├───── SAFE (0.0-0.3)    → ALLOW
    ├───── LOW (0.3-0.5)     → ALLOW + Monitor
    ├───── MEDIUM (0.5-0.7)  → Sanitize + Process
    ├───── HIGH (0.7-0.9)    → Block + Alert
    └───── CRITICAL (0.9+)   → Block + Report
```

---

## 🔄 Flowchart 4: Routing Decision

```
┌─────────────────────────────────────────────────────────────┐
│                  ROUTING REQUEST                            │
│  (Task: {type}, Priority: {priority}, Budget: ${budget})   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Get Strategy      │
                  │ (QUALITY/COST/    │
                  │  SPEED/BALANCED)  │
                  └───────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Score Providers   │
                  │ Quality × 0.4     │
                  │ Cost × 0.3        │
                  │ Speed × 0.2       │
                  │ Availability × 0.1│
                  └───────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Sort by Score     │
                  └───────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Check Budget      │
                  └───────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                In Budget     Over Budget
                    │               │
                    ▼               ▼
          ┌─────────────────┐ ┌──────────┐
          │ Select Top      │ │ Try      │
          │ Provider        │ │ Cheaper  │
          └─────────────────┘ │ Provider │
                    │         └──────────┘
                    │               │
                    └───────┬───────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Provider          │
                  │ Available?        │
                  └───────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                   Yes             No
                    │               │
                    ▼               ▼
          ┌─────────────────┐ ┌──────────┐
          │ EXECUTE         │ │ Fallback │
          │                 │ │ to Next  │
          └─────────────────┘ └──────────┘
```

---

## 🧠 Flowchart 5: Memory Operations

```
┌─────────────────────────────────────────────────────────────┐
│                  MEMORY OPERATION                           │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ RECORD    │   │ SEARCH    │   │ RETRIEVE  │
    │ (Episode) │   │ (Query)   │   │ (Context) │
    └───────────┘   └───────────┘   └───────────┘
            │               │               │
            ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ Determine │   │ Vector    │   │ Check     │
    │ Tier      │   │ Search    │   │ Cache     │
    └───────────┘   └───────────┘   └───────────┘
            │               │               │
            ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ L1:       │   │ Top-K     │   │ Cache Hit │
    │ Working   │   │ Results   │   │ → Return  │
    │ (Fast)    │   │           │   │           │
    └───────────┘   └───────────┘   └───────────┘
            │               │               │
            ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ L2:       │   │ Re-Rank   │   │ Cache Miss│
    │ Episodic  │   │ by        │   │ → Load    │
    │ (History) │   │ Relevance │   │ from Tier │
    └───────────┘   └───────────┘   └───────────┘
            │               │               │
            ▼               ▼               ▼
    ┌───────────┐   ┌───────────┐   ┌───────────┐
    │ L3:       │   │ Return    │   │ Merge     │
    │ Semantic  │   │ Results   │   │ Results   │
    │ (Patterns)│   │ + Context │   │           │
    └───────────┘   └───────────┘   └───────────┘
            │
            ▼
    ┌───────────┐
    │ L4:       │
    │ Procedural│
    │ (Skills)  │
    └───────────┘
```

---

## ⚡ Flowchart 6: Rate Limiting

```
┌─────────────────────────────────────────────────────────────┐
│                    REQUEST ARRIVED                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Check Rate Limiter│
                  │ (Token Bucket)    │
                  └───────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Tokens Available? │
                  └───────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                   Yes             No
                    │               │
                    ▼               ▼
          ┌─────────────────┐ ┌──────────┐
          │ Consume Tokens  │ │ Calculate│
          │ Allow Request   │ │ Wait Time│
          └─────────────────┘ └──────────┘
                    │               │
                    │               ▼
                    │         ┌──────────┐
                    │         │ Return   │
                    │         │ Retry-   │
                    │         │ After    │
                    │         └──────────┘
                    │               │
                    └───────┬───────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Execute Request   │
                  └───────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Success?          │
                  └───────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                   Yes             No
                    │               │
                    ▼               ▼
          ┌─────────────────┐ ┌──────────┐
          │ Update Stats    │ │ Check    │
          │ (Success)       │ │ Error    │
          └─────────────────┘ │ Type     │
                              └──────────┘
                                      │
                              ┌───────┴───────┐
                              │               │
                          Recoverable     Critical
                              │               │
                              ▼               ▼
                        ┌───────────┐  ┌──────────┐
                        │ Retry     │  │ Block +  │
                        │ (Backoff) │  │ Alert    │
                        └───────────┘  └──────────┘
```

---

## 🎯 Flowchart 7: Task Execution

```
┌─────────────────────────────────────────────────────────────┐
│                    TASK RECEIVED                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Validate Task     │
                  └───────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                 Valid          Invalid
                    │               │
                    ▼               ▼
          ┌─────────────────┐ ┌──────────┐
          │ Check           │ │ REJECT   │
          │ Dependencies    │ │ + Errors │
          └─────────────────┘ └──────────┘
                    │
                    ▼
          ┌─────────────────┐
          │ Dependencies    │
          │ Satisfied?      │
          └─────────────────┘
                    │
            ┌───────┴───────┐
            │               │
           Yes             No
            │               │
            ▼               ▼
      ┌───────────┐   ┌──────────┐
      │ EXECUTE   │   │ WAIT in  │
      │           │   │ Queue    │
      └───────────┘   └──────────┘
            │
            ▼
      ┌─────────────────┐
      │ Quality Check   │
      │ (MAD Panel)     │
      └─────────────────┘
            │
            ▼
      ┌─────────────────┐
      │ Quality Score   │
      └─────────────────┘
            │
    ┌───────┴───────────────┐
    │                       │
 ≥70%                  <70%
    │                       │
    ▼                       ▼
┌──────────┐         ┌──────────┐
│ APPROVE  │         │ REFINE   │
│ + Return │         │ + Retry  │
└──────────┘         └──────────┘
                            │
                            ▼
                      ┌──────────┐
                      │ Max      │
                      │ Retries? │
                      └──────────┘
                            │
                    ┌───────┴───────┐
                    │               │
                   Yes             No
                    │               │
                    ▼               ▼
          ┌─────────────────┐ ┌──────────┐
          │ FAIL Task       │ │ Back to  │
          │ + Escalate      │ │ Execute  │
          └─────────────────┘ └──────────┘
```

---

## 📊 Quick Decision Matrix

| Symptom | First Check | Likely Cause | Quick Fix |
|---------|-------------|--------------|-----------|
| **Slow Response** | Router Stats | Wrong strategy | Switch to SPEED_FIRST |
| **High Cost** | Provider Scores | Using expensive model | Switch to COST_OPTIMIZED |
| **Frequent Failures** | Healing Logs | Rate limiting | Add more API keys |
| **Memory Errors** | RSS Usage | Context overflow | Clear working memory |
| **Security Blocks** | Firewall Logs | False positive | Adjust strictness |
| **Axiom Violations** | Validator Output | New dependencies | Update KNOWN_PACKAGES |

---

## 🔍 Debug Checklist

```
□ Check error code (GAAP_XXX_XXX)
□ Review exception details
□ Check recoverable flag
□ Inspect healing history
□ Review provider stats
□ Check memory usage
□ Inspect security logs
□ Validate axiom checks
□ Review OODA iterations
□ Check quality scores
```

---

*GAAP Troubleshooting Flowcharts - Last Updated: February 17, 2026*
