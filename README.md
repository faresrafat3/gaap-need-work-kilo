# GAAP - Generative Agentic Architecture Platform (v0.9.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-1,500+-green.svg)](#)
[![Coverage](https://img.shields.io/badge/coverage-~55%25-yellow.svg)](#)
[![Lines of Code](https://img.shields.io/badge/LOC-90,000+-orange.svg)](#)

**GAAP** is an autonomous AI coding agent with a **4-layer OODA architecture**. It integrates Deep Research, Self-Healing, Meta-Learning, and Swarm Intelligence into a unified cognitive system with a full Web GUI featuring real-time updates.

## ✨ Features

- **4-Layer OODA Architecture**: Observe-Orient-Decide-Act cognitive loop
- **Deep Research Engine**: Web-augmented research with ETS scoring and hypothesis building
- **Self-Healing System**: Automatic error detection and recovery
- **Meta-Learning**: Continuous learning from past executions
- **Swarm Intelligence**: Multi-agent coordination and collaboration
- **Web Interface**: Full GUI with real-time updates
- **Metacognition & Confidence Scoring**: Self-awareness and decision confidence
- **Advanced LLM Interaction**: Dynamic persona adoption and tool synthesis

## 🧠 Cognitive Architecture

GAAP implements a 4-layer OODA (Observe-Orient-Decide-Act) cognitive system:

```
User Input → Layer 0: Observe → Layer 1: Orient → Layer 2: Decide → Layer 3: Act → Learn → Loop
```

| Layer | Engine Component | Responsibility |
|-------|------------------|----------------|
| **OBSERVE** | `PromptFirewall` | Security scanning, intent classification, and environment sensing |
| **ORIENT** | `StrategicToT` | Deep Research (STORM) and Strategy Generation (Tree of Thoughts) |
| **DECIDE** | `TacticalDecomposer` | Multi-domain task breakdown and dependency graph (DAG) construction |
| **ACT** | `SpecializedExecutors` | Action execution with Genetic Twin verification and Tool Synthesis |
| **LEARN** | `Metacognition` | Reflective learning, reputation updates, and episodic memory storage |

## 📊 Project Status

- **38 specifications implemented (73%)**
- **~90,000 lines of Python code**
- **1,500+ tests**
- **~55% test coverage**

## 📦 Installation

```bash
# Clone and install the package
git clone https://github.com/gaap-system/gaap.git
cd gaap
pip install -e .

# Frontend setup
cd frontend && npm install && npm run build
```

## 🛠️ Development Commands

```bash
make check   # Run formatters, lints, and type-checks
make test    # Run test suite
make dream   # Consolidate episodic memory into semantic patterns
make eval    # Run the Sovereign IQ test
make audit   # Verify axiomatic integrity
```

## 💻 Usage

### CLI

```bash
gaap --help
gaap run "Your task here"
gaap research "Research topic"
```

### API

```bash
# Start the API server
gaap api --port 8000
```

### Web GUI

```bash
# Start the web interface
gaap web
# Then open http://localhost:5173 in your browser
```

## 📚 Documentation

- [Architecture Guide](docs/ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Quick Reference](docs/QUICK_REFERENCE.md)
- [Troubleshooting](docs/TROUBLESHOOTING_FLOWCHARTS.md)
- [Evolution Plan 2026](docs/evolution_plan_2026/) - 50+ technical specifications

---

**GAAP: The future of autonomous cognitive engineering.**
