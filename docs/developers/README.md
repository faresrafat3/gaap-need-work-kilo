# GAAP Developer Guide

Guide for developers contributing to GAAP.

---

## Project Overview

GAAP (General AI Assistant Platform) is an autonomous AI coding agent built with a 4-layer OODA cognitive architecture. It combines Deep Research, Self-Healing, Meta-Learning, and Swarm Intelligence into a unified cognitive system.

### Key Features

- **4-Layer OODA Architecture**: Observe-Orient-Decide-Act cognitive cycle
- **Multi-Provider Support**: Kimi, DeepSeek, GLM, and more
- **Self-Healing**: Automatic error recovery
- **Swarm Intelligence**: Distributed agent collaboration
- **Web Interface**: Next.js frontend with real-time updates

### Architecture Highlights

```
┌─────────────────────────────────────────────┐
│              User Interface                  │
│         (Web / CLI / API)                   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           4-Layer OODA System                │
│  ┌─────────┬─────────┬─────────┬─────────┐  │
│  │ Layer 0 │ Layer 1 │ Layer 2 │ Layer 3 │  │
│  │Interface│Strategic│Tactical │Execution│  │
│  │Observe  │ Orient  │ Decide  │   Act   │  │
│  └─────────┴─────────┴─────────┴─────────┘  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           Supporting Systems                 │
│  ┌────────┐ ┌────────┐ ┌────────┐          │
│  │ Memory │ │  Swarm │ │Security│          │
│  │ System │ │Intel.  │ │ Layer  │          │
│  └────────┘ └────────┘ └────────┘          │
└─────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend

| Component | Technology |
|-----------|------------|
| Framework | FastAPI (Python 3.12) |
| Database | SQLite / PostgreSQL |
| WebSocket | Native FastAPI WebSocket |
| Task Queue | Asyncio / Celery (optional) |
| Testing | pytest |
| Linting | Ruff |
| Types | mypy |

### Frontend

| Component | Technology |
|-----------|------------|
| Framework | Next.js 16 |
| Language | TypeScript 5 |
| Styling | Tailwind CSS 3 |
| UI Components | shadcn/ui |
| State | Zustand |
| Animation | Framer Motion |
| Testing | Vitest + Testing Library |

### Infrastructure

| Component | Technology |
|-----------|------------|
| Container | Docker |
| Orchestration | Docker Compose / Kubernetes |
| Monitoring | Prometheus + Grafana |
| Logging | Structured JSON logs |

---

## Documentation Sections

- [Setup Guide](./setup.md) - Development environment setup
- [Architecture](./architecture.md) - System architecture details
- [Contributing](./contributing.md) - Contribution guidelines

---

## Quick Links

- [API Documentation](../api/README.md)
- [Deployment Guide](../deployment/README.md)
- [User Guide](../users/README.md)
- [Security Policy](../../SECURITY.md)

---

## Getting Help

- 📧 Email: dev@gaap.io
- 💬 [GitHub Discussions](https://github.com/gaap-system/gaap/discussions)
- 🐛 [GitHub Issues](https://github.com/gaap-system/gaap/issues)

## License

MIT License - see [LICENSE](../../LICENSE) for details.
