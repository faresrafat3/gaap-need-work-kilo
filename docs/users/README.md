# GAAP User Guide

Welcome to GAAP (General AI Assistant Platform) - your autonomous AI coding companion.

## What is GAAP?

GAAP is an intelligent AI assistant designed to help with software development tasks. It uses a cognitive architecture called the **OODA loop** (Observe-Orient-Decide-Act) to understand your requests and execute complex multi-step tasks.

```
┌─────────────────────────────────────────────────────────────┐
│                     GAAP at a Glance                         │
├─────────────────────────────────────────────────────────────┤
│  🤖 Multi-provider AI support (Kimi, DeepSeek, GLM)        │
│  🧠 4-layer cognitive architecture (OODA)                    │
│  🔄 Self-healing error recovery                            │
│  💬 Real-time chat with streaming responses                │
│  📊 Session management and history                         │
│  🔒 Built-in security with 7-layer firewall                │
│  📈 Usage tracking and budget controls                     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Links

- [Quick Start](./quickstart.md) - Get up and running in 5 minutes
- [Provider Configuration](./providers.md) - Set up AI providers
- [Web Interface](./web-interface.md) - Using the web UI

## Key Concepts

### Sessions

A **session** is a conversation or task context with GAAP. Each session:
- Has a unique ID for continuity
- Maintains conversation history
- Tracks token usage and costs
- Can be paused, resumed, or exported

### Providers

**Providers** are the AI models that power GAAP:
- **Kimi** (default): kimi-k2.5-thinking
- **DeepSeek**: deepseek-chat
- **GLM**: GLM-5

GAAP automatically falls back to available providers if the primary fails.

### OODA Loop

GAAP's cognitive cycle:

1. **Observe**: Understand your request
2. **Orient**: Plan the approach
3. **Decide**: Break down into tasks
4. **Act**: Execute and self-heal

## Use Cases

### Code Generation
```
User: "Create a Python function to validate email addresses"
GAAP: Generates complete, tested function with docstrings
```

### Code Review
```
User: "Review this authentication module for security issues"
GAAP: Analyzes code and reports findings with suggestions
```

### Architecture Design
```
User: "Design a scalable microservices architecture for e-commerce"
GAAP: Creates comprehensive design with diagrams
```

### Research
```
User: "Research the latest React patterns for 2024"
GAAP: Compiles findings with code examples
```

### Debugging
```
User: "Help debug this error: Connection refused on port 5432"
GAAP: Diagnoses and suggests solutions
```

## Getting Started

See [Quick Start Guide](./quickstart.md) for detailed setup instructions.

## Support

- 📚 [Documentation](./README.md)
- 💬 [GitHub Discussions](https://github.com/gaap-system/gaap/discussions)
- 🐛 [Issue Tracker](https://github.com/gaap-system/gaap/issues)

## Next Steps

1. [Install GAAP](./quickstart.md#installation)
2. [Configure Providers](./providers.md)
3. [Try the Web Interface](./web-interface.md)
4. [Read the Developer Guide](../developers/README.md)
