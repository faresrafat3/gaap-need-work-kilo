# GAAP CLI Guide

Complete reference for the GAAP command-line interface.

## Installation

```bash
pip install -e ".[dev]"
gaap --help
```

---

## Commands Reference

### gaap chat

Quick one-shot chat with GAAP.

```bash
# Basic usage
gaap chat "Write a Python function for binary search"

# Specify model
gaap chat "Explain async/await" --model llama-3.3-70b

# Set budget
gaap chat "Complex task" --budget 5.0

# JSON output
gaap chat "Hello" --format json
```

**Options:**
- `--model, -m`: Model to use
- `--budget, -b`: Budget limit in USD
- `--format, -f`: Output format (text, json)
- `--provider, -p`: Provider to use

---

### gaap interactive

Start an interactive chat session.

```bash
gaap interactive
```

**Session Commands:**
```
/help          Show help
/clear         Clear conversation
/history       Show conversation history
/model NAME    Switch model
/budget AMOUNT Set budget
/save FILE     Save conversation
/load FILE     Load conversation
/quit          Exit session
```

**Example Session:**
```
$ gaap interactive
GAAP Interactive Mode v1.0.0
Model: llama-3.3-70b-versatile | Budget: $10.00

> Write a function to sort a list

def sort_list(items):
    return sorted(items)

> Now add error handling

def sort_list(items):
    if not isinstance(items, list):
        raise TypeError("Expected a list")
    return sorted(items)

> /save conversation.json
Saved to conversation.json

> /quit
Goodbye!
```

---

### gaap providers

Manage LLM providers.

#### List Providers

```bash
gaap providers list
```

Output:
```
Provider        Status    Models              Priority
groq            active    3 models            85
cerebras        active    2 models            95
gemini          active    2 models            40
mistral         active    3 models            70
github          active    3 models            60
g4f             active    10+ models          50
```

#### Test Provider

```bash
# Test specific provider
gaap providers test groq

# Test all providers
gaap providers test --all

# Detailed output
gaap providers test groq --verbose
```

Output:
```
Testing groq...
  Connection: OK
  Authentication: OK
  Model llama-3.3-70b-versatile: OK (227ms)
  Model llama-3.1-8b-instant: OK (152ms)
  
Status: PASSED
Latency: 190ms average
```

#### Provider Info

```bash
gaap providers info groq
```

---

### gaap models

View and manage available models.

#### List Models

```bash
gaap models list
```

Output:
```
Model                        Provider    Context    Cost/1M
llama-3.3-70b-versatile      groq        128K      $0.00
llama3.3-70b                 cerebras    128K      $0.00
gemini-1.5-flash             gemini      1M        $0.00
mistral-large-latest         mistral     32K       $0.00
gpt-4o-mini                  github      128K      $0.00
```

#### Model Tiers

```bash
gaap models tiers
```

Output:
```
Tier 1 (Strategic):
  - claude-3-5-opus
  - gpt-4o
  - gemini-1.5-pro

Tier 2 (Tactical):
  - llama-3.3-70b-versatile
  - mistral-large
  - gpt-4o-mini

Tier 3 (Efficient):
  - llama-3.1-8b
  - mistral-small

Tier 4 (Private):
  - ollama models
```

#### Model Info

```bash
gaap models info llama-3.3-70b-versatile
```

Output:
```
Model: llama-3.3-70b-versatile
Provider: groq
Context Window: 128,000 tokens
Max Output: 4,096 tokens
Cost: Free (tier limit)
Capabilities: text-generation, function-calling
Average Latency: 227ms
```

---

### gaap config

Configuration management.

#### Show Configuration

```bash
gaap config show
```

Output:
```
System:
  name: GAAP-Production-Alpha
  environment: production
  log_level: INFO

Budget:
  monthly_limit: $5000.00
  daily_limit: $200.00
  per_task_limit: $10.00

Execution:
  max_parallel_tasks: 10
  genetic_twin_enabled: true
  self_healing_enabled: true

Providers:
  - groq (priority: 85)
  - cerebras (priority: 95)
```

#### Set Configuration

```bash
gaap config set default_budget 20.0
gaap config set system.log_level DEBUG
gaap config set execution.max_parallel_tasks 5
```

#### Get Configuration

```bash
gaap config get system.log_level
gaap config get budget.monthly_limit
```

#### Load Configuration

```bash
gaap config load ./config.yaml
gaap config load ./config.json
```

---

### gaap history

View and manage request history.

#### List History

```bash
# Recent requests
gaap history list

# With limit
gaap history list --limit 50

# Filter by date
gaap history list --since "2026-02-01"
```

Output:
```
ID              Time                 Status    Cost      Model
req_abc123      2026-02-16 10:30    SUCCESS   $0.023    llama-3.3-70b
req_def456      2026-02-16 10:25    SUCCESS   $0.015    llama-3.3-70b
req_ghi789      2026-02-16 10:20    FAILED    $0.000    gemini-1.5-flash
```

#### Search History

```bash
gaap history search "binary"
gaap history search "function" --limit 10
```

#### Show Request Details

```bash
gaap history show req_abc123
```

Output:
```
Request: req_abc123
Time: 2026-02-16 10:30:45
Duration: 1.23s

Input:
  Write a Python function for binary search

Output:
  def binary_search(arr, target):
      left, right = 0, len(arr) - 1
      while left <= right:
          mid = (left + right) // 2
          if arr[mid] == target:
              return mid
          elif arr[mid] < target:
              left = mid + 1
          else:
              right = mid - 1
      return -1

Metrics:
  Model: llama-3.3-70b-versatile
  Provider: groq
  Tokens: 150 (input: 20, output: 130)
  Cost: $0.023
  Quality Score: 92
```

#### Clear History

```bash
gaap history clear
gaap history clear --confirm
```

---

### gaap doctor

Run system diagnostics.

```bash
gaap doctor
```

Output:
```
GAAP System Diagnostics
========================

System:
  Python Version: 3.12.0
  Platform: Linux
  GAAP Version: 1.0.0

Configuration:
  Status: OK
  Environment: production
  Log Level: INFO

Providers:
  groq:       OK (227ms)
  cerebras:   OK (511ms)
  gemini:     OK (384ms)
  mistral:    OK (603ms)
  github:     OK (1500ms)
  g4f:        OK (variable)

Environment:
  GROQ_API_KEY: Set
  CEREBRAS_API_KEY: Set
  GEMINI_API_KEY: Set
  MISTRAL_API_KEY: Set

Memory:
  Working Memory: 0/100 items
  Episodic Memory: 0 episodes
  Semantic Memory: 0 rules

Security:
  Firewall: Enabled (strictness: high)
  Audit Trail: Enabled

All systems operational!
```

**Options:**
- `--verbose, -v`: Detailed output
- `--fix`: Attempt to fix issues

---

### gaap status

Show system status.

```bash
gaap status
```

Output:
```
GAAP Status
===========
Version: 1.0.0
Uptime: 2 hours 30 minutes
Requests: 1,234 total (98.5% success)
Active Tasks: 2
Memory Usage: 512 MB
Budget Used: $12.34 / $100.00
```

---

### gaap version

Show version information.

```bash
gaap version
```

Output:
```
GAAP v1.0.0
Python 3.12.0
Platform: Linux-6.1.0-x86_64
```

---

### gaap web

Start the web interface.

```bash
# Default port (8501)
gaap web

# Custom port
gaap web --port 8502

# External access
gaap web --host 0.0.0.0 --port 8501
```

Options:
- `--host`: Host to bind (default: localhost)
- `--port`: Port to use (default: 8501)
- `--headless`: Run without opening browser

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GAAP_ENVIRONMENT` | Environment | production |
| `GAAP_LOG_LEVEL` | Log level | INFO |
| `GAAP_BUDGET_MONTHLY` | Monthly budget | 5000.0 |
| `GAAP_BUDGET_DAILY` | Daily budget | 200.0 |
| `GROQ_API_KEY` | Groq API key | - |
| `CEREBRAS_API_KEY` | Cerebras API key | - |
| `GEMINI_API_KEY` | Gemini API key | - |
| `MISTRAL_API_KEY` | Mistral API key | - |
| `GITHUB_TOKEN` | GitHub token | - |

---

## Configuration File

Create `~/.gaap/config.yaml`:

```yaml
system:
  name: MyGAAP
  environment: development
  log_level: DEBUG

budget:
  monthly_limit: 1000
  daily_limit: 50

execution:
  max_parallel_tasks: 5
  genetic_twin_enabled: true

providers:
  - name: groq
    api_key: ${GROQ_API_KEY}
    priority: 85
    enabled: true
    
  - name: cerebras
    api_key: ${CEREBRAS_API_KEY}
    priority: 95
    enabled: true
```

---

## Shell Completion

### Bash

```bash
# Add to ~/.bashrc
eval "$(_GAAP_COMPLETE=bash_source gaap)"
```

### Zsh

```bash
# Add to ~/.zshrc
eval "$(_GAAP_COMPLETE=zsh_source gaap)"
```

### Fish

```bash
# Add to ~/.config/fish/completions/gaap.fish
eval (env _GAAP_COMPLETE=fish_source gaap)
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Provider error |
| 4 | Rate limit exceeded |
| 5 | Budget exceeded |
| 6 | Security violation |
| 7 | Task timeout |
| 130 | Interrupted (Ctrl+C) |

---

## Tips & Tricks

### Quick One-Liner

```bash
# Quick question
gaap chat "What is 2+2?"  # Output: 4

# Quick code generation
gaap chat "Write a hello world in Python" | python
```

### Pipe Input

```bash
# Pipe file content
cat requirements.txt | gaap chat "Suggest improvements for these dependencies"

# Pipe command output
ls -la | gaap chat "Explain these files"
```

### Output Processing

```bash
# JSON output for scripting
gaap chat "List 5 colors" --format json | jq '.output'

# Save output
gaap chat "Write a script" > script.py
```

### Batch Processing

```bash
# Process multiple inputs
for file in *.txt; do
    gaap chat "Summarize: $(cat $file)" > "${file%.txt}.summary.txt"
done
```

---

## Next Steps

- [API Reference](API_REFERENCE.md) - Programmatic usage
- [Providers Guide](PROVIDERS.md) - Provider setup
- [Examples](examples/) - Code examples