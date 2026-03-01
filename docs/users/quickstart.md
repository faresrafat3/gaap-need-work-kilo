# GAAP Quick Start Guide

Get GAAP running in 5 minutes.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Git

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/gaap-system/gaap.git
cd gaap
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Verify
python -c "import gaap; print('GAAP installed successfully')"
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Environment Configuration

Backend:
```bash
cp .env.example .env
# Edit .env with your API keys
```

Frontend:
```bash
cp .env.example .env.local
# Edit with backend URL
```

### 5. Start Services

Terminal 1 - Backend:
```bash
uvicorn gaap.api.main:app --reload --port 8000
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

Open http://localhost:3000 in your browser.

## First Chat

### Using Web Interface

1. Open http://localhost:3000
2. Type your message in the chat box
3. Press Enter or click Send
4. Watch the AI response stream in real-time

### Using API

```bash
# Health check
curl http://localhost:8000/api/health

# Send a message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello GAAP!"}'
```

### Using Python

```python
import requests

# Chat with GAAP
response = requests.post(
    "http://localhost:8000/api/chat",
    json={"message": "Write a hello world in Python"}
)
print(response.json()["response"])
```

## Common Tasks

### Create a Session

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Project",
    "description": "Working on feature X",
    "priority": "high"
  }'
```

### List Sessions

```bash
curl http://localhost:8000/api/sessions
```

### Check Provider Status

```bash
curl http://localhost:8000/api/providers
```

## Docker Quick Start

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Troubleshooting

### Backend won't start

```bash
# Check Python version
python --version  # Must be 3.10+

# Check dependencies
pip list | grep gaap

# Port in use
lsof -i :8000
```

### Frontend won't start

```bash
# Clear and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### API connection failed

```bash
# Check backend is running
curl http://localhost:8000/api/health

# Check frontend env
# .env.local should have:
# PYTHON_API_URL=http://localhost:8000
```

## Next Steps

- [Configure Providers](./providers.md) - Set up AI providers
- [Web Interface Guide](./web-interface.md) - Full UI walkthrough
- [Developer Guide](../developers/README.md) - Build with GAAP
