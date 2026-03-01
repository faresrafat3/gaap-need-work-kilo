# GAAP Development Setup

Setup your development environment for GAAP.

---

## Prerequisites

### Required

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Backend runtime |
| Node.js | 18+ | Frontend runtime |
| Git | 2.30+ | Version control |
| Make | Any | Build automation |

### Optional

| Tool | Purpose |
|------|---------|
| Docker | Containerized development |
| Docker Compose | Multi-service orchestration |
| Redis | Distributed rate limiting |

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/gaap-system/gaap.git
cd gaap
```

### 2. Backend Setup

#### Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

#### Install Dependencies

```bash
# Install in development mode
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements-dev.txt
```

#### Verify Installation

```bash
# Check Python version
python --version  # Should be 3.10+

# Run quick test
python -c "import gaap; print('GAAP imported successfully')"
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Verify installation
npm run typecheck
```

---

## Environment Setup

### Backend Environment

```bash
# Copy example environment
cp .env.example .env

# Edit with your settings
nano .env  # or your preferred editor
```

#### Required Variables

```bash
# Environment
GAAP_ENVIRONMENT=development
GAAP_LOG_LEVEL=DEBUG

# At least one LLM provider API key
GAAP_KILO_API_KEY=your_key_here
# OR
GAAP_OPENAI_API_KEY=your_key_here
```

#### Optional Variables

```bash
# GitHub integration
GAAP_GITHUB_TOKEN=your_github_token

# Security
GAAP_JWT_SECRET=your_jwt_secret
GAAP_CAPABILITY_SECRET=your_capability_secret

# Database (defaults to SQLite)
GAAP_DB_PATH=.gaap/gaap.db

# Rate limiting
GAAP_RATE_LIMIT_PER_MINUTE=60
```

### Frontend Environment

```bash
cd frontend
cp .env.example .env.local
```

#### Required Variables

```bash
# Backend API URL
PYTHON_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

---

## Running Locally

### Option 1: Using Make

```bash
# Start both frontend and backend
make dev

# Start backend only
make dev-backend

# Start frontend only
make dev-frontend
```

### Option 2: Manual Start

#### Terminal 1: Backend

```bash
# From project root
source .venv/bin/activate

# Run with auto-reload
uvicorn gaap.api.main:app --reload --port 8000

# Or using the module
python -m gaap.api.main
```

Backend will be available at: `http://localhost:8000`

API docs at: `http://localhost:8000/docs`

#### Terminal 2: Frontend

```bash
cd frontend

# Development server
npm run dev

# Or with specific port
npm run dev -- --port 3000
```

Frontend will be available at: `http://localhost:3000`

### Option 3: Using Start Script

```bash
cd frontend
./start.sh
```

For production-like environment:

```bash
./start.sh production
```

---

## Docker Development

### Using Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Development Container

```bash
# Build development image
docker build -t gaap-dev -f Dockerfile.backend .

# Run with volume mount
docker run -it -v $(pwd):/app -p 8000:8000 gaap-dev
```

---

## Database Setup

### SQLite (Default)

SQLite is used by default. The database file is created automatically at `~/.gaap/gaap.db`.

```bash
# Initialize database
python -c "from gaap.storage.sqlite_store import init_db; init_db()"
```

### PostgreSQL (Optional)

```bash
# Install psycopg2
pip install psycopg2-binary

# Update .env
DATABASE_URL=postgresql://user:pass@localhost/gaap
```

---

## IDE Setup

### VS Code

Recommended extensions:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "charliermarsh.ruff",
    "ms-python.mypy-type-checker",
    "bradlc.vscode-tailwindcss",
    "esbenp.prettier-vscode"
  ]
}
```

Configuration (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "./.venv/bin/python",
  "python.linting.enabled": true,
  "python.formatting.provider": "ruff",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.ruff": true
  }
}
```

### PyCharm

1. Open project in PyCharm
2. Set Python interpreter to `.venv/bin/python`
3. Enable Ruff plugin
4. Configure mypy integration

---

## Verifying Setup

### Health Check

```bash
# Backend health
curl http://localhost:8000/api/health

# Frontend health
curl http://localhost:3000/api/health
```

### Test Chat Endpoint

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello GAAP!"}'
```

### Run Tests

```bash
# Backend tests
pytest

# Frontend tests
cd frontend && npm test

# With coverage
pytest --cov=gaap --cov-report=html
cd frontend && npm run test:coverage
```

---

## Troubleshooting

### Backend Issues

**ImportError: No module named 'gaap'**
```bash
# Ensure you're in the virtual environment
source .venv/bin/activate

# Reinstall in editable mode
pip install -e .
```

**Port already in use**
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn gaap.api.main:app --port 8001
```

**Database locked (SQLite)**
```bash
# Check for zombie processes
ps aux | grep python

# Kill all python processes related to GAAP
pkill -f "gaap"
```

### Frontend Issues

**Node modules issues**
```bash
# Clear and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Port 3000 in use**
```bash
# Use different port
npm run dev -- --port 3001
```

**Build errors**
```bash
# Clear Next.js cache
rm -rf .next
npm run build
```

### Common Fixes

```bash
# Reset everything
make clean
make install

# Or manually:
rm -rf .venv node_modules .next .pytest_cache
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cd frontend && npm install
```

---

## Next Steps

1. Read the [Architecture Guide](./architecture.md)
2. Check out [Contributing Guidelines](./contributing.md)
3. Review [API Documentation](../api/README.md)
4. Explore the [Examples](../../examples/)
