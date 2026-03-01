# Docker Deployment

Deploy GAAP using Docker and Docker Compose.

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/gaap-system/gaap.git
cd gaap

# Create environment file
cp .env.example .env
# Edit .env with your settings

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

Access GAAP at: `http://localhost:3000`

---

## Docker Compose Configuration

### Services

The `docker-compose.yml` defines two services:

```yaml
version: '3.8'

services:
  gaap-backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    volumes:
      - gaap-data:/app/data
    environment:
      - ENVIRONMENT=production

  gaap-frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - PYTHON_API_URL=http://gaap-backend:8000
    depends_on:
      gaap-backend:
        condition: service_healthy
```

### Services Overview

| Service | Port | Description |
|---------|------|-------------|
| gaap-backend | 8000 | FastAPI backend |
| gaap-frontend | 3000 | Next.js frontend |

### Volumes

| Volume | Purpose |
|--------|---------|
| gaap-data | Persistent database storage |

---

## Dockerfiles

### Backend Dockerfile

Multi-stage build for optimized production image:

```dockerfile
# Stage 1: Builder
FROM python:3.12-slim AS builder
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev

# Install Python packages
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim AS runtime
WORKDIR /app

# Create non-root user
RUN groupadd --gid 1000 gaap && \
    useradd --uid 1000 --gid gaap gaap

# Copy packages from builder
COPY --from=builder /root/.local /home/gaap/.local
ENV PATH=/home/gaap/.local/bin:$PATH

# Switch to non-root
USER gaap

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "gaap.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4"]
```

### Frontend Dockerfile

Multi-stage build for Next.js:

```dockerfile
# Stage 1: Dependencies
FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Stage 2: Builder
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# Stage 3: Runner
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy built application
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s \
    CMD node -e "require('http').get('http://localhost:3000/api/health', \
    (r) => {process.exit(r.statusCode === 200 ? 0 : 1)})"

CMD ["node", "server.js"]
```

---

## Environment Variables

### Backend

| Variable | Description | Default |
|----------|-------------|---------|
| `GAAP_ENVIRONMENT` | Environment mode | `production` |
| `GAAP_LOG_LEVEL` | Log level | `INFO` |
| `GAAP_DB_PATH` | SQLite path | `/app/data/gaap.db` |
| `GAAP_KILO_API_KEY` | LLM API key | - |
| `CORS_ORIGINS` | Allowed origins | - |

### Frontend

| Variable | Description | Default |
|----------|-------------|---------|
| `PYTHON_API_URL` | Backend URL | `http://gaap-backend:8000` |
| `NEXT_PUBLIC_APP_URL` | Frontend URL | - |
| `NODE_ENV` | Node environment | `production` |

### Docker Compose Override

Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.8'

services:
  gaap-backend:
    environment:
      - GAAP_LOG_LEVEL=DEBUG
    volumes:
      - ./gaap:/app/gaap  # Mount for development
```

---

## Production Deployment

### With PostgreSQL

Extend `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: gaap
      POSTGRES_PASSWORD: secure_password
      POSTGRES_DB: gaap
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gaap"]
      interval: 10s
      timeout: 5s
      retries: 5

  gaap-backend:
    environment:
      - DATABASE_URL=postgresql://gaap:secure_password@postgres:5432/gaap
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres-data:
```

### With Redis

```yaml
services:
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

  gaap-backend:
    environment:
      - GAAP_REDIS_URL=redis://redis:6379/0
```

### With Nginx Reverse Proxy

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - gaap-frontend
      - gaap-backend
```

`nginx.conf`:

```nginx
events { worker_connections 1024; }

http {
    upstream backend {
        server gaap-backend:8000;
    }

    upstream frontend {
        server gaap-frontend:3000;
    }

    server {
        listen 80;
        server_name gaap.example.com;

        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
        }
    }
}
```

---

## Commands

### Basic Operations

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View logs
docker-compose logs
docker-compose logs -f gaap-backend
docker-compose logs --tail=100

# Scale backend
docker-compose up -d --scale gaap-backend=3

# Rebuild after code changes
docker-compose up -d --build

# Execute command in container
docker-compose exec gaap-backend python -m pytest
docker-compose exec gaap-frontend npm test
```

### Troubleshooting

```bash
# Check container status
docker-compose ps

# Inspect container
docker-compose exec gaap-backend sh

# Check resource usage
docker stats

# View container logs
docker logs gaap-backend-1

# Restart service
docker-compose restart gaap-backend
```

---

## Security Hardening

### Non-Root Users

Both Dockerfiles create and use non-root users:
- Backend: `gaap` user (UID 1000)
- Frontend: `nextjs` user (UID 1001)

### Read-Only Filesystems

Add to `docker-compose.yml`:

```yaml
services:
  gaap-backend:
    read_only: true
    tmpfs:
      - /tmp
    volumes:
      - gaap-data:/app/data:rw
```

### Security Options

```yaml
services:
  gaap-backend:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
```

---

## Backup and Restore

### Database Backup

```bash
# Backup SQLite
docker-compose exec gaap-backend tar czf - /app/data > backup.tar.gz

# Backup PostgreSQL
docker-compose exec postgres pg_dump -U gaap gaap > backup.sql
```

### Restore

```bash
# Restore SQLite
docker-compose down
tar xzf backup.tar.gz
docker-compose up -d

# Restore PostgreSQL
docker-compose exec -T postgres psql -U gaap gaap < backup.sql
```

---

## Health Checks

Services include built-in health checks:

| Service | Endpoint | Interval |
|---------|----------|----------|
| Backend | `/health` | 30s |
| Frontend | `/api/health` | 30s |
| PostgreSQL | `pg_isready` | 10s |

Check health:

```bash
# Backend
curl http://localhost:8000/health

# Frontend
curl http://localhost:3000/api/health
```

---

## Updating

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose pull
docker-compose up -d --build

# Database migrations (if needed)
docker-compose exec gaap-backend alembic upgrade head
```

---

## Monitoring

### Resource Usage

```bash
# Container stats
docker stats

# Disk usage
docker system df

# Clean up
docker system prune
```

### Logs

```bash
# Centralized logging with Fluentd/ELK
docker-compose logs -f | fluentd
```
