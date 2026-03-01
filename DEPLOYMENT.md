# GAAP Deployment Guide

## Prerequisites

- Node.js 18+
- Python 3.10+ (for backend)
- Docker & Docker Compose (optional)
- Git

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/gaap.git
cd gaap
```

### 2. Environment Setup

#### Frontend
```bash
cd frontend
cp .env.example .env.local
# Edit .env.local with your settings
```

#### Backend
```bash
cd ..
cp .env.example .env
# Edit .env with your settings
```

### 3. Development Mode

#### Option A: Using start script
```bash
cd frontend
./start.sh
```

#### Option B: Manual start
```bash
# Terminal 1: Backend
cd gaap
uvicorn gaap.api.main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

### 4. Production Deployment

#### Using Docker Compose (Recommended)
```bash
docker-compose up -d
```

This will:
- Build and start the Python backend on port 8000
- Build and start the Next.js frontend on port 3000
- Connect them via internal network

#### Manual Production Build
```bash
cd frontend
export PYTHON_API_URL=https://api.yourdomain.com
export NEXT_PUBLIC_APP_URL=https://yourdomain.com
./start.sh production
```

## Environment Variables

### Required
- `PYTHON_API_URL` - Backend API URL
- `NEXT_PUBLIC_APP_URL` - Frontend URL

### Optional
- `API_TIMEOUT` - Request timeout (default: 30000ms)
- `RATE_LIMIT_REQUESTS_PER_MINUTE` - Rate limit (default: 60)
- `NEXT_PUBLIC_SENTRY_DSN` - Sentry error tracking

## Troubleshooting

### Backend Connection Issues
1. Check backend is running: `curl http://localhost:8000/health`
2. Verify `PYTHON_API_URL` is correct
3. Check firewall settings

### Build Errors
1. Clear node_modules: `rm -rf node_modules && npm install`
2. Clear Next.js cache: `rm -rf .next`
3. Run TypeScript check: `npm run typecheck`

### Rate Limiting
- Default: 60 requests/minute per IP
- Adjust with `RATE_LIMIT_REQUESTS_PER_MINUTE`

## Monitoring

- Health check: `/api/health`
- Logs: `docker-compose logs -f gaap-frontend`
- Analytics: Vercel Analytics (if enabled)
