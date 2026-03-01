#!/bin/bash
# GAAP Full System Startup Script
# Starts Backend + Frontend + Database + Monitoring

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           🚀 GAAP Full System Startup                    ║"
echo "║                                                          ║"
echo "║  Starting: Backend + Frontend + DB + Monitoring         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}❌ Error: Must run from GAAP project root${NC}"
    exit 1
fi

# Environment
export PYTHON_API_URL="${PYTHON_API_URL:-http://localhost:8000}"
export DATABASE_URL="${DATABASE_URL:-postgresql+asyncpg://gaap:gaap@localhost:5432/gaap}"
export REDIS_URL="${REDIS_URL:-redis://localhost:6379}"

echo ""
echo -e "${BLUE}📋 Configuration:${NC}"
echo "  Backend URL: $PYTHON_API_URL"
echo "  Database: PostgreSQL"
echo "  Cache: Redis"
echo ""

# Function to check if service is ready
check_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    echo -n "  Waiting for $name..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo -e "${GREEN} ✅${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    echo -e "${RED} ❌ (timeout)${NC}"
    return 1
}

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down...${NC}"
    docker-compose down 2>/dev/null || true
    docker-compose -f docker-compose.monitoring.yml down 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

echo -e "${BLUE}🔧 Step 1: Checking Dependencies${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker.${NC}"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose not found. Please install Docker Compose.${NC}"
    exit 1
fi

echo -e "${GREEN}  ✅ Docker ready${NC}"

# Check Node.js for frontend
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}⚠️  Node.js not found. Frontend will not start.${NC}"
    START_FRONTEND=false
else
    echo -e "${GREEN}  ✅ Node.js ready${NC}"
    START_FRONTEND=true
fi

echo ""
echo -e "${BLUE}🐳 Step 2: Starting Infrastructure (PostgreSQL + Redis)${NC}"
docker-compose up -d postgres redis
echo -e "${GREEN}  ✅ PostgreSQL and Redis started${NC}"

echo ""
echo -e "${BLUE}⏳ Step 3: Waiting for Database${NC}"
sleep 5
check_service "http://localhost:5432" "PostgreSQL" || echo -e "${YELLOW}  ⚠️  PostgreSQL check skipped${NC}"

echo ""
echo -e "${BLUE}🐍 Step 4: Installing Python Dependencies${NC}"
pip install -e . -q
echo -e "${GREEN}  ✅ Python dependencies installed${NC}"

echo ""
echo -e "${BLUE}🔄 Step 5: Running Database Migrations${NC}"
cd /home/fares/Projects/GAAP
alembic upgrade head || echo -e "${YELLOW}  ⚠️  Migration may have already run${NC}"
echo -e "${GREEN}  ✅ Database migrations complete${NC}"

echo ""
echo -e "${BLUE}🐍 Step 6: Starting Python Backend${NC}"
python -m gaap.api.main &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Wait for backend
check_service "http://localhost:8000/health" "Backend API"

echo ""
if [ "$START_FRONTEND" = true ]; then
    echo -e "${BLUE}🎨 Step 7: Starting Frontend${NC}"
    cd /home/fares/Projects/GAAP/frontend
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "  Installing frontend dependencies..."
        npm install
    fi
    
    npm run dev &
    FRONTEND_PID=$!
    echo "  Frontend PID: $FRONTEND_PID"
    
    # Wait for frontend
    check_service "http://localhost:3000" "Frontend"
else
    echo -e "${YELLOW}⚠️  Step 7: Skipping Frontend (Node.js not found)${NC}"
fi

echo ""
echo -e "${BLUE}📊 Step 8: Starting Monitoring Stack${NC}"
cd /home/fares/Projects/GAAP
docker-compose -f docker-compose.monitoring.yml up -d
echo -e "${GREEN}  ✅ Monitoring stack started${NC}"

# Wait for monitoring
check_service "http://localhost:9090" "Prometheus"
check_service "http://localhost:3001" "Grafana"

echo ""
echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              ✅ All Systems Operational!                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo -e "${BLUE}🌐 Access Points:${NC}"
echo "  🌐 Web App:       http://localhost:3000"
echo "  🔧 API:           http://localhost:8000"
echo "  📚 API Docs:      http://localhost:8000/docs"
echo "  📊 Grafana:       http://localhost:3001 (admin/admin)"
echo "  📈 Prometheus:    http://localhost:9090"
echo "  🚨 AlertManager:  http://localhost:9093"
echo ""

echo -e "${BLUE}📋 Health Checks:${NC}"
echo "  Backend:  curl http://localhost:8000/health"
echo "  Frontend: curl http://localhost:3000/api/health"
echo ""

echo -e "${BLUE}🛠️ Useful Commands:${NC}"
echo "  View logs:        docker-compose logs -f"
echo "  Stop all:         docker-compose down"
echo "  Run tests:        pytest tests/ -v"
echo "  Database shell:   docker-compose exec postgres psql -U gaap"
echo ""

echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Keep script running
wait
