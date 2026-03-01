#!/bin/bash
# GAAP Web App - Full Stack Startup Script

set -e

echo "🚀 GAAP Web App - Full Stack Startup"
echo "======================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from GAAP project root directory"
    exit 1
fi

# Set environment
export PYTHON_API_URL="${PYTHON_API_URL:-http://localhost:8000}"
export NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-http://localhost:8000}"

echo -e "${BLUE}📋 Configuration:${NC}"
echo "  Backend URL: $PYTHON_API_URL"
echo "  Frontend will proxy to backend"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM

# Check Python dependencies
echo -e "${BLUE}🔧 Checking Python dependencies...${NC}"
python3 -c "import gaap" 2>/dev/null || {
    echo "Installing GAAP package..."
    pip install -e . -q
}
echo -e "${GREEN}  ✅ Python ready${NC}"
echo ""

# Check Node.js dependencies
echo -e "${BLUE}🔧 Checking Node.js dependencies...${NC}"
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi
echo -e "${GREEN}  ✅ Node.js ready${NC}"
echo ""

# Start Backend
echo -e "${YELLOW}🐍 Starting Python Backend...${NC}"
cd /home/fares/Projects/GAAP
python3 -m gaap.api.main &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"
echo "  Waiting for backend to be ready..."

# Wait for backend
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}  ✅ Backend ready at http://localhost:8000${NC}"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Start Frontend
echo -e "${YELLOW}🎨 Starting Next.js Frontend...${NC}"
cd /home/fares/Projects/GAAP/frontend
npm run dev &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"
echo "  Waiting for frontend to be ready..."

# Wait for frontend
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo -e "${GREEN}  ✅ Frontend ready at http://localhost:3000${NC}"
        break
    fi
    sleep 1
    echo -n "."
done
echo ""

# Summary
echo ""
echo "======================================"
echo -e "${GREEN}✅ Both services are running!${NC}"
echo "======================================"
echo ""
echo "📱 Access Points:"
echo "  🌐 Web App:     http://localhost:3000"
echo "  🔧 Backend API: http://localhost:8000"
echo "  📊 API Docs:    http://localhost:8000/docs"
echo ""
echo "🔍 Health Checks:"
echo "  Backend:  curl http://localhost:8000/health"
echo "  Frontend: curl http://localhost:3000/api/health"
echo ""
echo "📁 Key Endpoints:"
echo "  Providers: http://localhost:3000/api/providers/live"
echo "  Sessions:  http://localhost:3000/api/sessions"
echo "  Chat:      http://localhost:3000/api/chat"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for both processes
wait
