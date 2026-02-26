#!/bin/bash

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting GAAP - All in One${NC}\n"

# Kill existing processes
pkill -f "uvicorn" 2>/dev/null
pkill -f "vite" 2>/dev/null
sleep 1

# Start Backend
echo -e "${YELLOW}📡 Starting Backend (port 8000)...${NC}"
cd /home/fares/Projects/GAAP
uvicorn gaap.api.main:app --reload --port 8000 &
BACKEND_PID=$!

sleep 3

# Start Frontend
echo -e "${YELLOW}🌐 Starting Frontend (port 3000)...${NC}"
cd /home/fares/Projects/GAAP/frontend
npm run dev &
FRONTEND_PID=$!

echo -e "\n${GREEN}✅ GAAP is running!${NC}\n"
echo -e "  📡 Backend:  http://localhost:8000"
echo -e "  🌐 Frontend: http://localhost:3000"
echo -e "  📖 API Docs: http://localhost:8000/docs\n"
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop all servers\n"

# Wait for Ctrl+C
trap "echo -e '\n${YELLOW}🛑 Stopping servers...${NC}'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
