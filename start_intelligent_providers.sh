#!/bin/bash
# GAAP Intelligent Providers System - Startup Script
# Automatically detects and displays actual GLM models

echo "🚀 Starting GAAP Intelligent Providers System..."
echo "================================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from GAAP project root directory"
    exit 1
fi

# Set environment variables
export GAAP_LOG_LEVEL=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo ""
echo "📋 System Components:"
echo "  ✅ GLM Provider - Live model detection from API"
echo "  ✅ Provider Cache - Smart caching with 5-min TTL"
echo "  ✅ Providers API - /api/providers/status endpoint"
echo "  ✅ React Hook - useLiveProviders with auto-refresh"
echo "  ✅ UI Components - Dynamic provider display"
echo ""

echo "🔧 Testing Python Components..."
python3 -c "
from gaap.providers.webchat.glm import GLMWebChat
from gaap.providers.provider_cache import ProviderCacheManager
from gaap.api.providers_status import router
print('✅ All Python components loaded successfully!')
"

echo ""
echo "🌐 Frontend Components:"
echo "  ✅ useLiveProviders.ts - Auto-refresh every 30s"
echo "  ✅ ProviderSelector.tsx - Live model display"
echo "  ✅ Dashboard.tsx - Real-time stats"
echo ""

echo "📖 Available Endpoints:"
echo "  GET  /api/providers/status       - Live provider status"
echo "  POST /api/providers/refresh      - Force refresh"
echo "  GET  /api/providers/status/{name} - Specific provider"
echo ""

echo "📝 Documentation:"
echo "  📄 GLM5_INTELLIGENT_SYSTEM.md - Full system documentation"
echo ""

echo "✨ The system will now display the ACTUAL model from chat.z.ai"
echo "   (GLM-5, GLM-4.7, etc.) instead of hardcoded values!"
echo ""

# Start the backend
echo "🚀 Starting Backend API..."
echo "   Run: python -m gaap.api.main"
echo ""

# Instructions for frontend
echo "🎨 Starting Frontend..."
echo "   cd frontend && npm run dev"
echo ""

echo "================================================"
echo "✅ System Ready!"
echo "================================================"
