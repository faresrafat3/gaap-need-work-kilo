#!/bin/bash

# GAAP Frontend Startup Script
# Usage: ./start.sh [development|production]

set -e

ENV=${1:-development}

echo "🚀 Starting GAAP Frontend in $ENV mode..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+"
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js 18+ is required. Current version: $(node -v)"
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm ci
fi

# Setup environment
if [ "$ENV" = "production" ]; then
    echo "🔧 Setting up production environment..."
    
    # Check required environment variables
    if [ -z "$PYTHON_API_URL" ]; then
        echo "❌ PYTHON_API_URL is not set!"
        echo "   Please set it: export PYTHON_API_URL=https://api.yourdomain.com"
        exit 1
    fi
    
    if [ -z "$NEXT_PUBLIC_APP_URL" ]; then
        echo "❌ NEXT_PUBLIC_APP_URL is not set!"
        echo "   Please set it: export NEXT_PUBLIC_APP_URL=https://yourdomain.com"
        exit 1
    fi
    
    # Build the application
    echo "🏗️  Building application..."
    npm run build
    
    # Start production server
    echo "✅ Starting production server..."
    npm start
    
else
    echo "🔧 Setting up development environment..."
    
    # Check if .env.local exists
    if [ ! -f ".env.local" ]; then
        echo "⚠️  .env.local not found. Copying from .env.example..."
        cp .env.example .env.local
        echo "   Please review and update .env.local with your settings"
    fi
    
    # Start development server
    echo "✅ Starting development server..."
    npm run dev
fi
