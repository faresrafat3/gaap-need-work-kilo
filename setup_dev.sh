#!/bin/bash
# GAAP Development Setup Script
# Run this to set up your development environment

set -e

echo "🚀 Setting up GAAP development environment..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install package in development mode
echo "📚 Installing GAAP in development mode..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pip install pre-commit
pre-commit install

# Create .gaap_env if it doesn't exist
if [ ! -f ".gaap_env" ]; then
    echo "📝 Creating .gaap_env template..."
    cat > .gaap_env << 'EOF'
# GAAP Environment Variables
# Add your API keys here

# GROQ_API_KEY=gsk_...
# GEMINI_API_KEY=...
# MISTRAL_API_KEY=...
# CEREBRAS_API_KEY=...
# GITHUB_TOKEN=...
EOF
    echo "⚠️  Please edit .gaap_env and add your API keys"
fi

# Run quick verification
echo "🔍 Running verification..."
python -c "from gaap import GAAPEngine; print('✓ GAAP imports work')"

# Run tests
echo "🧪 Running tests..."
pytest tests/unit/ -v --tb=short -q

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .gaap_env with your API keys"
echo "  2. Run 'source .venv/bin/activate' to activate the environment"
echo "  3. Run 'pytest' to run tests"
echo "  4. Run 'black gaap/ tests/' to format code"
echo "  5. Run 'mypy gaap/' to type check"
echo ""