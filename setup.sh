#!/bin/bash

echo "🚀 Setting up tldr environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install basic dependencies
echo "📦 Installing basic dependencies..."
uv sync

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
uv pip install torch --index-url https://download.pytorch.org/whl/cu128

# Activate virtual environment
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "   source .venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "   python example.py"
