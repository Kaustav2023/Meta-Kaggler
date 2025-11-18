#!/bin/bash

echo "🤖 Autonomous Kaggle Competition Companion - Demo"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate 2>/dev/null || .venv\Scripts\activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Run in mock mode
echo ""
echo "🚀 Running experiment in MOCK mode..."
echo ""
python main.py --mock --dataset "demo-classification-dataset"

echo ""
echo "✅ Demo complete! Check artifacts/ directory for outputs."
echo ""
