#!/bin/bash
# Quick Start Script for Deep Q-Learning Project

echo "================================================"
echo "Deep Q-Learning for Atari - Quick Start"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "================================================"
echo "Installation Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Train a model:"
echo "   python train.py --timesteps 100000"
echo ""
echo "2. Play with trained model:"
echo "   python play.py --model-path ./dqn_model.zip"
echo ""
echo "3. Run hyperparameter experiments:"
echo "   python run_experiments.py --timesteps 500000"
echo ""
echo "4. Generate experiment table:"
echo "   python run_experiments.py --generate-table"
echo ""
echo "5. View training progress with TensorBoard:"
echo "   tensorboard --logdir ./logs/"
echo ""
echo "================================================"
