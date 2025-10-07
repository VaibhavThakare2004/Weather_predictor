#!/usr/bin/env bash
# build.sh - Render build script

set -o errexit  # exit on error

echo "🚀 Starting Render deployment..."

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Python dependencies installed"

# Navigate to frontend directory and install Node dependencies
if [ -d "frontend" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    
    echo "🏗️ Building React frontend..."
    npm run build
    
    echo "✅ Frontend built successfully"
    cd ..
else
    echo "⚠️ Frontend directory not found, skipping frontend build"
fi

echo "🎉 Build completed successfully!"