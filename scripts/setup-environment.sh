#!/bin/bash
# Python Environment Setup Script
# Sets up virtual environment and installs dependencies

echo "🐍 Python Environment Setup"
echo "============================"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    echo "💡 Please install Python 3.8 or later"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🔍 Detected Python version: $python_version"

# Verify minimum version (3.8+)
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✅ Python version is compatible"
else
    echo "❌ Python 3.8 or later is required"
    echo "💡 Current version: $python_version"
    exit 1
fi

# Check if requirements.txt exists
if [[ ! -f "requirements.txt" ]]; then
    echo "❌ requirements.txt not found"
    echo "💡 Please ensure you're running this from the project root directory"
    exit 1
fi

echo ""
echo "📦 Virtual Environment Setup"
echo "============================"

# Check if virtual environment already exists
if [[ -d "venv" ]]; then
    echo "📁 Virtual environment already exists"
    read -p "🔄 Recreate virtual environment? (y/N): " recreate
    recreate=${recreate:-N}
    
    if [[ "$recreate" =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing virtual environment..."
        rm -rf venv
    else
        echo "✅ Using existing virtual environment"
    fi
fi

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    echo "📦 Creating virtual environment..."
    if python3 -m venv venv; then
        echo "✅ Virtual environment created successfully"
    else
        echo "❌ Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Verify activation
if [[ "$VIRTUAL_ENV" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
echo ""
echo "📈 Package Management Setup"
echo "=========================="
echo "📈 Upgrading pip..."
if pip install --upgrade pip; then
    echo "✅ pip upgraded successfully"
else
    echo "⚠️  pip upgrade failed, continuing with existing version"
fi

# Install wheel for better package installation
echo "🔧 Installing wheel for better package builds..."
pip install wheel

# Install dependencies
echo ""
echo "📥 Installing Dependencies"
echo "========================="
echo "📥 Installing packages from requirements.txt..."

if pip install -r requirements.txt; then
    echo "✅ All dependencies installed successfully"
else
    echo "❌ Some dependencies failed to install"
    echo "💡 You may need to install system dependencies or check requirements.txt"
    exit 1
fi

# Create data directory and initialize database
echo ""
echo "📁 Directory Setup"
echo "=================="
echo "📁 Creating data directory..."
mkdir -p data/
echo "✅ Data directory ready"

echo "🗄️  Initializing database..."
python -c "
from app.services.chatbot import Chatbot
try:
    # Initialize the database with proper schema
    chatbot = Chatbot('data/chat.db')
    print('✅ Database initialized with proper schema')
except Exception as e:
    print(f'⚠️  Database initialization failed: {e}')
    print('💡 Database will be created when the app first runs')
"

# Verify installation
echo ""
echo "🔍 Installation Verification"
echo "============================"

# Check key packages
key_packages=("streamlit" "sqlite3")
all_good=true

for package in "${key_packages[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        echo "✅ $package: Available"
    else
        echo "❌ $package: Not available"
        all_good=false
    fi
done

# Show installed packages
echo ""
echo "📋 Installed Packages:"
pip list --format=freeze | head -10
if [[ $(pip list --format=freeze | wc -l) -gt 10 ]]; then
    echo "... and $(( $(pip list --format=freeze | wc -l) - 10 )) more packages"
fi

echo ""
echo "✅ Environment setup complete!"
echo ""

if [[ "$all_good" == true ]]; then
    echo "🌟 Python environment is ready for development!"
    echo ""
    echo "🎯 To activate the environment manually:"
    echo "   source venv/bin/activate"
    echo ""
    echo "🚀 To start the application:"
    echo "   source venv/bin/activate"
    echo "   python -m streamlit run app/main.py"
else
    echo "⚠️  Some packages may not be working correctly"
    echo "💡 Check the error messages above and requirements.txt"
fi
