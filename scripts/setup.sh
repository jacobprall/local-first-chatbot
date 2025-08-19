#!/bin/bash
# Main Setup Script for Local-First Chatbot
# Orchestrates all setup components

echo "🚀 Local-First Chatbot Setup"
echo "============================="
echo ""
echo "This script will set up your complete development environment:"
echo "  🔌 SQLite Extensions (AI + Vector)"
echo "  🐍 Python Environment (venv + dependencies)"
echo "  🧠 AI Models (optional downloads)"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [[ ! -f "app/main.py" ]]; then
    echo "❌ Error: Could not find app/main.py"
    echo "💡 Please run this script from the project root or scripts directory"
    exit 1
fi

echo "📍 Working directory: $(pwd)"
echo ""

# Function to run a setup script
run_setup_script() {
    local script_name=$1
    local script_path="scripts/$script_name"
    local description=$2
    
    echo "▶️  Starting: $description"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [[ -f "$script_path" && -x "$script_path" ]]; then
        if bash "$script_path"; then
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "✅ Completed: $description"
            echo ""
            return 0
        else
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "❌ Failed: $description"
            echo ""
            return 1
        fi
    else
        echo "❌ Script not found or not executable: $script_path"
        echo ""
        return 1
    fi
}

# Function to show interactive setup menu
show_setup_menu() {
    echo "🎯 Setup Options:"
    echo "1) Complete setup (recommended)"
    echo "2) Extensions only"
    echo "3) Python environment only"
    echo "4) AI models only"
    echo "5) Custom setup (choose components)"
    echo "6) Exit"
    echo ""
}

# Function to run complete setup
run_complete_setup() {
    local failed=0
    
    echo "🔄 Running complete setup..."
    echo ""
    
    # Extensions
    if ! run_setup_script "setup-extensions.sh" "SQLite Extensions Setup"; then
        echo "⚠️  Extensions setup failed, but continuing..."
        failed=1
    fi
    
    # Environment
    if ! run_setup_script "setup-environment.sh" "Python Environment Setup"; then
        echo "❌ Environment setup failed - this is critical!"
        return 1
    fi
    
    # Models (optional)
    if ! run_setup_script "setup-models.sh" "AI Models Setup"; then
        echo "⚠️  Models setup failed, but continuing..."
        failed=1
    fi
    
    return $failed
}

# Function to run custom setup
run_custom_setup() {
    echo "📋 Select components to set up:"
    echo ""
    
    read -p "🔌 Set up SQLite extensions? (Y/n): " setup_extensions
    setup_extensions=${setup_extensions:-Y}
    
    read -p "🐍 Set up Python environment? (Y/n): " setup_environment
    setup_environment=${setup_environment:-Y}
    
    read -p "🧠 Set up AI models? (y/N): " setup_models
    setup_models=${setup_models:-N}
    
    echo ""
    local failed=0
    
    if [[ "$setup_extensions" =~ ^[Yy]$ ]]; then
        if ! run_setup_script "setup-extensions.sh" "SQLite Extensions Setup"; then
            failed=1
        fi
    fi
    
    if [[ "$setup_environment" =~ ^[Yy]$ ]]; then
        if ! run_setup_script "setup-environment.sh" "Python Environment Setup"; then
            echo "❌ Environment setup is critical for the application to work!"
            return 1
        fi
    fi
    
    if [[ "$setup_models" =~ ^[Yy]$ ]]; then
        if ! run_setup_script "setup-models.sh" "AI Models Setup"; then
            failed=1
        fi
    fi
    
    return $failed
}

# Main execution
echo "🎯 How would you like to proceed?"
echo ""

# Check for command line arguments
if [[ $# -gt 0 ]]; then
    case "$1" in
        "--complete"|"-c")
            echo "🔄 Running complete setup from command line..."
            run_complete_setup
            exit $?
            ;;
        "--extensions"|"-e")
            echo "🔌 Running extensions setup only..."
            run_setup_script "setup-extensions.sh" "SQLite Extensions Setup"
            exit $?
            ;;
        "--environment"|"--env")
            echo "🐍 Running environment setup only..."
            run_setup_script "setup-environment.sh" "Python Environment Setup"
            exit $?
            ;;
        "--models"|"-m")
            echo "🧠 Running models setup only..."
            run_setup_script "setup-models.sh" "AI Models Setup"
            exit $?
            ;;
        "--help"|"-h")
            echo "Usage: $0 [option]"
            echo ""
            echo "Options:"
            echo "  --complete, -c     Run complete setup"
            echo "  --extensions, -e   Set up SQLite extensions only"
            echo "  --environment      Set up Python environment only"
            echo "  --models, -m       Set up AI models only"
            echo "  --help, -h         Show this help message"
            echo ""
            echo "Interactive mode (no arguments): Show setup menu"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "💡 Use --help for available options"
            exit 1
            ;;
    esac
fi

# Interactive mode
while true; do
    show_setup_menu
    read -p "Enter your choice (1-6): " choice
    echo ""
    
    case $choice in
        1)
            if run_complete_setup; then
                break
            else
                echo "⚠️  Setup completed with some warnings. Check the output above."
                break
            fi
            ;;
        2)
            run_setup_script "setup-extensions.sh" "SQLite Extensions Setup"
            break
            ;;
        3)
            run_setup_script "setup-environment.sh" "Python Environment Setup"
            break
            ;;
        4)
            run_setup_script "setup-models.sh" "AI Models Setup"
            break
            ;;
        5)
            if run_custom_setup; then
                break
            else
                echo "⚠️  Setup completed with some warnings. Check the output above."
                break
            fi
            ;;
        6)
            echo "👋 Setup cancelled. You can run this script again anytime."
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please enter 1-6."
            echo ""
            ;;
    esac
done

# Final status check
echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""

# Check what's been set up
extensions_ready=false
environment_ready=false
models_ready=false

# Check extensions
if [[ -f "extensions/ai.so" || -f "extensions/ai.dylib" || -f "extensions/ai.dll" ]] && [[ -f "extensions/vector.so" || -f "extensions/vector.dylib" || -f "extensions/vector.dll" ]]; then
    echo "🔌 Extensions: ✅ Both SQLite-AI and SQLite-Vector ready"
    extensions_ready=true
elif [[ -f "extensions/ai.so" || -f "extensions/ai.dylib" || -f "extensions/ai.dll" ]]; then
    echo "🔌 Extensions: ⚠️  SQLite-AI ready, SQLite-Vector missing"
else
    echo "🔌 Extensions: ❌ Not installed"
fi

# Check environment
if [[ -d "venv" && -f "venv/bin/activate" ]]; then
    echo "🐍 Environment: ✅ Python virtual environment ready"
    environment_ready=true
else
    echo "🐍 Environment: ❌ Not set up"
fi

# Check models
model_count=$(find models/ -name "*.gguf" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$model_count" -gt 0 ]]; then
    echo "🧠 Models: ✅ $model_count model(s) available"
    models_ready=true
else
    echo "🧠 Models: ⚠️  No models installed"
fi

echo ""

# Provide next steps
if [[ "$environment_ready" == true ]]; then
    echo "🚀 Ready to start! Next steps:"
    echo ""
    echo "1️⃣  Activate the Python environment:"
    echo "   source venv/bin/activate"
    echo ""
    echo "2️⃣  Start the application:"
    echo "   python -m streamlit run app/main.py"
    echo ""
    
    if [[ "$models_ready" == true ]]; then
        first_model=$(find models/ -name "*.gguf" -exec basename {} \; 2>/dev/null | head -1)
        echo "3️⃣  Or start with a specific model:"
        echo "   MODEL_PATH=./models/$first_model python -m streamlit run app/main.py"
        echo ""
    fi
    
    if [[ "$extensions_ready" == true ]]; then
        echo "🌟 You have full AI and vector search capabilities!"
    elif [[ "$extensions_ready" == false ]]; then
        echo "💡 For full functionality, run: ./scripts/setup-extensions.sh"
    fi
else
    echo "❌ Python environment not ready. Please run:"
    echo "   ./scripts/setup-environment.sh"
fi

echo ""
echo "📚 For more information, check the documentation or run individual setup scripts."
