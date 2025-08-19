#!/bin/bash
# AI Models Setup Script
# Downloads sample AI models from Hugging Face

echo "🧠 AI Models Setup"
echo "=================="

# Check if curl is available
if ! command -v curl &> /dev/null; then
    echo "❌ curl is required but not installed"
    exit 1
fi

# Create models directory
echo "📁 Creating models directory..."
mkdir -p models/
echo "✅ Models directory ready"

# Function to download TinyLlama from Hugging Face
download_tinyllama() {
    local model_name="TinyLlama-1.1B-Chat-v1.0"
    local filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    local url="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/${filename}"
    local output_path="models/${filename}"
    
    echo "🔄 Downloading TinyLlama 1.1B Chat..."
    echo "   Source: ${url}"
    echo "   Size: ~637MB"
    echo "   This may take a few minutes..."
    
    # Check if file already exists
    if [[ -f "$output_path" ]]; then
        echo "✅ TinyLlama model already exists: ${filename}"
        return 0
    fi
    
    if curl -L --progress-bar --fail -o "$output_path" "$url"; then
        echo "✅ TinyLlama model downloaded: ${filename}"
        echo "🧠 Model ready for chat and text generation"
        
        # Verify file size (should be around 637MB)
        local file_size=$(stat -f%z "$output_path" 2>/dev/null || stat -c%s "$output_path" 2>/dev/null)
        if [[ -n "$file_size" && "$file_size" -gt 600000000 ]]; then
            echo "✅ File size verification passed"
        else
            echo "⚠️  Downloaded file seems smaller than expected - may be incomplete"
        fi
    else
        echo "❌ Failed to download TinyLlama model"
        echo "💡 You can manually download from: $url"
        rm -f "$output_path"  # Clean up partial download
    fi
}

# Function to download Phi-3.5 Mini from Hugging Face
download_phi3_mini() {
    local model_name="Phi-3.5-mini-instruct"
    local filename="Phi-3.5-mini-instruct-Q4_K_M.gguf"
    local url="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/${filename}"
    local output_path="models/${filename}"
    
    echo "🔄 Downloading Phi-3.5 Mini Instruct..."
    echo "   Source: ${url}"
    echo "   Size: ~2.2GB"
    echo "   This may take 10-15 minutes..."
    
    # Check if file already exists
    if [[ -f "$output_path" ]]; then
        echo "✅ Phi-3.5 Mini model already exists: ${filename}"
        return 0
    fi
    
    if curl -L --progress-bar --fail -o "$output_path" "$url"; then
        echo "✅ Phi-3.5 Mini model downloaded: ${filename}"
        echo "🧠 High-quality model ready for advanced tasks"
        
        # Verify file size (should be around 2.2GB)
        local file_size=$(stat -f%z "$output_path" 2>/dev/null || stat -c%s "$output_path" 2>/dev/null)
        if [[ -n "$file_size" && "$file_size" -gt 2000000000 ]]; then
            echo "✅ File size verification passed"
        else
            echo "⚠️  Downloaded file seems smaller than expected - may be incomplete"
        fi
    else
        echo "❌ Failed to download Phi-3.5 Mini model"
        echo "💡 You can manually download from: $url"
        rm -f "$output_path"  # Clean up partial download
    fi
}

# Function to download Qwen2.5 0.5B from Hugging Face
download_qwen_mini() {
    local model_name="Qwen2.5-0.5B-Instruct"
    local filename="qwen2.5-0.5b-instruct-q4_k_m.gguf"
    local url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/${filename}"
    local output_path="models/${filename}"
    
    echo "🔄 Downloading Qwen2.5 0.5B Instruct..."
    echo "   Source: ${url}"
    echo "   Size: ~394MB"
    echo "   This may take a few minutes..."
    
    # Check if file already exists
    if [[ -f "$output_path" ]]; then
        echo "✅ Qwen2.5 0.5B model already exists: ${filename}"
        return 0
    fi
    
    if curl -L --progress-bar --fail -o "$output_path" "$url"; then
        echo "✅ Qwen2.5 0.5B model downloaded: ${filename}"
        echo "🧠 Ultra-fast model ready for quick responses"
        
        # Verify file size (should be around 394MB)
        local file_size=$(stat -f%z "$output_path" 2>/dev/null || stat -c%s "$output_path" 2>/dev/null)
        if [[ -n "$file_size" && "$file_size" -gt 350000000 ]]; then
            echo "✅ File size verification passed"
        else
            echo "⚠️  Downloaded file seems smaller than expected - may be incomplete"
        fi
    else
        echo "❌ Failed to download Qwen2.5 model"
        echo "💡 You can manually download from: $url"
        rm -f "$output_path"  # Clean up partial download
    fi
}

# Main execution
echo "🚀 Starting AI Models Setup"
echo ""

# Check existing models
existing_models=$(find models/ -name "*.gguf" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$existing_models" -gt 0 ]]; then
    echo "📋 Found $existing_models existing model(s):"
    find models/ -name "*.gguf" -exec basename {} \; 2>/dev/null | sed 's/^/   - /'
    echo ""
fi

read -p "📥 Download sample AI models from Hugging Face? (y/N): " download_models
download_models=${download_models:-N}

if [[ ! "$download_models" =~ ^[Yy]$ ]]; then
    echo "⏭️  Skipping model download"
    echo "💡 You can add your own GGUF models to the ./models/ directory"
    echo "💡 Visit https://huggingface.co/models?library=ggml for more models"
    exit 0
fi

echo ""
echo "📡 Available models from Hugging Face:"
echo "1) TinyLlama 1.1B Chat (Fast, good for testing) - ~637MB"
echo "2) Phi-3.5 Mini Instruct (High quality, compact) - ~2.2GB"
echo "3) Qwen2.5 0.5B Instruct (Very fast, tiny) - ~394MB"
echo "4) Download all models"
echo "5) Skip model download"

while true; do
    read -p "Select a model to download (1-5): " model_choice
    case $model_choice in
        1)
            download_tinyllama
            break
            ;;
        2)
            download_phi3_mini
            break
            ;;
        3)
            download_qwen_mini
            break
            ;;
        4)
            echo "📥 Downloading all models..."
            download_tinyllama
            download_phi3_mini
            download_qwen_mini
            break
            ;;
        5)
            echo "⏭️  Skipping model download"
            break
            ;;
        *)
            echo "❌ Invalid choice. Please enter 1-5."
            ;;
    esac
done

echo ""
echo "✅ Models setup complete!"
echo ""

# Check final status
final_model_count=$(find models/ -name "*.gguf" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$final_model_count" -gt 0 ]]; then
    echo "🧠 AI Models: $final_model_count model(s) available"
    echo "📋 Available models:"
    find models/ -name "*.gguf" -exec basename {} \; 2>/dev/null | sed 's/^/   - /'
    echo ""
    echo "🎯 Example usage:"
    first_model=$(find models/ -name "*.gguf" -exec basename {} \; 2>/dev/null | head -1)
    if [[ -n "$first_model" ]]; then
        echo "   python app/main.py --model-path ./models/$first_model"
    fi
else
    echo "🧠 AI Models: No models installed"
    echo "💡 You can run this script again to download models"
    echo "💡 Or add your own GGUF models to the ./models/ directory"
fi

echo ""
echo "🌟 Models ready for AI applications!"
