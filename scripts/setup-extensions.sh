#!/bin/bash
# SQLite Extensions Setup Script
# Downloads and installs SQLite-AI and SQLite-Vector extensions

echo "🔌 SQLite Extensions Setup"
echo "=========================="

# Check if curl is available
if ! command -v curl &> /dev/null; then
    echo "❌ curl is required but not installed"
    exit 1
fi

# Check if unzip is available
if ! command -v unzip &> /dev/null; then
    echo "❌ unzip is required but not installed"
    exit 1
fi

# Function to detect platform for SQLite-AI
detect_platform() {
    local os=$(uname -s)
    local arch=$(uname -m)
    
    case "$os" in
        "Darwin")
            echo "macos"  # macOS releases are universal
            ;;
        "Linux")
            case "$arch" in
                "x86_64") echo "linux-x86_64" ;;
                "aarch64"|"arm64") echo "linux-arm64" ;;
                *) echo "unknown" ;;
            esac
            ;;
        "MINGW"*|"MSYS"*|"CYGWIN"*)
            echo "windows-x86_64"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Function to detect platform for sqlite-vector (different naming convention)
detect_vector_platform() {
    local os=$(uname -s)
    local arch=$(uname -m)
    
    case "$os" in
        "Darwin")
            echo "macos"  # macOS releases are universal
            ;;
        "Linux")
            case "$arch" in
                "x86_64") echo "linux-x86_64" ;;
                "aarch64"|"arm64") echo "linux-arm64" ;;
                *) echo "unknown" ;;
            esac
            ;;
        "MINGW"*|"MSYS"*|"CYGWIN"*)
            echo "windows-x86_64"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Function to get file extension for platform
get_extension() {
    local platform=$1
    case "$platform" in
        macos*) echo "dylib" ;;
        linux-*) echo "so" ;;
        windows-*) echo "dll" ;;
        *) echo "so" ;;
    esac
}

# Function to download SQLite-AI extension
download_extension() {
    echo ""
    echo "📦 SQLite-AI Extension Setup"
    echo "============================="
    
    # Create extensions directory
    mkdir -p extensions/
    
    # Check if extension already exists
    if [[ -f "extensions/ai.so" || -f "extensions/ai.dylib" || -f "extensions/ai.dll" ]]; then
        echo "✅ SQLite-AI extension already exists"
        return 0
    fi
    
    local detected_platform=$(detect_platform)
    local platform=""
    
    if [[ "$detected_platform" == "unknown" ]]; then
        echo "❓ Unable to auto-detect your platform."
        echo "Please select your platform:"
        echo "1) macOS (Universal)"
        echo "2) Linux x86_64"
        echo "3) Linux ARM64"
        echo "4) Windows x86_64"
        echo "5) Skip extension download"
        
        while true; do
            read -p "Enter your choice (1-5): " choice
            case $choice in
                1) platform="macos"; break ;;
                2) platform="linux-x86_64"; break ;;
                3) platform="linux-arm64"; break ;;
                4) platform="windows-x86_64"; break ;;
                5) echo "⏭️  Skipping extension download"; return 0 ;;
                *) echo "❌ Invalid choice. Please enter 1-5." ;;
            esac
        done
    else
        platform="$detected_platform"
        echo "🔍 Detected platform: $platform"
        read -p "📥 Download SQLite-AI extension for this platform? (Y/n): " confirm
        confirm=${confirm:-Y}
        
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "⏭️  Skipping extension download"
            return 0
        fi
    fi
    
    local extension=$(get_extension "$platform")
    
    echo "📡 Getting latest release information..."
    
    # Get the latest release info from GitHub API
    local api_url="https://api.github.com/repos/sqliteai/sqlite-ai/releases/latest"
    local release_info
    
    if ! release_info=$(curl -s "$api_url"); then
        echo "❌ Failed to fetch release information"
        echo "💡 You can manually download from: https://github.com/sqliteai/sqlite-ai/releases"
        return 1
    fi
    
    # Extract the version from the release info
    local version
    version=$(echo "$release_info" | grep -o '"tag_name": "[^"]*"' | sed 's/"tag_name": "//; s/"//')
    
    if [[ -z "$version" ]]; then
        echo "❌ Could not extract version information"
        return 1
    fi
    
    echo "📦 Latest version: $version"
    
    # Construct download URL for the zip file
    local download_url="https://github.com/sqliteai/sqlite-ai/releases/download/${version}/ai-${platform}-${version}.zip"
    local zip_filename="ai-${platform}-${version}.zip"
    
    echo "📥 Downloading SQLite-AI extension..."
    echo "   URL: $download_url"
    
    if curl -L -o "$zip_filename" "$download_url"; then
        echo "✅ Downloaded: $zip_filename"
        
        # Extract the zip file
        echo "📂 Extracting extension..."
        if unzip -o -q "$zip_filename"; then
            echo "✅ Extracted successfully"
            
            # Find the extension file and rename it to the expected name
            local found_extension=""
            local target_filename="extensions/ai.$extension"
            
            # Look for the extension file in common locations
            if [[ -f "ai.$extension" ]]; then
                found_extension="ai.$extension"
                echo "🔍 Found: ai.$extension"
            elif [[ -f "sqlite-ai.$extension" ]]; then
                found_extension="sqlite-ai.$extension"
                echo "🔍 Found: sqlite-ai.$extension"
            elif [[ -f "libsqlite-ai.$extension" ]]; then
                found_extension="libsqlite-ai.$extension"
                echo "🔍 Found: libsqlite-ai.$extension"
            else
                # Search recursively for the extension file (be more specific to avoid false matches)
                found_extension=$(find . -name "ai.$extension" -o -name "*ai*.$extension" | grep -v "/venv/" | head -1)
                if [[ -n "$found_extension" ]]; then
                    echo "🔍 Found: $found_extension"
                fi
            fi
            
            if [[ -n "$found_extension" && -f "$found_extension" ]]; then
                # Move/rename to expected location in extensions directory
                mv "$found_extension" "$target_filename"
                
                # Set executable permissions
                chmod +x "$target_filename"
                
                echo "✅ Extension ready: $target_filename"
                echo "🔍 File permissions: $(ls -la "$target_filename")"
                
                # Clean up
                rm -f "$zip_filename"
                rm -rf ai-${platform}-${version} 2>/dev/null || true
                
                return 0
            else
                echo "❌ Could not find extension file in archive"
                echo "💡 Contents of archive:"
                unzip -l "$zip_filename" | head -20
                rm -f "$zip_filename"
                return 1
            fi
        else
            echo "❌ Failed to extract archive"
            rm -f "$zip_filename"
            return 1
        fi
    else
        echo "❌ Failed to download extension"
        echo "💡 You can manually download from: $download_url"
        return 1
    fi
}

# Function to download sqlite-vector extension
download_vector_extension() {
    echo ""
    echo "📦 SQLite-Vector Extension Setup"
    echo "================================"
    
    # Check if extension already exists
    if [[ -f "extensions/vector.so" || -f "extensions/vector.dylib" || -f "extensions/vector.dll" ]]; then
        echo "✅ SQLite-Vector extension already exists"
        return 0
    fi
    
    local detected_platform=$(detect_vector_platform)
    local platform=""
    
    if [[ "$detected_platform" == "unknown" ]]; then
        echo "❓ Unable to auto-detect your platform."
        echo "Please select your platform:"
        echo "1) macOS (Universal)"
        echo "2) Linux x86_64"
        echo "3) Linux ARM64"
        echo "4) Windows x86_64"
        echo "5) Skip vector extension download"
        
        while true; do
            read -p "Enter your choice (1-5): " choice
            case $choice in
                1) platform="macos"; break ;;
                2) platform="linux-x86_64"; break ;;
                3) platform="linux-arm64"; break ;;
                4) platform="windows-x86_64"; break ;;
                5) echo "⏭️  Skipping vector extension download"; return 0 ;;
                *) echo "❌ Invalid choice. Please enter 1-5." ;;
            esac
        done
    else
        platform="$detected_platform"
        echo "🔍 Detected platform: $platform"
        read -p "📥 Download SQLite-Vector extension for this platform? (Y/n): " confirm
        confirm=${confirm:-Y}
        
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "⏭️  Skipping vector extension download"
            return 0
        fi
    fi
    
    local extension=$(get_extension "$platform")
    
    echo "📡 Getting latest release information..."
    
    # Get the latest release info from GitHub API
    local api_url="https://api.github.com/repos/sqliteai/sqlite-vector/releases/latest"
    local release_info
    
    if ! release_info=$(curl -s "$api_url"); then
        echo "❌ Failed to fetch release information"
        echo "💡 You can manually download from: https://github.com/sqliteai/sqlite-vector/releases"
        return 1
    fi
    
    # Extract the version from the release info
    local version
    version=$(echo "$release_info" | grep -o '"tag_name": "[^"]*"' | sed 's/"tag_name": "//; s/"//')
    
    if [[ -z "$version" ]]; then
        echo "❌ Could not extract version information"
        return 1
    fi
    
    echo "📦 Latest version: $version"
    
    # Construct download URL for the zip file
    local download_url="https://github.com/sqliteai/sqlite-vector/releases/download/${version}/vector-${platform}-${version}.zip"
    local zip_filename="vector-${platform}-${version}.zip"
    
    echo "📥 Downloading SQLite-Vector extension..."
    echo "   URL: $download_url"
    
    if curl -L -o "$zip_filename" "$download_url"; then
        echo "✅ Downloaded: $zip_filename"
        
        # Extract the zip file
        echo "📂 Extracting extension..."
        if unzip -o -q "$zip_filename"; then
            echo "✅ Extracted successfully"
            
            # Find the extension file and rename it to the expected name
            local found_extension=""
            local target_filename="extensions/vector.$extension"
            
            # Look for the extension file in common locations
            if [[ -f "vector.$extension" ]]; then
                found_extension="vector.$extension"
                echo "🔍 Found: vector.$extension"
            elif [[ -f "libvector.$extension" ]]; then
                found_extension="libvector.$extension"
                echo "🔍 Found: libvector.$extension"
            else
                # Search recursively for the extension file (be more specific to avoid false matches)
                found_extension=$(find . -name "vector.$extension" -o -name "*vector*.$extension" | grep -v "/venv/" | head -1)
                if [[ -n "$found_extension" ]]; then
                    echo "🔍 Found: $found_extension"
                fi
            fi
            
            if [[ -n "$found_extension" && -f "$found_extension" ]]; then
                # Move/rename to expected location in extensions directory
                mv "$found_extension" "$target_filename"
                
                # Set executable permissions
                chmod +x "$target_filename"
                
                echo "✅ Extension ready: $target_filename"
                echo "🔍 File permissions: $(ls -la "$target_filename")"
                
                # Clean up
                rm -f "$zip_filename"
                rm -rf vector-${platform}-${version} 2>/dev/null || true
                
                return 0
            else
                echo "❌ Could not find extension file in archive"
                echo "💡 Contents of archive:"
                unzip -l "$zip_filename" | head -20
                rm -f "$zip_filename"
                return 1
            fi
        else
            echo "❌ Failed to extract archive"
            rm -f "$zip_filename"
            return 1
        fi
    else
        echo "❌ Failed to download extension"
        echo "💡 You can manually download from: $download_url"
        return 1
    fi
}

# Main execution
echo "🚀 Starting SQLite Extensions Setup"

# Download SQLite-AI extension
download_extension

# Download SQLite-Vector extension
download_vector_extension

echo ""
echo "✅ Extensions setup complete!"
echo ""

# Check status
ai_extension_status=""
if [[ -f "extensions/ai.so" || -f "extensions/ai.dylib" || -f "extensions/ai.dll" ]]; then
    ai_extension_status="🤖 SQLite-AI extension: Ready for AI magic!"
else
    ai_extension_status="📱 SQLite-AI extension: Not installed"
fi

vector_extension_status=""
if [[ -f "extensions/vector.so" || -f "extensions/vector.dylib" || -f "extensions/vector.dll" ]]; then
    vector_extension_status="🔍 SQLite-Vector extension: Ready for vector search!"
else
    vector_extension_status="📱 SQLite-Vector extension: Not installed"
fi

echo "$ai_extension_status"
echo "$vector_extension_status"

echo ""
if [[ -f "extensions/ai.so" || -f "extensions/ai.dylib" || -f "extensions/ai.dll" ]] && [[ -f "extensions/vector.so" || -f "extensions/vector.dylib" || -f "extensions/vector.dll" ]]; then
    echo "🌟 All extensions ready for intelligent database applications!"
elif [[ -f "extensions/ai.so" || -f "extensions/ai.dylib" || -f "extensions/ai.dll" ]]; then
    echo "🌟 AI extension ready! Run again to add vector search capabilities."
else
    echo "💡 Run this script again to download extensions."
fi
