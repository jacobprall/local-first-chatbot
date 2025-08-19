# Local Chatbot with SQLite

A complete local chatbot implementation using:
- **SQLite** as the core database
- **sqlite-ai** for language model integration
- **sqlite-vector** for semantic search and embeddings
- **Streamlit** for an intuitive web interface

Test models, RAG strategies and more from your local machine.

## ✨ Features

- **🔧 One-Line Setup**: Automated installation and configuration
- **🤖 Local AI Chat**: Run language models entirely offline using GGUF format
- **📚 Knowledge Base**: Upload and process documents (text, markdown) with automatic chunking
- **🔍 Semantic Search**: Vector embeddings and hybrid search (semantic + full-text)
- **💾 Persistent Storage**: All data stored locally in SQLite database
- **🎛️ Model Management**: Easy switching between different language models
- **📊 Performance Monitoring**: Real-time statistics and system information

## 🚀 Quick Start

### One-Line Installation

```bash
bash scripts/setup.sh --complete
```

### Manual Setup

1. **Clone and navigate to the project**:
   ```bash
   git clone <repository-url>
   cd local-first-chatbot
   ```

2. **Run the setup script**:
   ```bash
   bash scripts/setup.sh
   ```
   Choose option 1 for complete setup, or customize your installation.

3. **Activate the environment and start the app**:
   ```bash
   source venv/bin/activate
   python -m streamlit run app/main.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📁 Project Structure

```
local-first-chatbot/
├── app/                    # Main application code
│   ├── main.py            # Streamlit UI and main entry point
│   └── services/          # Core business logic
│       ├── ai.py          # AI service (sqlite-ai wrapper)
│       ├── chatbot.py     # Main chatbot orchestration
│       ├── database.py    # Database management and connections
│       └── vector.py      # Vector operations (sqlite-vector wrapper)
├── data/                  # Local data storage
│   ├── chat.db           # SQLite database (created automatically)
│   └── rome-guide.md     # Sample knowledge document
├── extensions/            # SQLite extensions (downloaded by setup)
│   ├── ai.dylib          # sqlite-ai extension
│   └── vector.dylib      # sqlite-vector extension
├── models/               # AI models storage
│   └── *.gguf           # GGUF format language models
├── scripts/             # Setup and utility scripts
│   ├── setup.sh         # Main setup orchestrator
│   ├── setup-environment.sh  # Python environment setup
│   ├── setup-extensions.sh   # SQLite extensions setup
│   └── setup-models.sh       # AI models download
└── requirements.txt     # Python dependencies
```

### Key Files Overview

- **`app/main.py`**: Streamlit interface with three main tabs (Chat, Knowledge, System)
- **`app/services/chatbot.py`**: Core chatbot logic, message handling, and RAG implementation
- **`app/services/ai.py`**: Wrapper for sqlite-ai extension functions (text generation, embeddings)
- **`app/services/vector.py`**: Vector and hybrid (fts/semantic) search operations using sqlite-vector
- **`app/services/database.py`**: Database connection management and schema initialization

## 🧠 Model Configuration

### Supported Models

The application works with GGUF format models. Setup script provides these options:

1. **TinyLlama 1.1B Chat** (~637MB) - Fast, good for testing
2. **Phi-3.5 Mini Instruct** (~2.2GB) - High quality, compact
3. **Qwen2.5 0.5B Instruct** (~394MB) - Very fast, tiny model

### Model Parameters

Key parameters that affect output and performance:

- **`n_predict`**: Maximum tokens to generate (default: 16 for chat, 8 for recovery)
- **Temperature**: Controls randomness (0.1-2.0, higher = more creative)
- **Top-p**: Nucleus sampling threshold (0.1-0.99, controls diversity)
- **Top-k**: Limits vocabulary to top K tokens (typically 40)

### Performance Considerations

- **Model Size**: Larger models provide better quality but require more memory
- **Context Length**: Longer prompts slow generation but provide better context
- **Sampling Strategy**: Greedy sampling (default) is fastest, temperature/top-p add creativity
- **Vector Search**: Embedding generation adds latency but improves relevance

## 🔍 RAG Implementation

### Document Processing

1. **Upload**: Support for text and markdown files
2. **Chunking**: Automatic text splitting for optimal retrieval
3. **Embedding**: Generate vector embeddings using the loaded model
4. **Storage**: Store chunks and embeddings in SQLite

### Search Strategies

- **Semantic Search**: Vector similarity using embeddings
- **Full-Text Search**: SQLite FTS for keyword matching
- **Hybrid Search**: Combines both approaches for best results

### Knowledge Base Management

- View uploaded documents and statistics
- Clear knowledge base
- Optimize vector storage
- Monitor embedding generation

## 🔧 Advanced Configuration

### Database Schema

The application automatically creates these tables:
- `documents`: Store document chunks and metadata
- `chat_sessions`: Persist conversation history
- `files`: Track uploaded files

### Extension Management

SQLite extensions are automatically loaded:
- **sqlite-ai**: Provides `llm_*` functions for AI operations
- **sqlite-vector**: Provides `vector_*` functions for embeddings

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Fails**: Ensure GGUF model is compatible and path is correct
2. **Extensions Not Found**: Run `bash scripts/setup-extensions.sh` to reinstall
3. **Memory Issues**: Try smaller models or reduce context length
4. **Slow Performance**: Check model size and consider hardware limitations

### Debug Mode

Enable verbose logging by setting environment variable:
```bash
STREAMLIT_LOGGER_LEVEL=debug python -m streamlit run app/main.py
```

## 📋 Requirements

- **Python 3.8+**
- **SQLite 3.38+** (for extension support)
- **4GB+ RAM** (recommended for larger models)
- **macOS/Linux** (Windows support via WSL)

## 🚧 Future Work

- **sqlite-sync Integration**: Multi-device synchronization for a truly local-first chatbot.
- **Mutli-modal model support**: Support for image and audio models.

## 🤝 Contributing

This project demonstrates local-first AI principles. Contributions welcome for:
- Additional model support
- Enhanced RAG strategies
- UI/UX improvements
- Performance optimizations

---

**Built with ❤️ using SQLite, sqlite-ai, sqlite-vector, and Streamlit**
