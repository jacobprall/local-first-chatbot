import os
from typing import Dict, Any, Generator, Optional, List
from .database import DatabaseManager
from .ai import AIService
from .vector import VectorService
from .chat_history import ChatHistoryManager
from .model_manager import model_manager
from .model_configuration import ModelConfiguration, ModelConfigurations, SamplingConfiguration, SamplingConfigurations
from .search_strategy import SearchContext, SearchStrategies
from .error_recovery import ErrorRecoveryHandler, ErrorRecoveryConfigurations
from .message_handlers import MessageHandler, MessageHandlerFactory
from .value_objects import SearchResult, SearchResults, ModelInfo, KnowledgeStats, FileInfo, ChatSession, ErrorStats
from .chatbot_interface import IChatbot


# Configuration Constants
class ModelConstants:
    """Model configuration constants for consistent behavior across the application."""
    # Text generation limits
    DEFAULT_PREDICTION_TOKENS = 16    # Standard response length
    RECOVERY_PREDICTION_TOKENS = 8    # Shorter responses for error recovery
    DEFAULT_CONTEXT_SIZE = 128        # Model context window size
    DEFAULT_GPU_LAYERS = 0            # CPU-only by default for compatibility

    # Content processing limits
    MAX_CONTEXT_SNIPPET_LENGTH = 100  # Max chars per context document
    MAX_LOG_MESSAGE_LENGTH = 50       # Truncation for log messages
    MAX_CONTEXT_PREVIEW_LENGTH = 30   # Preview length in logs
    DEFAULT_SEARCH_LIMIT = 2          # Default number of context documents


class PromptTemplates:
    """Standardized prompt formats for consistent model interaction."""
    SIMPLE_CHAT = "Human: {message}\nAssistant:"
    RAG_CHAT = "Context: {context}\nHuman: {message}\nAssistant:"
    RECOVERY_CHAT = "Q: {message}\nA:"  # Simplified format for error recovery


class Chatbot(IChatbot):
    """
    RAG-enabled chatbot that provides intelligent responses using retrieval-augmented generation.

    This chatbot combines language model capabilities with knowledge base search to provide
    contextually relevant responses. All interactions use RAG for enhanced accuracy.
    """

    def __init__(self, db_path: str, model_path: Optional[str] = None):
        """
        Initialize the chatbot with database and optional model.

        Args:
            db_path: Path to SQLite database for persistence
            model_path: Optional path to GGUF model file for immediate loading
        """
        # Core services initialization
        self.db_manager = DatabaseManager(db_path)
        self.ai = AIService(self.db_manager)
        self.vector = VectorService(self.db_manager)
        self.chat_history = ChatHistoryManager(self.db_manager)

        # Session state
        self.current_chat_id: Optional[str] = None
        self.message_count = 0

        # Configure search strategy with fallback for robustness
        self.search_context = SearchContext(SearchStrategies.robust_fallback())

        # Configure error recovery for resilient operation
        self.error_recovery = ErrorRecoveryConfigurations.robust()

        # Initialize message handler for RAG processing
        self.message_handler = MessageHandlerFactory.create_handler(
            "rag", self.ai, self.vector, self.search_context, self.error_recovery
        )

        # Set up model manager and load model if provided
        model_manager.initialize_ai_service(self.ai)
        if model_path and os.path.exists(model_path) and not model_manager.is_model_loaded():
            self.load_model(model_path)
    
    def _reset_inference_state(self) -> None:
        """
        Reset the inference state to prevent corruption between messages.

        This is necessary because sqlite-ai maintains internal state that can
        become corrupted after multiple inference calls, leading to assertion
        errors or degraded performance.
        """
        try:
            # Clear and recreate the inference context with fresh parameters
            with self.db_manager.get_connection() as conn:
                conn.execute("SELECT llm_context_free()")
                conn.execute(f"SELECT llm_context_create('n_ctx={ModelConstants.DEFAULT_CONTEXT_SIZE}')")
                conn.commit()

            # Reset sampler to ensure consistent behavior
            with self.db_manager.get_connection() as conn:
                conn.execute("SELECT llm_sampler_free()")
                conn.commit()
            self.ai.configure_sampler_greedy()

        except Exception as e:
            print(f"Failed to reset inference state: {e}")
    
    def load_model(self, model_path: str, config: Optional[ModelConfiguration] = None, options: str = "") -> bool:
        """
        Load a GGUF model for inference.

        Args:
            model_path: Path to the GGUF model file
            config: Model configuration object with parameters
            options: Legacy options string (use config parameter instead)

        Returns:
            bool: True if model loaded successfully
        """
        if config is None and not options:
            config = ModelConfigurations.balanced()

        return model_manager.load_model(model_path, config, options)

    def load_model_with_preset(self, model_path: str, preset: str = "balanced") -> bool:
        """
        Load a model with a predefined configuration preset.

        Args:
            model_path: Path to the GGUF model file
            preset: Configuration preset name ('fast', 'balanced', 'high_quality', 'creative', 'gpu')

        Returns:
            bool: True if model loaded successfully
        """
        preset_configs = {
            "fast": ModelConfigurations.fast_inference(),
            "balanced": ModelConfigurations.balanced(),
            "high_quality": ModelConfigurations.high_quality(),
            "creative": ModelConfigurations.creative(),
            "gpu": ModelConfigurations.gpu_accelerated()
        }

        config = preset_configs.get(preset, ModelConfigurations.balanced())
        return self.load_model(model_path, config)
    
    def send_message(self, message: str) -> str:
        """
        Send a message and get a RAG-enhanced response.

        The response is generated using retrieval-augmented generation,
        combining relevant knowledge base context with the language model.

        Args:
            message: User's input message

        Returns:
            str: Generated response with contextual knowledge
        """
        response = self.message_handler.process_message(message)
        self.message_count = self.message_handler.message_count
        return response

    def get_current_handler_type(self) -> str:
        """Get the type of the current message handler (always 'rag')."""
        return self.message_handler.get_handler_type()

    def reset_message_counters(self) -> None:
        """Reset message counters for conversation tracking."""
        self.message_count = 0
        self.message_handler.reset_message_count()

    def add_knowledge(self, content: str, metadata: Dict[str, Any] = None) -> List[int]:
        """Add content to the knowledge base for RAG"""
        try:
            # Generate embeddings for the content if model is loaded
            if model_manager.is_model_loaded():
                try:
                    # Generate embedding for the full content first
                    embedding = self.ai.generate_embedding(content)
                    # Add document with chunks, passing the embedding
                    chunk_ids = self.vector.add_document_with_chunks(content, metadata)
                    
                    # Generate embeddings for each chunk individually for better RAG performance
                    for chunk_id in chunk_ids:
                        doc = self.vector.get_document(chunk_id)
                        if doc and doc["content"]:
                            try:
                                chunk_embedding = self.ai.generate_embedding(doc["content"])
                                self.vector.add_embedding_to_document(chunk_id, chunk_embedding)
                            except Exception as e:
                                print(f"Failed to generate embedding for chunk {chunk_id}: {e}")
                    
                    return chunk_ids
                except Exception as e:
                    print(f"Failed to generate embeddings: {e}")
                    # Fallback to adding without embeddings
                    return self.vector.add_document_with_chunks(content, metadata)
            else:
                # No model loaded, add without embeddings
                return self.vector.add_document_with_chunks(content, metadata)
            
        except Exception as e:
            print(f"Failed to add knowledge: {e}")
            return []
    
    def search_knowledge(self, query: str, limit: int = 3) -> SearchResults:
        """Search the knowledge base using the configured search strategy."""
        import time
        start_time = time.time()

        print(f"\n🔎 KNOWLEDGE SEARCH: '{query}' (limit={limit})")
        print(f"   Model loaded: {model_manager.is_model_loaded()}")
        print(f"   Search strategy: {self.search_context.get_current_strategy_name()}")

        try:
            query_embedding = None

            # Generate embedding for semantic search if needed
            # Some strategies (semantic, hybrid) require embeddings for vector similarity
            if self.search_context.requires_embedding() and model_manager.is_model_loaded():
                print(f"   🧠 Generating embedding for query...")
                try:
                    query_embedding = self.ai.generate_embedding(query)
                    embedding_size = len(query_embedding) if query_embedding else 0
                    print(f"   ✅ Generated {embedding_size} byte embedding")
                except Exception as e:
                    print(f"   ❌ Failed to generate embedding: {e}")
            elif self.search_context.requires_embedding():
                print(f"   ⚠️  Strategy requires embedding but no model loaded")
            else:
                print(f"   ℹ️  Strategy doesn't require embedding")

            # Execute search using current strategy
            raw_results = self.search_context.search(query, self.vector, query_embedding, limit)

            # Convert to value objects
            search_results = [SearchResult.from_dict(result) for result in raw_results]
            search_time_ms = (time.time() - start_time) * 1000

            results = SearchResults(
                results=search_results,
                query=query,
                total_found=len(search_results),
                search_time_ms=search_time_ms
            )

            if not results.is_empty():
                print(f"   ✅ Search returned {len(results)} results in {search_time_ms:.1f}ms")
            else:
                print(f"   ⚠️  Search returned no results")

            return results

        except Exception as e:
            print(f"   ❌ Knowledge search completely failed: {e}")
            return SearchResults(query=query, search_time_ms=(time.time() - start_time) * 1000)

    def search_knowledge_legacy(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Legacy method returning dictionaries for backward compatibility."""
        results = self.search_knowledge(query, limit)
        return results.to_dict_list()

    def set_search_strategy(self, strategy_name: str) -> bool:
        """
        Change the search strategy at runtime.

        Args:
            strategy_name: Name of the strategy ('hybrid', 'semantic', 'text', 'robust_fallback')

        Returns:
            bool: True if strategy was set successfully
        """
        try:
            strategy_map = {
                "hybrid": SearchStrategies.hybrid(),
                "semantic": SearchStrategies.semantic(),
                "text": SearchStrategies.text(),
                "robust_fallback": SearchStrategies.robust_fallback(),
                "embedding_fallback": SearchStrategies.embedding_fallback(),
                "text_only_fallback": SearchStrategies.text_only_fallback()
            }

            if strategy_name not in strategy_map:
                print(f"Unknown search strategy: {strategy_name}")
                return False

            self.search_context.set_strategy(strategy_map[strategy_name])
            print(f"Search strategy changed to: {strategy_name}")
            return True

        except Exception as e:
            print(f"Failed to set search strategy: {e}")
            return False

    def get_search_strategy_info(self) -> Dict[str, Any]:
        """Get information about the current search strategy."""
        return {
            "current_strategy": self.search_context.get_current_strategy_name(),
            "requires_embedding": self.search_context.requires_embedding(),
            "model_loaded": model_manager.is_model_loaded(),
            "available_strategies": ["hybrid", "semantic", "text", "robust_fallback", "embedding_fallback", "text_only_fallback"]
        }
    
    def optimize_vectors(self) -> bool:
        """Optimize vectors for better RAG performance."""
        if not self.vector.vector_enabled:
            print("Vector extension not available")
            return False

        try:
            # Only try to preload if we have documents with embeddings
            if self.vector.get_vector_stats().get("documents_with_embeddings", 0) > 0:
                self.vector.preload_quantized_vectors()
                print("Vector optimization complete")
                return True
            else:
                print("No documents with embeddings found, skipping vector optimization")
                return False
        except Exception as e:
            print(f"Failed to optimize vectors: {e}")
            return False
    
    def get_knowledge_stats(self) -> KnowledgeStats:
        """Get statistics about the knowledge base as a value object."""
        vector_stats = self.vector.get_vector_stats()
        return KnowledgeStats(
            document_count=self.vector.get_document_count(),
            chunk_count=self.vector.get_chunk_count(),
            rag_enabled=True,  # Always enabled
            vector_extension=vector_stats["vector_enabled"],
            vector_initialized=vector_stats.get("vector_initialized", False),
            documents_with_embeddings=vector_stats.get("documents_with_embeddings", 0)
        )

    def get_knowledge_stats_legacy(self) -> Dict[str, Any]:
        """Legacy method returning dictionary for backward compatibility."""
        return self.get_knowledge_stats().to_dict()
    
    def clear_knowledge(self) -> bool:
        """Clear all knowledge from the vector store"""
        return self.vector.clear_all_documents()
    
    def upload_file(self, file_content: bytes, filename: str, file_type: str) -> Optional[int]:
        """Upload a file and add its content to the knowledge base"""
        return self.vector.upload_file(file_content, filename, file_type)
    
    def get_uploaded_files(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of uploaded files"""
        return self.vector.get_uploaded_files(limit)
    
    def delete_uploaded_file(self, file_id: int) -> bool:
        """Delete an uploaded file"""
        return self.vector.delete_uploaded_file(file_id)
    
    def get_file_stats(self) -> Dict[str, Any]:
        """Get file upload statistics"""
        try:
            return self.vector.get_file_stats()
        except Exception as e:
            print(f"Error getting file stats: {e}")
            return {"total_files": 0, "total_size": 0, "file_types": {}}
    
    def fix_vector_search(self) -> bool:
        """Fix vector search issues by cleaning up corrupted embeddings"""
        try:
            print("🔧 Attempting to fix vector search issues...")
            
            # Fix corrupted embeddings
            embeddings_fixed = self.vector.fix_vector_embeddings()
            
            # Try to rebuild quantization
            quantization_fixed = self.vector.rebuild_quantization_if_needed()
            
            print(f"   Embeddings fixed: {embeddings_fixed}")
            print(f"   Quantization rebuilt: {quantization_fixed}")
            
            return embeddings_fixed or quantization_fixed
            
        except Exception as e:
            print(f"Failed to fix vector search: {e}")
            return False
    
    # ... (rest of the existing methods remain the same)
    def send_message_stream(self, message: str) -> Generator[str, None, None]:
        """Send a message and stream the response token by token."""
        try:
            response = self.send_message(message)
            words = response.split()
            for word in words:
                yield word + " "
        except Exception as e:
            yield f"Error: {e}"
    
    def start_chat(self, title: str = None) -> str:
        """Start a new chat session."""
        model_manager.validate_model_loaded()
        
        # Create a new chat session in the database
        if not title:
            from datetime import datetime
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        self.current_chat_id = self.chat_history.create_chat_session(
            title=title,
            model_used=model_manager.get_model_path(),
            rag_enabled=True  # Always enabled
        )
        return self.current_chat_id
    
    def end_chat(self) -> None:
        """End the current chat session."""
        self.current_chat_id = None
    
    def get_model_info(self) -> ModelInfo:
        """Get information about the loaded model as a value object."""
        raw_info = model_manager.get_model_info()
        return ModelInfo(
            status=raw_info.get("status", "Unknown"),
            model_path=raw_info.get("model_path"),
            model_config=raw_info.get("model_config"),
            chat_active=self.current_chat_id is not None,
            version=raw_info.get("version")
        )

    def get_model_info_legacy(self) -> Dict[str, Any]:
        """Legacy method returning dictionary for backward compatibility."""
        return self.get_model_info().to_dict()
    
    def configure_sampling(self, config: Optional[SamplingConfiguration] = None,
                          temperature: float = None, top_p: float = None, top_k: Optional[int] = None) -> None:
        """
        Configure sampling parameters for text generation.

        Args:
            config: SamplingConfiguration object (preferred)
            temperature: Legacy parameter (deprecated, use config)
            top_p: Legacy parameter (deprecated, use config)
            top_k: Legacy parameter (deprecated, use config)
        """
        try:
            # Use config object if provided, otherwise create from legacy parameters
            if config is None:
                if temperature is not None or top_p is not None or top_k is not None:
                    # Legacy support
                    config = SamplingConfiguration(
                        temperature=temperature if temperature is not None else 0.7,
                        top_p=top_p if top_p is not None else 0.9,
                        top_k=top_k
                    )
                else:
                    config = SamplingConfigurations.balanced()

            config.validate()

            if config.is_greedy():
                self.ai.configure_sampler_greedy()
            else:
                self.ai.configure_sampler_temperature(config.temperature)
                if config.top_p > 0:
                    self.ai.configure_sampler_top_p(config.top_p, config.min_keep)
                if config.top_k and config.top_k > 0:
                    self.ai.configure_sampler_top_k(config.top_k)

        except Exception as e:
            print(f"Failed to configure sampling: {e}")
            self.ai.configure_sampler_greedy()

    def configure_sampling_preset(self, preset: str = "balanced") -> None:
        """
        Configure sampling with a predefined preset.

        Args:
            preset: Preset name ('greedy', 'balanced', 'creative', 'focused')
        """
        preset_configs = {
            "greedy": SamplingConfigurations.greedy(),
            "balanced": SamplingConfigurations.balanced(),
            "creative": SamplingConfigurations.creative(),
            "focused": SamplingConfigurations.focused()
        }

        config = preset_configs.get(preset, SamplingConfigurations.balanced())
        self.configure_sampling(config)

    def get_error_recovery_stats(self) -> ErrorStats:
        """Get error recovery statistics as a value object."""
        raw_stats = self.error_recovery.get_error_stats()
        return ErrorStats(
            total_errors=raw_stats.get("total_errors", 0),
            successful_recoveries=raw_stats.get("successful_recoveries", 0),
            recovery_rate=raw_stats.get("recovery_rate", 0.0),
            error_types=raw_stats.get("error_types", {}),
            recovery_strategies_used=raw_stats.get("recovery_strategies_used", {})
        )

    def get_error_recovery_stats_legacy(self) -> Dict[str, Any]:
        """Legacy method returning dictionary for backward compatibility."""
        return self.get_error_recovery_stats().to_dict()

    def clear_error_log(self) -> None:
        """Clear the error recovery log."""
        self.error_recovery.clear_error_log()

    def set_error_recovery_mode(self, mode: str = "robust") -> bool:
        """
        Change the error recovery mode.

        Args:
            mode: Recovery mode ('basic', 'robust', 'text_generation_focused')

        Returns:
            bool: True if mode was set successfully
        """
        try:
            mode_configs = {
                "basic": ErrorRecoveryConfigurations.basic(),
                "robust": ErrorRecoveryConfigurations.robust(),
                "text_generation_focused": ErrorRecoveryConfigurations.text_generation_focused()
            }

            if mode not in mode_configs:
                print(f"Unknown error recovery mode: {mode}")
                return False

            self.error_recovery = mode_configs[mode]
            print(f"Error recovery mode changed to: {mode}")
            return True

        except Exception as e:
            print(f"Failed to set error recovery mode: {e}")
            return False
    
    def generate_embedding(self, text: str) -> bytes:
        """Generate embedding for the given text."""
        model_manager.validate_model_loaded()
        return self.ai.generate_embedding(text)
    
    def get_version(self) -> str:
        """Get the sqlite-ai extension version."""
        return self.ai.get_version()
    
    def is_ready(self) -> bool:
        """Check if the chatbot is ready for inference."""
        return model_manager.is_model_loaded()
    
    def reset_conversation(self) -> None:
        """Reset the conversation state completely"""
        try:
            print("Resetting conversation state...")
            self.message_count = 0
            
            if self.current_chat_id:
                self.end_chat()
            
            self._reset_inference_state()
            
            print("Conversation state reset successfully!")
        except Exception as e:
            print(f"Failed to reset conversation state: {e}")
    
    def save_chat(self, messages: List[Dict[str, str]], title: str = None) -> Optional[str]:
        """Save current chat messages to database"""
        try:
            session_id = self.chat_history.save_current_chat(
                messages=messages,
                title=title,
                model_used=model_manager.get_model_path(),
                rag_enabled=True  # Always enabled
            )
            return session_id
        except Exception as e:
            print(f"Failed to save chat: {e}")
            return None
    
    def load_chat(self, session_id: str) -> List[Dict[str, str]]:
        """Load chat messages from database"""
        try:
            messages = self.chat_history.get_chat_messages(session_id)
            # Convert to the format expected by Streamlit
            return [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        except Exception as e:
            print(f"Failed to load chat: {e}")
            return []
    
    def get_chat_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of saved chat sessions"""
        return self.chat_history.get_chat_sessions(limit)
    
    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session"""
        return self.chat_history.delete_chat_session(session_id)
    
    def get_chat_stats(self) -> Dict[str, Any]:
        """Get chat history statistics"""
        return self.chat_history.get_chat_stats()
    
    def add_message_to_current_chat(self, role: str, content: str) -> bool:
        """Add a message to the current chat session"""
        if self.current_chat_id:
            return self.chat_history.add_message(self.current_chat_id, role, content)
        return False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.current_chat_id:
            self.end_chat()