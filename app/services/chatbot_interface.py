"""
Chatbot Interface definition for improved testability and extensibility.

This module defines clear contracts for chatbot operations following
the Extract Interface refactoring pattern.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator
from .value_objects import SearchResults, ModelInfo, KnowledgeStats, ErrorStats
from .model_configuration import ModelConfiguration, SamplingConfiguration


class IChatbot(ABC):
    """
    Interface defining the contract for chatbot operations.
    
    This interface provides a clear contract for chatbot functionality,
    improving testability and enabling different implementations.
    """
    
    # Core messaging operations
    @abstractmethod
    def send_message(self, message: str) -> str:
        """Send a message and get a complete response."""
        pass
    
    @abstractmethod
    def send_message_stream(self, message: str) -> Generator[str, None, None]:
        """Send a message and stream the response token by token."""
        pass
    
    # Model management
    @abstractmethod
    def load_model(self, model_path: str, config: Optional[ModelConfiguration] = None, options: str = "") -> bool:
        """Load a GGUF model for inference."""
        pass
    
    @abstractmethod
    def load_model_with_preset(self, model_path: str, preset: str = "balanced") -> bool:
        """Load a model with a predefined configuration preset."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if the chatbot is ready for inference."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Get information about the loaded model."""
        pass
    
    # Knowledge management
    @abstractmethod
    def add_knowledge(self, content: str, metadata: Dict[str, Any] = None) -> List[int]:
        """Add content to the knowledge base for RAG."""
        pass
    
    @abstractmethod
    def search_knowledge(self, query: str, limit: int = 3) -> SearchResults:
        """Search the knowledge base using the configured search strategy."""
        pass
    
    @abstractmethod
    def clear_knowledge(self) -> bool:
        """Clear all knowledge from the vector store."""
        pass
    
    @abstractmethod
    def get_knowledge_stats(self) -> KnowledgeStats:
        """Get statistics about the knowledge base."""
        pass
    
    # File management
    @abstractmethod
    def upload_file(self, file_content: bytes, filename: str, file_type: str) -> Optional[int]:
        """Upload a file and add its content to the knowledge base."""
        pass
    
    @abstractmethod
    def get_uploaded_files(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of uploaded files."""
        pass
    
    @abstractmethod
    def delete_uploaded_file(self, file_id: int) -> bool:
        """Delete an uploaded file."""
        pass
    
    # Chat session management
    @abstractmethod
    def start_chat(self, title: str = None) -> str:
        """Start a new chat session."""
        pass
    
    @abstractmethod
    def end_chat(self) -> None:
        """End the current chat session."""
        pass
    
    @abstractmethod
    def save_chat(self, messages: List[Dict[str, str]], title: str = None) -> Optional[str]:
        """Save current chat messages to database."""
        pass
    
    @abstractmethod
    def load_chat(self, session_id: str) -> List[Dict[str, str]]:
        """Load chat messages from database."""
        pass
    
    @abstractmethod
    def get_chat_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of saved chat sessions."""
        pass
    
    @abstractmethod
    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session."""
        pass
    
    # Configuration and customization
    @abstractmethod
    def configure_sampling(self, config: Optional[SamplingConfiguration] = None, 
                          temperature: float = None, top_p: float = None, top_k: Optional[int] = None) -> None:
        """Configure sampling parameters for text generation."""
        pass
    
    @abstractmethod
    def configure_sampling_preset(self, preset: str = "balanced") -> None:
        """Configure sampling with a predefined preset."""
        pass
    
    @abstractmethod
    def set_search_strategy(self, strategy_name: str) -> bool:
        """Change the search strategy at runtime."""
        pass
    
    @abstractmethod
    def get_search_strategy_info(self) -> Dict[str, Any]:
        """Get information about the current search strategy."""
        pass
    
    # Error handling and monitoring
    @abstractmethod
    def get_error_recovery_stats(self) -> ErrorStats:
        """Get error recovery statistics."""
        pass
    
    @abstractmethod
    def set_error_recovery_mode(self, mode: str = "robust") -> bool:
        """Change the error recovery mode."""
        pass
    
    # Utility operations
    @abstractmethod
    def generate_embedding(self, text: str) -> bytes:
        """Generate embedding for the given text."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get the sqlite-ai extension version."""
        pass
    
    @abstractmethod
    def reset_conversation(self) -> None:
        """Reset the conversation state completely."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass


class IModelManager(ABC):
    """Interface for model management operations."""
    
    @abstractmethod
    def load_model(self, model_path: str, config: Optional[ModelConfiguration] = None, options: str = "") -> bool:
        """Load a GGUF model for inference."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the current model."""
        pass
    
    @abstractmethod
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        pass
    
    @abstractmethod
    def get_model_path(self) -> Optional[str]:
        """Get the path of the currently loaded model."""
        pass
    
    @abstractmethod
    def get_model_config(self) -> Optional[ModelConfiguration]:
        """Get the configuration of the currently loaded model."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        pass
    
    @abstractmethod
    def validate_model_loaded(self) -> None:
        """Validate that a model is loaded, raise exception if not."""
        pass


class ISearchStrategy(ABC):
    """Interface for search strategies."""
    
    @abstractmethod
    def search(self, query: str, vector_service, query_embedding: Optional[bytes] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Execute the search strategy."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this search strategy."""
        pass
    
    @abstractmethod
    def requires_embedding(self) -> bool:
        """Check if this strategy requires query embeddings."""
        pass


class IErrorRecoveryHandler(ABC):
    """Interface for error recovery handlers."""
    
    @abstractmethod
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Handle an error using available recovery strategies."""
        pass
    
    @abstractmethod
    def execute_with_recovery(self, func, *args, operation_type: str = "unknown", **kwargs) -> Any:
        """Execute a function with automatic error recovery."""
        pass
    
    @abstractmethod
    def get_error_stats(self) -> Dict[str, Any]:
        """Get statistics about handled errors."""
        pass
    
    @abstractmethod
    def clear_error_log(self) -> None:
        """Clear the error log."""
        pass


class IMessageHandler(ABC):
    """Interface for message handlers."""
    
    @abstractmethod
    def process_message(self, message: str) -> str:
        """Process a message and return a response."""
        pass
    
    @abstractmethod
    def get_handler_type(self) -> str:
        """Get the type name of this handler."""
        pass
    
    @abstractmethod
    def increment_message_count(self) -> None:
        """Increment the message counter."""
        pass
    
    @abstractmethod
    def reset_message_count(self) -> None:
        """Reset the message counter."""
        pass


# Type aliases for better readability
ChatbotInterface = IChatbot
ModelManagerInterface = IModelManager
SearchStrategyInterface = ISearchStrategy
ErrorRecoveryInterface = IErrorRecoveryHandler
MessageHandlerInterface = IMessageHandler
