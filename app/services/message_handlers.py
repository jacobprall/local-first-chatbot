"""
Message Handler for RAG-enhanced message processing.

This module provides the core message processing logic using retrieval-augmented
generation to combine language model capabilities with knowledge base search.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from .ai import AIService
from .vector import VectorService
from .search_strategy import SearchContext
from .error_recovery import ErrorRecoveryHandler
from .model_manager import model_manager


class MessageHandler(ABC):
    """Base class for message processing with common functionality."""
    
    def __init__(self, ai_service: AIService, vector_service: VectorService, 
                 search_context: SearchContext, error_recovery: ErrorRecoveryHandler):
        self.ai_service = ai_service
        self.vector_service = vector_service
        self.search_context = search_context
        self.error_recovery = error_recovery
        self.message_count = 0
    
    @abstractmethod
    def process_message(self, message: str) -> str:
        """
        Process a message and return a response.
        
        Args:
            message: The input message to process
            
        Returns:
            str: The generated response
        """
        pass
    
    @abstractmethod
    def get_handler_type(self) -> str:
        """Get the type name of this handler."""
        pass
    
    def increment_message_count(self) -> None:
        """Increment the message counter."""
        self.message_count += 1
    
    def reset_message_count(self) -> None:
        """Reset the message counter."""
        self.message_count = 0


class RAGMessageHandler(MessageHandler):
    """
    Processes messages using Retrieval-Augmented Generation.

    Combines knowledge base search with language model generation to provide
    contextually relevant responses. This is the primary message processing
    strategy for the chatbot.
    """

    def __init__(self, ai_service: AIService, vector_service: VectorService,
                 search_context: SearchContext, error_recovery: ErrorRecoveryHandler,
                 search_limit: int = 2, chatbot_instance=None):
        super().__init__(ai_service, vector_service, search_context, error_recovery)
        self.search_limit = search_limit
        self.chatbot_instance = chatbot_instance
    
    def process_message(self, message: str) -> str:
        """
        Process a message using RAG (Retrieval-Augmented Generation).

        The process involves:
        1. Searching the knowledge base for relevant context
        2. Building a prompt that includes the context
        3. Generating a response using the language model
        4. Applying error recovery if needed

        Args:
            message: User's input message

        Returns:
            str: Generated response with contextual knowledge
        """
        model_manager.validate_model_loaded()
        self.increment_message_count()

        def _process_rag_message():
            print(f"\n🤖 Processing RAG message #{self.message_count}: '{self._truncate_for_log(message)}'")
            self._reset_inference_state()

            # Retrieve relevant context from knowledge base
            context_docs = self._search_for_context(message)
            prompt = self._build_rag_prompt(message, context_docs)

            # Generate response with context
            print(f"   🚀 Generating response...")
            response = self._generate_response(prompt)
            cleaned_response = self._clean_response(response)

            print(f"   ✅ RAG response generated: {len(cleaned_response)} characters")
            return cleaned_response

        # Error recovery context for robust operation
        context = {
            "operation_type": "text_generation",
            "message": message,
            "ai_service": self.ai_service,
            "reset_function": self._reset_inference_state
        }

        try:
            return _process_rag_message()
        except Exception as e:
            print(f"   ❌ RAG message failed: {e}")
            return self.error_recovery.handle_error(e, context)
    
    def _truncate_for_log(self, message: str, max_length: int = 50) -> str:
        """Truncate message for logging purposes."""
        return f"{message[:max_length]}{'...' if len(message) > max_length else ''}"
    
    def _search_for_context(self, message: str) -> List[Dict[str, Any]]:
        """Search for relevant context documents."""
        print(f"   🔍 Searching for relevant context...")
        
        # Generate embedding if required by strategy and model is loaded
        query_embedding = None
        if self.search_context.requires_embedding() and model_manager.is_model_loaded():
            try:
                query_embedding = self.ai_service.generate_embedding(message)
                print(f"   ✅ Generated embedding for search")
            except Exception as e:
                print(f"   ❌ Failed to generate embedding: {e}")
        
        return self.search_context.search(message, self.vector_service, query_embedding, self.search_limit)
    
    def _build_rag_prompt(self, message: str, context_docs: List[Dict[str, Any]]) -> str:
        """Build a RAG prompt with context if available."""
        if context_docs:
            print(f"   📚 Found {len(context_docs)} relevant documents for context")
            context = self._extract_context_from_docs(context_docs)
            prompt = f"Context: {context}\nHuman: {message}\nAssistant:"
            print(f"   📝 Built RAG prompt with {len(context)} chars of context")
            return prompt
        else:
            print(f"   ⚠️  No relevant documents found, using message without context")
            return f"Human: {message}\nAssistant:"
    
    def _extract_context_from_docs(self, context_docs: List[Dict[str, Any]]) -> str:
        """Extract and format context from documents."""
        from .chatbot import ModelConstants  # Import here to avoid circular imports
        context_parts = []
        
        for i, doc in enumerate(context_docs):
            content_snippet = doc['content'][:ModelConstants.MAX_CONTEXT_SNIPPET_LENGTH]
            search_type = doc.get("search_type", "unknown")
            score = doc.get("combined_score", doc.get("semantic_score", doc.get("text_score", 0)))
            preview = content_snippet[:ModelConstants.MAX_CONTEXT_PREVIEW_LENGTH]
            print(f"      - Doc {doc['id']}: {search_type} (score: {score:.3f}) - '{preview}...'")
            context_parts.append(content_snippet)
        
        return " ".join(context_parts)
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a response using the AI service."""
        from .chatbot import ModelConstants  # Import here to avoid circular imports

        # Get prediction tokens from chatbot instance if available, otherwise use default
        if self.chatbot_instance and hasattr(self.chatbot_instance, 'get_output_length'):
            prediction_tokens = self.chatbot_instance.get_output_length()
        else:
            prediction_tokens = ModelConstants.DEFAULT_PREDICTION_TOKENS

        return self.ai_service.generate_text(prompt, f"n_predict={prediction_tokens}")
    
    def _clean_response(self, response: str) -> str:
        """Clean the response by removing unwanted parts."""
        if "Human:" in response:
            response = response.split("Human:")[0]
        return response.strip()
    
    def _reset_inference_state(self) -> None:
        """Reset the inference state to avoid corruption."""
        try:
            from .chatbot import ModelConstants  # Import here to avoid circular imports
            with self.ai_service.db_manager.get_connection() as conn:
                conn.execute("SELECT llm_context_free()")
                conn.execute(f"SELECT llm_context_create('n_ctx={ModelConstants.DEFAULT_CONTEXT_SIZE}')")
                conn.commit()
            
            with self.ai_service.db_manager.get_connection() as conn:
                conn.execute("SELECT llm_sampler_free()")
                conn.commit()
            self.ai_service.configure_sampler_greedy()
            
        except Exception as e:
            print(f"Failed to reset inference state: {e}")
    
    def get_handler_type(self) -> str:
        return "rag"
    
    def set_search_limit(self, limit: int) -> None:
        """Set the search limit for context retrieval."""
        self.search_limit = max(1, limit)


class MessageHandlerFactory:
    """
    Factory for creating message handlers.
    
    This factory encapsulates the creation logic for different message handlers
    and provides a clean interface for handler instantiation.
    """
    
    @staticmethod
    def create_handler(handler_type: str, ai_service: AIService, vector_service: VectorService,
                      search_context: SearchContext, error_recovery: ErrorRecoveryHandler,
                      **kwargs) -> MessageHandler:
        """
        Create a message handler of the specified type.
        
        Args:
            handler_type: Type of handler ('simple' or 'rag')
            ai_service: AI service instance
            vector_service: Vector service instance
            search_context: Search context instance
            error_recovery: Error recovery handler
            **kwargs: Additional arguments for specific handlers
            
        Returns:
            MessageHandler: The created handler instance
            
        Raises:
            ValueError: If handler_type is not recognized
        """
        if handler_type == "rag":
            search_limit = kwargs.get("search_limit", 2)
            chatbot_instance = kwargs.get("chatbot_instance", None)
            return RAGMessageHandler(ai_service, vector_service, search_context, error_recovery, search_limit, chatbot_instance)
        else:
            raise ValueError(f"Unknown handler type: {handler_type}. Only 'rag' is supported.")
    
    @staticmethod
    def get_available_types() -> List[str]:
        """Get list of available handler types."""
        return ["rag"]
