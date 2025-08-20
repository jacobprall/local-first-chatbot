"""
Interface definition for the chatbot service.

This module defines the abstract interface that chatbot implementations
must follow, providing a contract for the core chatbot functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Generator
from .value_objects import SearchResults, ModelInfo, KnowledgeStats, ErrorStats


class IChatbot(ABC):
    """
    Abstract interface for chatbot implementations.
    
    This interface defines the core methods that any chatbot implementation
    must provide, ensuring consistency across different chatbot types.
    """
    

    
    @abstractmethod
    def search_knowledge(self, query: str, limit: int = 3) -> SearchResults:
        """
        Search the knowledge base for relevant information.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            SearchResults: Container with search results and metadata
        """
        pass
    
    @abstractmethod
    def add_knowledge(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add content to the knowledge base.
        
        Args:
            content: The content to add
            metadata: Optional metadata about the content
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """
        Get information about the loaded model.
        
        Returns:
            ModelInfo: Information about the current model
        """
        pass
    
    @abstractmethod
    def get_knowledge_stats(self) -> KnowledgeStats:
        """
        Get statistics about the knowledge base.
        
        Returns:
            KnowledgeStats: Statistics about the knowledge base
        """
        pass
    
    @abstractmethod
    def get_error_recovery_stats(self) -> ErrorStats:
        """
        Get error recovery statistics.
        
        Returns:
            ErrorStats: Error recovery statistics
        """
        pass
    

    
    @abstractmethod
    def upload_file(self, file_path: str, chunk_size: int = 1000) -> bool:
        """
        Upload and process a file into the knowledge base.
        
        Args:
            file_path: Path to the file to upload
            chunk_size: Size of text chunks for processing
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    

