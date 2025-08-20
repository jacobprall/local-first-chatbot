"""
Search Strategy Pattern implementation for knowledge base search.

This module implements the Strategy pattern to encapsulate different search
algorithms and make them interchangeable at runtime.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .vector import VectorService


class SearchStrategy(ABC):
    """
    Abstract base class for search strategies.
    
    This interface defines the contract for different search implementations,
    allowing them to be used interchangeably.
    """
    
    @abstractmethod
    def search(self, query: str, vector_service: VectorService, 
               query_embedding: Optional[bytes] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Execute the search strategy.
        
        Args:
            query: The search query string
            vector_service: Vector service instance
            query_embedding: Pre-computed query embedding (optional)
            limit: Maximum number of results to return
            
        Returns:
            List of search results with metadata
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this search strategy."""
        pass
    
    @abstractmethod
    def requires_embedding(self) -> bool:
        """Check if this strategy requires query embeddings."""
        pass


class HybridSearchStrategy(SearchStrategy):
    """
    Hybrid search strategy combining semantic and full-text search.
    
    This strategy provides the best of both worlds by combining
    vector similarity search with traditional keyword matching.
    """
    
    def search(self, query: str, vector_service: VectorService, 
               query_embedding: Optional[bytes] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Execute hybrid search combining semantic and text search."""
        try:
            print(f"   🔄 Executing hybrid search strategy...")
            results = vector_service.hybrid_search(query, query_embedding, limit)
            
            if results:
                print(f"   ✅ Hybrid search returned {len(results)} results")
                # Add strategy metadata
                for result in results:
                    result["search_strategy"] = self.get_strategy_name()
                return results
            else:
                print(f"   ⚠️  Hybrid search returned no results")
                return []
                
        except Exception as e:
            print(f"   ❌ Hybrid search failed: {e}")
            return []
    
    def get_strategy_name(self) -> str:
        return "hybrid"
    
    def requires_embedding(self) -> bool:
        return True


class SemanticSearchStrategy(SearchStrategy):
    """
    Pure semantic search strategy using vector embeddings.
    
    This strategy focuses on semantic similarity and is best for
    conceptual queries that may not match exact keywords.
    """
    
    def search(self, query: str, vector_service: VectorService, 
               query_embedding: Optional[bytes] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Execute semantic search using vector similarity."""
        if not query_embedding:
            print(f"   ⚠️  No embedding provided for semantic search")
            return []
        
        try:
            print(f"   🔄 Executing semantic search strategy...")
            results = vector_service.similarity_search(query_embedding, limit)
            
            if results:
                print(f"   ✅ Semantic search returned {len(results)} results")
                # Add strategy metadata
                for result in results:
                    result["search_strategy"] = self.get_strategy_name()
                    result["search_type"] = "semantic"
                return results
            else:
                print(f"   ⚠️  Semantic search returned no results")
                return []
                
        except Exception as e:
            print(f"   ❌ Semantic search failed: {e}")
            return []
    
    def get_strategy_name(self) -> str:
        return "semantic"
    
    def requires_embedding(self) -> bool:
        return True


class TextSearchStrategy(SearchStrategy):
    """
    Full-text search strategy using keyword matching.
    
    This strategy is best for exact keyword matches and works
    without requiring embeddings or model loading.
    """
    
    def search(self, query: str, vector_service: VectorService, 
               query_embedding: Optional[bytes] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Execute full-text search using keyword matching."""
        try:
            print(f"   🔄 Executing text search strategy...")
            results = vector_service.search_documents(query, limit)
            
            if results:
                print(f"   ✅ Text search returned {len(results)} results")
                # Add strategy metadata
                for result in results:
                    result["search_strategy"] = self.get_strategy_name()
                    result["search_type"] = "text"
                return results
            else:
                print(f"   ⚠️  Text search returned no results")
                return []
                
        except Exception as e:
            print(f"   ❌ Text search failed: {e}")
            return []
    
    def get_strategy_name(self) -> str:
        return "text"
    
    def requires_embedding(self) -> bool:
        return False


class FallbackSearchStrategy(SearchStrategy):
    """
    Fallback search strategy that tries multiple strategies in order.
    
    This strategy attempts different search methods until one succeeds,
    providing robustness when individual strategies fail.
    """
    
    def __init__(self, strategies: List[SearchStrategy]):
        """
        Initialize with a list of strategies to try in order.
        
        Args:
            strategies: List of search strategies to try in order
        """
        self.strategies = strategies
    
    def search(self, query: str, vector_service: VectorService, 
               query_embedding: Optional[bytes] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Try each strategy until one succeeds."""
        print(f"   🔄 Executing fallback search strategy...")
        
        for i, strategy in enumerate(self.strategies):
            try:
                print(f"   🔄 Trying strategy {i+1}/{len(self.strategies)}: {strategy.get_strategy_name()}")
                
                # Skip strategies that require embeddings if we don't have one
                if strategy.requires_embedding() and not query_embedding:
                    print(f"   ⏭️  Skipping {strategy.get_strategy_name()} (no embedding)")
                    continue
                
                results = strategy.search(query, vector_service, query_embedding, limit)
                
                if results:
                    print(f"   ✅ Fallback succeeded with {strategy.get_strategy_name()}")
                    # Add fallback metadata
                    for result in results:
                        result["fallback_strategy"] = strategy.get_strategy_name()
                        result["search_strategy"] = self.get_strategy_name()
                    return results
                    
            except Exception as e:
                print(f"   ❌ Strategy {strategy.get_strategy_name()} failed: {e}")
                continue
        
        print(f"   ❌ All fallback strategies failed")
        return []
    
    def get_strategy_name(self) -> str:
        return "fallback"
    
    def requires_embedding(self) -> bool:
        # Fallback requires embedding if any of its strategies do
        return any(strategy.requires_embedding() for strategy in self.strategies)


class SearchContext:
    """
    Context class for managing search strategies.
    
    This class provides a clean interface for executing searches
    with different strategies and handles strategy selection logic.
    """
    
    def __init__(self, strategy: SearchStrategy):
        """
        Initialize with a search strategy.
        
        Args:
            strategy: The search strategy to use
        """
        self._strategy = strategy
    
    def set_strategy(self, strategy: SearchStrategy) -> None:
        """Change the search strategy at runtime."""
        self._strategy = strategy
    
    def search(self, query: str, vector_service: VectorService, 
               query_embedding: Optional[bytes] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Execute search using the current strategy."""
        return self._strategy.search(query, vector_service, query_embedding, limit)
    
    def get_current_strategy_name(self) -> str:
        """Get the name of the current strategy."""
        return self._strategy.get_strategy_name()
    
    def requires_embedding(self) -> bool:
        """Check if current strategy requires embeddings."""
        return self._strategy.requires_embedding()


# Predefined strategy configurations
class SearchStrategies:
    """Factory class for creating common search strategy configurations."""
    
    @staticmethod
    def hybrid() -> SearchStrategy:
        """Create a hybrid search strategy."""
        return HybridSearchStrategy()
    
    @staticmethod
    def semantic() -> SearchStrategy:
        """Create a semantic search strategy."""
        return SemanticSearchStrategy()
    
    @staticmethod
    def text() -> SearchStrategy:
        """Create a text search strategy."""
        return TextSearchStrategy()
    
    @staticmethod
    def robust_fallback() -> SearchStrategy:
        """Create a robust fallback strategy that tries multiple approaches."""
        return FallbackSearchStrategy([
            HybridSearchStrategy(),
            SemanticSearchStrategy(),
            TextSearchStrategy()
        ])
    
    @staticmethod
    def embedding_fallback() -> SearchStrategy:
        """Create a fallback strategy for when embeddings are available."""
        return FallbackSearchStrategy([
            HybridSearchStrategy(),
            SemanticSearchStrategy()
        ])
    
    @staticmethod
    def text_only_fallback() -> SearchStrategy:
        """Create a fallback strategy that doesn't require embeddings."""
        return FallbackSearchStrategy([
            TextSearchStrategy()
        ])
