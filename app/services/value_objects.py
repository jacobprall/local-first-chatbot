"""
Type-safe value objects for structured data representation.

Replaces primitive dictionaries and lists with strongly-typed objects
that provide validation, methods, and clear interfaces.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class SearchType(Enum):
    """Enumeration of search types."""
    SEMANTIC = "semantic"
    TEXT = "text"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SearchResult:
    """
    Structured representation of a knowledge base search result.

    Provides type safety and utility methods for search result data
    that was previously represented as primitive dictionaries.
    """
    id: int
    content: str
    search_type: SearchType = SearchType.UNKNOWN
    semantic_score: Optional[float] = None
    text_score: Optional[float] = None
    combined_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_best_score(self) -> float:
        """Get the best available score for this result."""
        if self.combined_score is not None:
            return self.combined_score
        elif self.semantic_score is not None:
            return self.semantic_score
        elif self.text_score is not None:
            return self.text_score
        else:
            return 0.0
    
    def get_content_preview(self, max_length: int = 100) -> str:
        """Get a truncated preview of the content."""
        if len(self.content) <= max_length:
            return self.content
        return f"{self.content[:max_length]}..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "id": self.id,
            "content": self.content,
            "search_type": self.search_type.value,
            "semantic_score": self.semantic_score,
            "text_score": self.text_score,
            "combined_score": self.combined_score,
            **self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Create SearchResult from dictionary."""
        search_type_str = data.get("search_type", "unknown")
        try:
            search_type = SearchType(search_type_str)
        except ValueError:
            search_type = SearchType.UNKNOWN
        
        # Extract known fields and put the rest in metadata
        known_fields = {"id", "content", "search_type", "semantic_score", "text_score", "combined_score"}
        metadata = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(
            id=data.get("id", 0),
            content=data.get("content", ""),
            search_type=search_type,
            semantic_score=data.get("semantic_score"),
            text_score=data.get("text_score"),
            combined_score=data.get("combined_score"),
            metadata=metadata
        )


@dataclass(frozen=True)
class SearchResults:
    """
    Value object representing a collection of search results.
    
    This provides additional functionality beyond a simple list.
    """
    results: List[SearchResult] = field(default_factory=list)
    query: str = ""
    total_found: int = 0
    search_time_ms: Optional[float] = None
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __iter__(self):
        return iter(self.results)
    
    def __getitem__(self, index):
        return self.results[index]
    
    def is_empty(self) -> bool:
        """Check if there are no results."""
        return len(self.results) == 0
    
    def get_top_result(self) -> Optional[SearchResult]:
        """Get the highest scoring result."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.get_best_score())
    
    def filter_by_score(self, min_score: float) -> 'SearchResults':
        """Filter results by minimum score."""
        filtered = [r for r in self.results if r.get_best_score() >= min_score]
        return SearchResults(
            results=filtered,
            query=self.query,
            total_found=len(filtered),
            search_time_ms=self.search_time_ms
        )
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries for backward compatibility."""
        return [result.to_dict() for result in self.results]


@dataclass(frozen=True)
class ModelInfo:
    """
    Value object representing model information.
    
    This replaces primitive dictionaries for model status and information.
    """
    status: str
    model_path: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    chat_active: bool = False
    version: Optional[str] = None
    loaded_at: Optional[datetime] = None
    
    def is_loaded(self) -> bool:
        """Check if the model is successfully loaded."""
        return self.status == "Model loaded"
    
    def has_error(self) -> bool:
        """Check if there's an error with the model."""
        return "error" in self.status.lower()
    
    def get_model_name(self) -> str:
        """Get the model name from the path."""
        if not self.model_path:
            return "Unknown"
        return self.model_path.split("/")[-1] if "/" in self.model_path else self.model_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "status": self.status,
            "model_path": self.model_path,
            "model_config": self.model_config,
            "chat_active": self.chat_active,
            "version": self.version,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None
        }


@dataclass(frozen=True)
class KnowledgeStats:
    """
    Value object representing knowledge base statistics.
    
    This replaces primitive dictionaries for knowledge base information.
    """
    document_count: int = 0
    chunk_count: int = 0
    rag_enabled: bool = False
    vector_extension: bool = False
    vector_initialized: bool = False
    documents_with_embeddings: int = 0
    
    def get_embedding_coverage(self) -> float:
        """Get the percentage of documents with embeddings."""
        if self.document_count == 0:
            return 0.0
        return (self.documents_with_embeddings / self.document_count) * 100
    
    def is_ready_for_rag(self) -> bool:
        """Check if the knowledge base is ready for RAG operations."""
        return (self.rag_enabled and 
                self.vector_extension and 
                self.documents_with_embeddings > 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "rag_enabled": self.rag_enabled,
            "vector_extension": self.vector_extension,
            "vector_initialized": self.vector_initialized,
            "documents_with_embeddings": self.documents_with_embeddings,
            "embedding_coverage": self.get_embedding_coverage(),
            "ready_for_rag": self.is_ready_for_rag()
        }


@dataclass(frozen=True)
class FileInfo:
    """
    Value object representing uploaded file information.
    
    This replaces primitive dictionaries for file metadata.
    """
    id: int
    filename: str
    file_type: str
    size: int
    uploaded_at: datetime
    processed: bool = False
    chunk_count: int = 0
    
    def get_size_formatted(self) -> str:
        """Get human-readable file size."""
        if self.size < 1024:
            return f"{self.size} B"
        elif self.size < 1024 * 1024:
            return f"{self.size / 1024:.1f} KB"
        else:
            return f"{self.size / (1024 * 1024):.1f} MB"
    
    def get_file_extension(self) -> str:
        """Get the file extension."""
        return self.filename.split(".")[-1].lower() if "." in self.filename else ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "id": self.id,
            "filename": self.filename,
            "file_type": self.file_type,
            "size": self.size,
            "size_formatted": self.get_size_formatted(),
            "uploaded_at": self.uploaded_at.isoformat(),
            "processed": self.processed,
            "chunk_count": self.chunk_count,
            "file_extension": self.get_file_extension()
        }


@dataclass(frozen=True)
class ChatSession:
    """
    Value object representing a chat session.
    
    This replaces primitive dictionaries for chat session information.
    """
    id: str
    title: str
    created_at: datetime
    model_used: Optional[str] = None
    rag_enabled: bool = False
    message_count: int = 0
    last_activity: Optional[datetime] = None
    
    def get_duration_minutes(self) -> Optional[float]:
        """Get session duration in minutes."""
        if not self.last_activity:
            return None
        delta = self.last_activity - self.created_at
        return delta.total_seconds() / 60
    
    def is_recent(self, hours: int = 24) -> bool:
        """Check if the session is recent."""
        if not self.last_activity:
            return False
        delta = datetime.now() - self.last_activity
        return delta.total_seconds() < (hours * 3600)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "model_used": self.model_used,
            "rag_enabled": self.rag_enabled,
            "message_count": self.message_count,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "duration_minutes": self.get_duration_minutes(),
            "is_recent": self.is_recent()
        }


@dataclass(frozen=True)
class ErrorStats:
    """
    Value object representing error recovery statistics.
    
    This replaces primitive dictionaries for error information.
    """
    total_errors: int = 0
    successful_recoveries: int = 0
    recovery_rate: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    recovery_strategies_used: Dict[str, int] = field(default_factory=dict)
    
    def get_most_common_error(self) -> Optional[str]:
        """Get the most common error type."""
        if not self.error_types:
            return None
        return max(self.error_types.items(), key=lambda x: x[1])[0]
    
    def get_most_effective_strategy(self) -> Optional[str]:
        """Get the most effective recovery strategy."""
        if not self.recovery_strategies_used:
            return None
        return max(self.recovery_strategies_used.items(), key=lambda x: x[1])[0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "total_errors": self.total_errors,
            "successful_recoveries": self.successful_recoveries,
            "recovery_rate": self.recovery_rate,
            "error_types": self.error_types,
            "recovery_strategies_used": self.recovery_strategies_used,
            "most_common_error": self.get_most_common_error(),
            "most_effective_strategy": self.get_most_effective_strategy()
        }
