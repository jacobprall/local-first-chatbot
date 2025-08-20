"""
Value objects for the chatbot application.

This module contains data classes that represent various entities and results
used throughout the application, providing type safety and structured data access.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class SearchResult:
    """Represents a single search result from the knowledge base."""
    id: str
    content: str
    score: float
    search_type: str = "unknown"
    semantic_score: Optional[float] = None
    text_score: Optional[float] = None
    combined_score: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Create a SearchResult from a dictionary."""
        return cls(
            id=str(data.get('id', '')),
            content=str(data.get('content', '')),
            score=float(data.get('score', 0.0)),
            search_type=str(data.get('search_type', 'unknown')),
            semantic_score=data.get('semantic_score'),
            text_score=data.get('text_score'),
            combined_score=data.get('combined_score', data.get('score', 0.0))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'content': self.content,
            'score': self.score,
            'search_type': self.search_type,
            'semantic_score': self.semantic_score,
            'text_score': self.text_score,
            'combined_score': self.combined_score
        }


@dataclass
class SearchResults:
    """Container for search results with metadata."""
    results: List[SearchResult]
    query: str
    total_found: int = 0
    search_time_ms: float = 0.0
    
    def __init__(self, results: List[SearchResult] = None, query: str = "", 
                 total_found: int = None, search_time_ms: float = 0.0):
        self.results = results or []
        self.query = query
        self.total_found = total_found if total_found is not None else len(self.results)
        self.search_time_ms = search_time_ms
    
    def is_empty(self) -> bool:
        """Check if there are no results."""
        return len(self.results) == 0
    
    def __len__(self) -> int:
        """Return the number of results."""
        return len(self.results)
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Convert results to list of dictionaries for backward compatibility."""
        return [result.to_dict() for result in self.results]


@dataclass
class ModelInfo:
    """Information about the loaded model."""
    status: str
    model_path: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    chat_active: bool = False
    version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'status': self.status,
            'model_path': self.model_path,
            'model_config': self.model_config,
            'chat_active': self.chat_active,
            'version': self.version
        }


@dataclass
class KnowledgeStats:
    """Statistics about the knowledge base."""
    document_count: int
    chunk_count: int
    rag_enabled: bool
    vector_extension: bool
    vector_initialized: bool = False
    documents_with_embeddings: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'document_count': self.document_count,
            'chunk_count': self.chunk_count,
            'rag_enabled': self.rag_enabled,
            'vector_extension': self.vector_extension,
            'vector_initialized': self.vector_initialized,
            'documents_with_embeddings': self.documents_with_embeddings
        }


@dataclass
class FileInfo:
    """Information about an uploaded file."""
    filename: str
    size: int
    upload_time: str
    file_type: str
    chunks_created: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'filename': self.filename,
            'size': self.size,
            'upload_time': self.upload_time,
            'file_type': self.file_type,
            'chunks_created': self.chunks_created
        }


@dataclass
class ChatSession:
    """Represents a chat session."""
    session_id: str
    title: str
    created_at: str
    message_count: int = 0
    last_activity: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'session_id': self.session_id,
            'title': self.title,
            'created_at': self.created_at,
            'message_count': self.message_count,
            'last_activity': self.last_activity
        }


@dataclass
class ErrorStats:
    """Statistics about error recovery."""
    total_errors: int
    successful_recoveries: int
    recovery_rate: float
    error_types: Dict[str, int]
    recovery_strategies_used: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'total_errors': self.total_errors,
            'successful_recoveries': self.successful_recoveries,
            'recovery_rate': self.recovery_rate,
            'error_types': self.error_types,
            'recovery_strategies_used': self.recovery_strategies_used
        }
