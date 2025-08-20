import os
import json
import hashlib
from typing import List, Dict, Any, Optional, BinaryIO
from .database import DatabaseManager


class VectorService:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        # Check if vector extension is available
        self.vector_enabled = self._check_vector_extension()
        
        # Initialize vector tables
        self.tables_initialized = False
        self._init_tables()
    
    def _check_vector_extension(self) -> bool:
        """Check if the vector extension is available"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("SELECT vector_version()")
                version = cursor.fetchone()
                if version:
                    print(f"Vector extension loaded successfully, version: {version[0]}")
                    return True
        except Exception as e:
            print(f"Vector extension not available: {e}")
        return False
    
    def _get_embedding_dimension(self) -> int:
        """Detect the embedding dimension from existing embeddings or test generation"""
        try:
            # First, check if we have existing embeddings
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT embedding FROM documents
                    WHERE embedding IS NOT NULL
                    LIMIT 1
                """)
                result = cursor.fetchone()

                if result and result[0]:
                    embedding_bytes = result[0]
                    dimension = len(embedding_bytes) // 4
                    print(f"Detected embedding dimension from existing data: {dimension}")
                    return dimension

            # If no existing embeddings, try to generate a test embedding
            try:
                from .ai import AIService
                ai_service = AIService(self.db_manager)
                test_embedding = ai_service.generate_embedding("test")
                if test_embedding:
                    dimension = len(test_embedding) // 4
                    print(f"Detected embedding dimension from test generation: {dimension}")
                    return dimension
            except Exception as e:
                print(f"Could not generate test embedding: {e}")

            # Default fallback
            print("Using default embedding dimension: 4096")
            return 4096

        except Exception as e:
            print(f"Error detecting embedding dimension: {e}")
            return 4096

    def _init_tables(self) -> None:
        """Initialize vector storage tables"""
        try:
            with self.db_manager.get_connection() as conn:
                # Create documents table for storing text chunks with embeddings
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        embedding BLOB,
                        metadata TEXT DEFAULT '{}',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Create files table for storing uploaded files
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS uploaded_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        file_hash TEXT NOT NULL UNIQUE,
                        content BLOB,
                        extracted_text TEXT,
                        upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.commit()

                # Initialize vector support for documents table if vector extension is available
                if self.vector_enabled:
                    try:
                        # Detect the appropriate embedding dimension
                        embedding_dim = self._get_embedding_dimension()

                        # Initialize the vector column with detected dimension
                        conn.execute(f"""
                            SELECT vector_init('documents', 'embedding', 'type=FLOAT32,dimension={embedding_dim},distance=COSINE')
                        """)
                        conn.commit()
                        print(f"Vector support initialized for documents table with {embedding_dim} dimensions")
                        self.vector_initialized = True
                        self.embedding_dimension = embedding_dim
                    except Exception as e:
                        print(f"Vector init failed (may already be initialized): {e}")
                        # Try to check if vector extension works anyway
                        try:
                            conn.execute("SELECT vector_version()")
                            print("Vector extension available, assuming table already initialized")
                            self.vector_initialized = True
                            self.embedding_dimension = self._get_embedding_dimension()
                        except:
                            print("Vector extension not working properly")
                            self.vector_initialized = False
                            self.embedding_dimension = None
                else:
                    self.vector_initialized = False
                    self.embedding_dimension = None

                print("Vector tables initialized successfully")
                self.tables_initialized = True

        except Exception as e:
            print(f"Warning: Could not initialize vector tables: {e}")
            self.tables_initialized = False
    
    def _validate_embedding_dimension(self, embedding: bytes) -> bool:
        """Validate that the embedding has the expected dimension"""
        if not embedding:
            return False

        embedding_dim = len(embedding) // 4
        expected_dim = getattr(self, 'embedding_dimension', None)

        if expected_dim is None:
            # If we don't know the expected dimension, detect it
            expected_dim = self._get_embedding_dimension()
            self.embedding_dimension = expected_dim

        if embedding_dim != expected_dim:
            print(f"⚠️ Embedding dimension mismatch: got {embedding_dim}, expected {expected_dim}")
            return False

        return True

    def add_embedding_to_document(self, document_id: int, embedding: bytes) -> bool:
        """Add an embedding to an existing document"""
        if not self.vector_enabled:
            return False

        # Validate embedding dimension
        if not self._validate_embedding_dimension(embedding):
            print(f"Skipping embedding for document {document_id} due to dimension mismatch")
            return False

        try:
            # Convert embedding bytes to JSON format for sqlite-vector
            import struct
            float_count = len(embedding) // 4
            embedding_array = list(struct.unpack(f'{float_count}f', embedding))
            embedding_json = json.dumps(embedding_array)

            with self.db_manager.get_connection() as conn:
                cursor = conn.execute(
                    "UPDATE documents SET embedding = vector_convert_f32(?) WHERE id = ?",
                    (embedding_json, document_id)
                )
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Failed to add embedding to document: {e}")
            return False
    
    def get_document(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get a document by ID"""
        result = self.db_manager.execute_query(
            "SELECT id, content, metadata, created_at FROM documents WHERE id = ?",
            (document_id,)
        )
        if result:
            row = result[0]
            return {
                "id": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]),
                "created_at": row[3]
            }
        return None
    
    def search_documents(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Simple text search in documents"""
        result = self.db_manager.execute_query(
            """SELECT id, content, metadata, created_at 
               FROM documents 
               WHERE content LIKE ? 
               ORDER BY created_at DESC 
               LIMIT ?""",
            (f"%{query}%", limit)
        )
        
        documents = []
        for row in result:
            documents.append({
                "id": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]),
                "created_at": row[3],
                "search_type": "text"
            })
        return documents
    
    def hybrid_search(self, query: str, query_embedding: bytes = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Hybrid search combining semantic and full-text search with score fusion"""
        print(f"\n🔍 HYBRID SEARCH START: '{query}' (limit={limit})")
        print(f"   Vector enabled: {self.vector_enabled}")
        print(f"   Has embedding: {query_embedding is not None}")
        
        results = {}
        semantic_count = 0
        text_count = 0
        
        # 1. Semantic Search (if embeddings available)
        if query_embedding and self.vector_enabled:
            print(f"   🧠 Attempting semantic search...")
            try:
                semantic_results = self.similarity_search(query_embedding, limit * 2)  # Get more for fusion
                semantic_count = len(semantic_results)
                print(f"   ✅ Semantic search found {semantic_count} results")
                
                for i, doc in enumerate(semantic_results):
                    doc_id = doc["id"]
                    # Convert distance to similarity score (lower distance = higher similarity)
                    semantic_score = 1.0 / (1.0 + doc["distance"]) if "distance" in doc else 0.5
                    print(f"      - Doc {doc_id}: distance={doc.get('distance', 'N/A')}, score={semantic_score:.3f}")
                    
                    if doc_id not in results:
                        doc["search_type"] = "semantic"
                        doc["semantic_score"] = semantic_score
                        results[doc_id] = doc
                    else:
                        results[doc_id]["semantic_score"] = semantic_score
                        results[doc_id]["search_type"] = "hybrid"
                        
            except Exception as e:
                print(f"   ❌ Semantic search failed: {e}")
        else:
            if not self.vector_enabled:
                print(f"   ⚠️  Skipping semantic search: vector extension not enabled")
            else:
                print(f"   ⚠️  Skipping semantic search: no embedding provided")
        
        # 2. Full-Text Search
        print(f"   📝 Attempting text search...")
        try:
            # Enhanced text search with better scoring
            text_results = self._enhanced_text_search(query, limit * 2)
            text_count = len(text_results)
            print(f"   ✅ Text search found {text_count} results")
            
            for i, doc in enumerate(text_results):
                doc_id = doc["id"]
                # Score based on position and relevance
                text_score = doc.get("text_score", 1.0 / (i + 1))
                print(f"      - Doc {doc_id}: text_score={text_score:.3f}")
                
                if doc_id not in results:
                    doc["search_type"] = "text"
                    doc["text_score"] = text_score
                    results[doc_id] = doc
                else:
                    results[doc_id]["text_score"] = text_score
                    results[doc_id]["search_type"] = "hybrid"
                    
        except Exception as e:
            print(f"   ❌ Text search failed: {e}")
        
        # 3. Score Fusion and Ranking
        print(f"   🔗 Fusing scores for {len(results)} unique documents...")
        final_results = []
        for doc in results.values():
            semantic_score = doc.get("semantic_score", 0.0)
            text_score = doc.get("text_score", 0.0)
            
            # Weighted combination (favor semantic slightly if available)
            if semantic_score > 0 and text_score > 0:
                # Both searches found this document - boost it
                combined_score = (0.6 * semantic_score + 0.4 * text_score) * 1.2
                print(f"      - Doc {doc['id']}: HYBRID (sem={semantic_score:.3f}, text={text_score:.3f}) -> {combined_score:.3f}")
            elif semantic_score > 0:
                combined_score = semantic_score
                print(f"      - Doc {doc['id']}: SEMANTIC ONLY -> {combined_score:.3f}")
            else:
                combined_score = text_score
                print(f"      - Doc {doc['id']}: TEXT ONLY -> {combined_score:.3f}")
            
            doc["combined_score"] = combined_score
            final_results.append(doc)
        
        # Sort by combined score and return top results
        final_results.sort(key=lambda x: x["combined_score"], reverse=True)
        final_count = min(len(final_results), limit)
        
        print(f"🎯 HYBRID SEARCH RESULT: {final_count}/{len(final_results)} results (semantic: {semantic_count}, text: {text_count})")
        for i, doc in enumerate(final_results[:limit]):
            print(f"   {i+1}. Doc {doc['id']}: {doc['search_type']} (score: {doc['combined_score']:.3f})")
        print("")
        
        return final_results[:limit]
    
    def _enhanced_text_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Enhanced text search with better scoring"""
        query_terms = query.lower().split()
        
        # Build a more sophisticated text search
        search_conditions = []
        search_params = []
        
        # Exact phrase search (highest score)
        search_conditions.append("content LIKE ?")
        search_params.append(f"%{query}%")
        
        # Individual term search
        for term in query_terms:
            if len(term) > 2:  # Skip very short terms
                search_conditions.append("content LIKE ?")
                search_params.append(f"%{term}%")
        
        search_sql = f"""
            SELECT id, content, metadata, created_at,
                   CASE 
                       WHEN content LIKE ? THEN 3.0  -- Exact phrase match
                       ELSE 1.0 + (LENGTH(content) - LENGTH(REPLACE(LOWER(content), ?, ''))) / LENGTH(?)  -- Term frequency
                   END as text_score
            FROM documents 
            WHERE {' OR '.join(search_conditions)}
            ORDER BY text_score DESC, created_at DESC
            LIMIT ?
        """
        
        # Add exact phrase params for scoring
        final_params = [f"%{query}%", query.lower(), query.lower()] + search_params + [limit]
        
        try:
            result = self.db_manager.execute_query(search_sql, final_params)
            
            documents = []
            for row in result:
                documents.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": json.loads(row[2]),
                    "created_at": row[3],
                    "text_score": row[4],
                    "search_type": "text"
                })
            return documents
            
        except Exception as e:
            print(f"Enhanced text search failed: {e}")
            # Fallback to simple search
            return self.search_documents(query, limit)
    
    def similarity_search(self, query_embedding: bytes, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar documents using vector similarity (safe implementation to avoid segfaults)"""
        print(f"      🔍 SIMILARITY SEARCH: Starting with {len(query_embedding)} byte embedding, limit={limit}")

        if not self.vector_enabled:
            print("      ❌ Vector extension not available, falling back to recent documents")
            return []  # Fallback when search fails

        # Validate query embedding dimension
        if not self._validate_embedding_dimension(query_embedding):
            print("      ❌ Query embedding dimension mismatch, skipping similarity search")
            return []

        try:
            # Check if we have documents with embeddings
            with self.db_manager.get_connection() as conn:
                count_cursor = conn.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL")
                embedding_count = count_cursor.fetchone()[0]
                print(f"      📈 Found {embedding_count} documents with embeddings in database")

                if embedding_count == 0:
                    print("      ⚠️  No documents with embeddings found, returning empty results")
                    return []

            # Use Python-based similarity search to avoid segfaults
            print(f"      🐍 Using Python-based similarity search (safer than sqlite-vector)")
            return self._python_similarity_search(query_embedding, limit)

        except Exception as e:
            print(f"Similarity search completely failed: {e}")
            print(f"      🔄 Falling back to no semantic results")
            return []  # Return empty results, let text search handle it
    
    def _python_similarity_search(self, query_embedding: bytes, limit: int = 5) -> List[Dict[str, Any]]:
        """Python-based similarity search to avoid segfaults in sqlite-vector"""
        try:
            import struct
            import numpy as np
            
            # Convert query embedding to numpy array
            float_count = len(query_embedding) // 4
            query_vector = np.array(struct.unpack(f'{float_count}f', query_embedding), dtype=np.float32)
            query_norm = np.linalg.norm(query_vector)
            
            if query_norm == 0:
                print("      ⚠️  Query embedding has zero norm, cannot compute similarity")
                return []
            
            # Normalize query vector for cosine similarity
            query_vector = query_vector / query_norm
            
            # Get all documents with embeddings
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id, content, metadata, created_at, embedding 
                    FROM documents 
                    WHERE embedding IS NOT NULL
                """)
                
                similarities = []
                for row in cursor.fetchall():
                    doc_id, content, metadata, created_at, embedding_bytes = row
                    
                    try:
                        # Convert document embedding to numpy array
                        doc_vector = np.array(struct.unpack(f'{len(embedding_bytes) // 4}f', embedding_bytes), dtype=np.float32)
                        doc_norm = np.linalg.norm(doc_vector)
                        
                        if doc_norm == 0:
                            continue  # Skip zero vectors
                        
                        # Normalize document vector
                        doc_vector = doc_vector / doc_norm
                        
                        # Compute cosine similarity
                        similarity = np.dot(query_vector, doc_vector)
                        distance = 1.0 - similarity  # Convert similarity to distance
                        
                        similarities.append({
                            "id": doc_id,
                            "content": content,
                            "metadata": json.loads(metadata),
                            "created_at": created_at,
                            "distance": float(distance),
                            "similarity": float(similarity)
                        })
                        
                    except Exception as e:
                        print(f"      ⚠️  Failed to process embedding for doc {doc_id}: {e}")
                        continue
                
                # Sort by distance (ascending = most similar first)
                similarities.sort(key=lambda x: x["distance"])
                
                # Return top results
                results = similarities[:limit]
                print(f"      ✅ Python similarity search returned {len(results)} results")
                
                for i, doc in enumerate(results):
                    print(f"         {i+1}. Doc {doc['id']}: similarity={doc['similarity']:.4f}")
                
                return results
                
        except ImportError:
            print("      ❌ NumPy not available, falling back to basic similarity")
            return self._basic_similarity_search(query_embedding, limit)
        except Exception as e:
            print(f"      ❌ Python similarity search failed: {e}")
            return []
    
    def _basic_similarity_search(self, query_embedding: bytes, limit: int = 5) -> List[Dict[str, Any]]:
        """Basic similarity search without numpy dependency"""
        try:
            import struct
            import math
            
            # Convert query embedding to list
            float_count = len(query_embedding) // 4
            query_vector = list(struct.unpack(f'{float_count}f', query_embedding))
            
            # Compute query vector magnitude
            query_mag = math.sqrt(sum(x * x for x in query_vector))
            if query_mag == 0:
                return []
            
            # Get all documents with embeddings
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    SELECT id, content, metadata, created_at, embedding 
                    FROM documents 
                    WHERE embedding IS NOT NULL
                """)
                
                similarities = []
                for row in cursor.fetchall():
                    doc_id, content, metadata, created_at, embedding_bytes = row
                    
                    try:
                        # Convert document embedding to list
                        doc_vector = list(struct.unpack(f'{len(embedding_bytes) // 4}f', embedding_bytes))
                        
                        # Compute document vector magnitude
                        doc_mag = math.sqrt(sum(x * x for x in doc_vector))
                        if doc_mag == 0:
                            continue
                        
                        # Compute cosine similarity
                        dot_product = sum(q * d for q, d in zip(query_vector, doc_vector))
                        similarity = dot_product / (query_mag * doc_mag)
                        distance = 1.0 - similarity
                        
                        similarities.append({
                            "id": doc_id,
                            "content": content,
                            "metadata": json.loads(metadata),
                            "created_at": created_at,
                            "distance": distance,
                            "similarity": similarity
                        })
                        
                    except Exception as e:
                        print(f"      ⚠️  Failed to process embedding for doc {doc_id}: {e}")
                        continue
                
                # Sort by distance (ascending = most similar first)
                similarities.sort(key=lambda x: x["distance"])
                
                # Return top results
                results = similarities[:limit]
                print(f"      ✅ Basic similarity search returned {len(results)} results")
                
                return results
                
        except Exception as e:
            print(f"      ❌ Basic similarity search failed: {e}")
            return []
    

    
    def get_document_count(self) -> int:
        """Get total number of unique documents by counting distinct file_id in metadata"""
        try:
            # Count unique file_id values from JSON metadata
            result = self.db_manager.execute_query("""
                SELECT COUNT(DISTINCT json_extract(metadata, '$.file_id')) 
                FROM documents 
                WHERE json_extract(metadata, '$.file_id') IS NOT NULL
            """)
            unique_docs = result[0][0] if result and result[0][0] is not None else 0
            
            # Also count manually added knowledge (without file_id)
            manual_result = self.db_manager.execute_query("""
                SELECT COUNT(DISTINCT json_extract(metadata, '$.source_type')) 
                FROM documents 
                WHERE json_extract(metadata, '$.source_type') = 'manual' 
                OR json_extract(metadata, '$.file_id') IS NULL
            """)
            manual_docs = manual_result[0][0] if manual_result and manual_result[0][0] is not None else 0
            
            return unique_docs + (1 if manual_docs > 0 else 0)
        except Exception as e:
            print(f"Error counting unique documents: {e}")
            # Fallback to simple count
            result = self.db_manager.execute_query("SELECT COUNT(*) FROM documents")
            return result[0][0] if result else 0
    
    def get_chunk_count(self) -> int:
        """Get total number of document chunks"""
        result = self.db_manager.execute_query("SELECT COUNT(*) FROM documents")
        return result[0][0] if result else 0
    
    def test_search_methods(self, query: str, limit: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Test different search methods for comparison"""
        results = {
            "text_search": [],
            "semantic_search": [],
            "hybrid_search": []
        }
        
        try:
            # Text search
            results["text_search"] = self.search_documents(query, limit)
        except Exception as e:
            print(f"Text search test failed: {e}")
        
        try:
            # Semantic search (if available)
            if self.vector_enabled:
                from .ai import AIService
                ai_service = AIService(self.db_manager)
                try:
                    query_embedding = ai_service.generate_embedding(query)
                    results["semantic_search"] = self.similarity_search(query_embedding, limit)
                except Exception as e:
                    print(f"Semantic search test failed: {e}")
        except Exception as e:
            print(f"Semantic search setup failed: {e}")
        
        try:
            # Hybrid search
            query_embedding = None
            if self.vector_enabled:
                from .ai import AIService
                ai_service = AIService(self.db_manager)
                try:
                    query_embedding = ai_service.generate_embedding(query)
                except:
                    pass
            results["hybrid_search"] = self.hybrid_search(query, query_embedding, limit)
        except Exception as e:
            print(f"Hybrid search test failed: {e}")
        
        return results
    
    def clear_all_documents(self) -> bool:
        """Clear all documents"""
        try:
            self.db_manager.execute_query("DELETE FROM documents")
            return True
        except Exception as e:
            print(f"Failed to clear documents: {e}")
            return False
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into simple chunks for demo purposes"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks
    
    def add_document_with_chunks(self, content: str, metadata: Dict[str, Any] = None, 
                                chunk_size: int = 500) -> List[int]:
        """Add a document by splitting it into chunks and return chunk IDs"""
        chunks = self._chunk_text(content, chunk_size)
        chunk_ids = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = (metadata or {}).copy()
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "original_length": len(content)
            })
            
            # Add chunk directly to database
            metadata_json = json.dumps(chunk_metadata)
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute(
                    "INSERT INTO documents (content, metadata) VALUES (?, ?) RETURNING id",
                    (chunk, metadata_json)
                )
                result = cursor.fetchone()
                conn.commit()
                chunk_id = result[0] if result else None
                
            if chunk_id:
                chunk_ids.append(chunk_id)
        
        # After adding all chunks, quantize the vectors if vector extension is available
        # Note: We don't auto-quantize here since chunks might not have embeddings yet
        # Quantization should be done explicitly after embeddings are added
        
        return chunk_ids
    
    def _extract_text_from_file(self, file_content: bytes, file_type: str, filename: str) -> str:
        """Extract text from various file types"""
        try:
            if file_type.startswith('text/') or file_type == 'application/json':
                # Plain text files
                return file_content.decode('utf-8', errors='ignore')
            
            elif filename.lower().endswith('.md'):
                # Markdown files
                return file_content.decode('utf-8', errors='ignore')
            
            elif filename.lower().endswith('.py'):
                # Python files
                text = file_content.decode('utf-8', errors='ignore')
                return f"Python code file: {filename}\n\n{text}"
            
            elif filename.lower().endswith(('.js', '.ts', '.jsx', '.tsx')):
                # JavaScript/TypeScript files
                text = file_content.decode('utf-8', errors='ignore')
                return f"JavaScript/TypeScript file: {filename}\n\n{text}"
            
            elif filename.lower().endswith(('.html', '.htm')):
                # HTML files - basic text extraction
                text = file_content.decode('utf-8', errors='ignore')
                # Simple HTML tag removal (basic)
                import re
                text = re.sub(r'<[^>]+>', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return f"HTML file: {filename}\n\n{text}"
            
            elif filename.lower().endswith('.csv'):
                # CSV files
                text = file_content.decode('utf-8', errors='ignore')
                return f"CSV file: {filename}\n\n{text}"
            
            elif filename.lower().endswith('.json'):
                # JSON files
                text = file_content.decode('utf-8', errors='ignore')
                try:
                    # Pretty print JSON for better readability
                    import json as json_lib
                    data = json_lib.loads(text)
                    formatted = json_lib.dumps(data, indent=2)
                    return f"JSON file: {filename}\n\n{formatted}"
                except:
                    return f"JSON file: {filename}\n\n{text}"
            
            else:
                # Try to decode as text for other file types
                try:
                    return file_content.decode('utf-8', errors='ignore')
                except:
                    return f"Binary file: {filename} (content not extractable as text)"
                    
        except Exception as e:
            print(f"Error extracting text from {filename}: {e}")
            return f"Error processing file: {filename}"
    
    def _calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA-256 hash of file content"""
        return hashlib.sha256(file_content).hexdigest()
    
    def upload_file(self, file_content: bytes, filename: str, file_type: str) -> Optional[int]:
        """Upload a file and return its ID"""
        try:
            # Calculate file hash to avoid duplicates
            file_hash = self._calculate_file_hash(file_content)
            file_size = len(file_content)
            
            # Check if file already exists
            existing = self.db_manager.execute_query(
                "SELECT id FROM uploaded_files WHERE file_hash = ?",
                (file_hash,)
            )
            if existing:
                print(f"File {filename} already exists (hash: {file_hash[:8]}...)")
                return existing[0][0]
            
            # Extract text content
            extracted_text = self._extract_text_from_file(file_content, file_type, filename)
            
            # Store file in database
            with self.db_manager.get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO uploaded_files 
                    (filename, file_type, file_size, file_hash, content, extracted_text) 
                    VALUES (?, ?, ?, ?, ?, ?) 
                    RETURNING id
                """, (filename, file_type, file_size, file_hash, file_content, extracted_text))
                result = cursor.fetchone()
                conn.commit()
            
            file_id = result[0] if result else None
            
            if file_id and extracted_text.strip():
                # Add extracted text to documents for RAG
                metadata = {
                    "source_type": "uploaded_file",
                    "filename": filename,
                    "file_type": file_type,
                    "file_id": file_id,
                    "file_size": file_size
                }
                
                # Add as chunked documents
                self.add_document_with_chunks(extracted_text, metadata)
                
            return file_id
            
        except Exception as e:
            print(f"Failed to upload file {filename}: {e}")
            return None
    
    def get_uploaded_files(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of uploaded files"""
        result = self.db_manager.execute_query("""
            SELECT id, filename, file_type, file_size, upload_date
            FROM uploaded_files 
            ORDER BY upload_date DESC 
            LIMIT ?
        """, (limit,))
        
        files = []
        for row in result:
            files.append({
                "id": row[0],
                "filename": row[1],
                "file_type": row[2],
                "file_size": row[3],
                "upload_date": row[4]
            })
        return files
    
    def delete_uploaded_file(self, file_id: int) -> bool:
        """Delete an uploaded file and its associated documents"""
        try:
            with self.db_manager.get_connection() as conn:
                # Delete associated documents - handle both integer and string file_id values
                # First try integer comparison (for newer uploads)
                cursor_docs = conn.execute("""
                    DELETE FROM documents
                    WHERE json_extract(metadata, '$.file_id') = ?
                """, (file_id,))
                deleted_docs = cursor_docs.rowcount

                # Also try string comparison (for legacy uploads or edge cases)
                cursor_docs_str = conn.execute("""
                    DELETE FROM documents
                    WHERE json_extract(metadata, '$.file_id') = ?
                """, (str(file_id),))
                deleted_docs += cursor_docs_str.rowcount

                # Delete the file
                cursor = conn.execute("DELETE FROM uploaded_files WHERE id = ?", (file_id,))
                conn.commit()

                print(f"Deleted {deleted_docs} document chunks for file {file_id}")
                return cursor.rowcount > 0
        except Exception as e:
            print(f"Failed to delete file: {e}")
            return False
    
    def quantize_vectors(self) -> bool:
        """Disabled quantization to avoid segfaults - using Python similarity instead"""
        print("Vector quantization disabled - using Python-based similarity search")
        return True
    
    def preload_quantized_vectors(self) -> bool:
        """Disabled preloading to avoid segfaults - using Python similarity instead"""
        print("Vector preloading disabled - using Python-based similarity search")
        return True
    
    def get_vector_stats(self) -> Dict[str, Any]:
        """Get vector-related statistics"""
        stats = {
            "vector_enabled": self.vector_enabled,
            "vector_initialized": getattr(self, 'vector_initialized', False),
            "documents_with_embeddings": 0,
            "quantized": False
        }
        
        if self.vector_enabled and self.tables_initialized:
            try:
                result = self.db_manager.execute_query("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL")
                stats["documents_with_embeddings"] = result[0][0] if result else 0
                
                # Check if vectors are quantized by trying to query the quantization table
                try:
                    conn = self.db_manager.get_persistent_connection()
                    quantized_result = conn.execute("SELECT COUNT(*) FROM vector_quantize_index WHERE table_name = 'documents'").fetchone()
                    stats["quantized"] = quantized_result and quantized_result[0] > 0
                except:
                    stats["quantized"] = False
                    
            except Exception as e:
                print(f"Failed to get vector stats: {e}")
        
        return stats
    
    def rebuild_quantization_if_needed(self) -> bool:
        """Disabled quantization rebuild to avoid segfaults - using Python similarity instead"""
        print("Quantization rebuild disabled - using Python-based similarity search")
        return True
    
    def fix_vector_embeddings(self) -> bool:
        """Fix corrupted vector embeddings by removing invalid ones"""
        if not self.vector_enabled:
            return False
        
        try:
            print("🔧 Checking for corrupted vector embeddings...")
            
            # Find documents with invalid embeddings
            with self.db_manager.get_connection() as conn:
                # Check for documents with embeddings that can't be processed
                cursor = conn.execute("""
                    SELECT id, embedding FROM documents 
                    WHERE embedding IS NOT NULL
                """)
                
                corrupted_ids = []
                valid_count = 0
                
                for row in cursor.fetchall():
                    doc_id, embedding = row
                    try:
                        # Try to convert the embedding to verify it's valid
                        if embedding:
                            import struct
                            float_count = len(embedding) // 4
                            if float_count > 0:
                                struct.unpack(f'{float_count}f', embedding)
                                valid_count += 1
                            else:
                                corrupted_ids.append(doc_id)
                    except Exception:
                        corrupted_ids.append(doc_id)
                
                print(f"   Found {valid_count} valid embeddings, {len(corrupted_ids)} corrupted")
                
                # Remove corrupted embeddings
                if corrupted_ids:
                    placeholders = ','.join(['?' for _ in corrupted_ids])
                    conn.execute(f"""
                        UPDATE documents 
                        SET embedding = NULL 
                        WHERE id IN ({placeholders})
                    """, corrupted_ids)
                    conn.commit()
                    print(f"   🗑️  Removed {len(corrupted_ids)} corrupted embeddings")
                
                return True
                
        except Exception as e:
            print(f"Failed to fix vector embeddings: {e}")
            return False
    
    def cleanup_orphaned_chunks(self) -> int:
        """Remove document chunks that reference non-existent files"""
        try:
            with self.db_manager.get_connection() as conn:
                # Find orphaned chunks - documents with file_id that don't exist in uploaded_files
                cursor = conn.execute("""
                    DELETE FROM documents
                    WHERE json_extract(metadata, '$.file_id') IS NOT NULL
                    AND json_extract(metadata, '$.file_id') NOT IN (
                        SELECT id FROM uploaded_files
                    )
                """)
                deleted_count = cursor.rowcount
                conn.commit()

                print(f"🗑️ Cleaned up {deleted_count} orphaned document chunks")
                return deleted_count

        except Exception as e:
            print(f"Failed to cleanup orphaned chunks: {e}")
            return 0

    def cleanup_mismatched_embeddings(self) -> int:
        """Remove embeddings with incorrect dimensions"""
        if not self.vector_enabled:
            return 0

        try:
            expected_dim = getattr(self, 'embedding_dimension', None)
            if expected_dim is None:
                expected_dim = self._get_embedding_dimension()
                self.embedding_dimension = expected_dim

            print(f"🔧 Checking for embeddings with incorrect dimensions (expected: {expected_dim})")

            with self.db_manager.get_connection() as conn:
                # Find embeddings with wrong dimensions
                cursor = conn.execute("""
                    SELECT id, embedding FROM documents
                    WHERE embedding IS NOT NULL
                """)

                mismatched_ids = []
                for row in cursor.fetchall():
                    doc_id, embedding = row
                    if embedding:
                        actual_dim = len(embedding) // 4
                        if actual_dim != expected_dim:
                            mismatched_ids.append(doc_id)
                            print(f"   Found mismatched embedding: doc {doc_id} has {actual_dim} dims, expected {expected_dim}")

                # Remove mismatched embeddings
                if mismatched_ids:
                    placeholders = ','.join(['?' for _ in mismatched_ids])
                    cursor = conn.execute(f"""
                        UPDATE documents
                        SET embedding = NULL
                        WHERE id IN ({placeholders})
                    """, mismatched_ids)
                    conn.commit()
                    print(f"🗑️ Cleared {len(mismatched_ids)} mismatched embeddings")
                    return len(mismatched_ids)
                else:
                    print("✅ All embeddings have correct dimensions")
                    return 0

        except Exception as e:
            print(f"Failed to cleanup mismatched embeddings: {e}")
            return 0

    def get_file_stats(self) -> Dict[str, Any]:
        """Get statistics about uploaded files"""
        if not getattr(self, 'tables_initialized', False):
            return {"total_files": 0, "total_size": 0, "file_types": {}}

        try:
            # Count files by type
            type_counts = self.db_manager.execute_query("""
                SELECT file_type, COUNT(*) as count
                FROM uploaded_files
                GROUP BY file_type
                ORDER BY count DESC
            """)

            # Total stats
            total_stats = self.db_manager.execute_query("""
                SELECT COUNT(*) as total_files,
                       SUM(file_size) as total_size
                FROM uploaded_files
            """)

            return {
                "total_files": total_stats[0][0] if total_stats and total_stats[0][0] is not None else 0,
                "total_size": total_stats[0][1] if total_stats and total_stats[0][1] is not None else 0,
                "file_types": dict(type_counts) if type_counts else {}
            }
        except Exception as e:
            print(f"Failed to get file stats: {e}")
            return {"total_files": 0, "total_size": 0, "file_types": {}}