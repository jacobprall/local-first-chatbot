import sqlite3
import json
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from .database import DatabaseManager


class ChatHistoryManager:
    """Manages chat history persistence and retrieval"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._init_tables()
    
    def _init_tables(self) -> None:
        """Initialize chat history tables"""
        try:
            with self.db_manager.get_connection() as conn:
                # Create chat sessions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        message_count INTEGER DEFAULT 0,
                        model_used TEXT,
                        rag_enabled BOOLEAN DEFAULT FALSE
                    )
                """)
                
                # Create chat messages table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT DEFAULT '{}',
                        FOREIGN KEY (session_id) REFERENCES chat_sessions (id) ON DELETE CASCADE
                    )
                """)
                
                # Create index for faster queries
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id 
                    ON chat_messages (session_id, timestamp)
                """)
                
                conn.commit()
                
        except Exception as e:
            print(f"Warning: Could not initialize chat history tables: {e}")
    
    def create_chat_session(self, title: str, model_used: str = None, rag_enabled: bool = False) -> str:
        """Create a new chat session and return its ID"""
        session_id = str(uuid.uuid4())
        
        try:
            with self.db_manager.get_connection() as conn:
                conn.execute("""
                    INSERT INTO chat_sessions (id, title, model_used, rag_enabled) 
                    VALUES (?, ?, ?, ?)
                """, (session_id, title, model_used, rag_enabled))
                conn.commit()
                
            return session_id
            
        except Exception as e:
            print(f"Failed to create chat session: {e}")
            return None
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a message to a chat session"""
        try:
            metadata_json = json.dumps(metadata or {})
            
            with self.db_manager.get_connection() as conn:
                # Add the message
                conn.execute("""
                    INSERT INTO chat_messages (session_id, role, content, metadata) 
                    VALUES (?, ?, ?, ?)
                """, (session_id, role, content, metadata_json))
                
                # Update session message count and timestamp
                conn.execute("""
                    UPDATE chat_sessions 
                    SET message_count = message_count + 1, 
                        updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (session_id,))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Failed to add message to session {session_id}: {e}")
            return False
    
    def get_chat_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get list of chat sessions ordered by most recent"""
        try:
            result = self.db_manager.execute_query("""
                SELECT id, title, created_at, updated_at, message_count, model_used, rag_enabled
                FROM chat_sessions 
                ORDER BY updated_at DESC 
                LIMIT ?
            """, (limit,))
            
            sessions = []
            for row in result:
                sessions.append({
                    "id": row[0],
                    "title": row[1],
                    "created_at": row[2],
                    "updated_at": row[3],
                    "message_count": row[4],
                    "model_used": row[5],
                    "rag_enabled": bool(row[6])
                })
            
            return sessions
            
        except Exception as e:
            print(f"Failed to get chat sessions: {e}")
            return []
    
    def get_chat_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a chat session"""
        try:
            result = self.db_manager.execute_query("""
                SELECT role, content, timestamp, metadata
                FROM chat_messages 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """, (session_id,))
            
            messages = []
            for row in result:
                messages.append({
                    "role": row[0],
                    "content": row[1],
                    "timestamp": row[2],
                    "metadata": json.loads(row[3])
                })
            
            return messages
            
        except Exception as e:
            print(f"Failed to get messages for session {session_id}: {e}")
            return []
    
    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update the title of a chat session"""
        try:
            result = self.db_manager.execute_query("""
                UPDATE chat_sessions 
                SET title = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            """, (title, session_id))
            
            return result > 0  # Returns number of affected rows
            
        except Exception as e:
            print(f"Failed to update session title: {e}")
            return False
    
    def delete_chat_session(self, session_id: str) -> bool:
        """Delete a chat session and all its messages"""
        try:
            with self.db_manager.get_connection() as conn:
                # Delete messages first (foreign key constraint)
                conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
                
                # Delete the session
                cursor = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Failed to delete chat session {session_id}: {e}")
            return False
    
    def save_current_chat(self, messages: List[Dict[str, str]], title: str = None, 
                         model_used: str = None, rag_enabled: bool = False) -> Optional[str]:
        """Save current chat messages as a new session"""
        if not messages:
            return None
        
        # Generate title if not provided
        if not title:
            first_message = next((msg for msg in messages if msg["role"] == "user"), None)
            if first_message:
                title = first_message["content"][:50] + ("..." if len(first_message["content"]) > 50 else "")
            else:
                title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Create session
        session_id = self.create_chat_session(title, model_used, rag_enabled)
        if not session_id:
            return None
        
        # Add all messages
        for message in messages:
            if not self.add_message(session_id, message["role"], message["content"]):
                print(f"Warning: Failed to add message to session {session_id}")
        
        return session_id
    
    def get_chat_stats(self) -> Dict[str, Any]:
        """Get statistics about chat history"""
        try:
            # Get session count
            session_result = self.db_manager.execute_query("SELECT COUNT(*) FROM chat_sessions")
            session_count = session_result[0][0] if session_result else 0
            
            # Get total message count
            message_result = self.db_manager.execute_query("SELECT COUNT(*) FROM chat_messages")
            message_count = message_result[0][0] if message_result else 0
            
            # Get most recent session
            recent_result = self.db_manager.execute_query("""
                SELECT updated_at FROM chat_sessions 
                ORDER BY updated_at DESC LIMIT 1
            """)
            last_activity = recent_result[0][0] if recent_result else None
            
            return {
                "total_sessions": session_count,
                "total_messages": message_count,
                "last_activity": last_activity
            }
            
        except Exception as e:
            print(f"Failed to get chat stats: {e}")
            return {
                "total_sessions": 0,
                "total_messages": 0,
                "last_activity": None
            }
