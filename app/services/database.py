import sqlite3
import os
import platform
import threading
from contextlib import contextmanager
from typing import Generator, Optional


class DatabaseManager:
    """Database manager that handles SQLite connections and extensions"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_directory()
        self._extension_suffix = self._get_extension_suffix()
        self._model_loaded = False
        self._model_path = None
        self._model_options = None
        self._persistent_conn = None
        self._conn_lock = threading.RLock()  # Reentrant lock for thread safety
        self._initialize_connection()
    
    def _get_extension_suffix(self) -> str:
        """Get the correct file extension for SQLite extensions based on platform"""
        system = platform.system().lower()
        if system == "darwin":
            return ".dylib"
        elif system == "windows":
            return ".dll"
        else:  # Linux and other Unix-like systems
            return ".so"
    
    def _ensure_directory(self):
        """Ensure the database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    def _initialize_connection(self):
        """Initialize the persistent connection with extensions loaded"""
        try:
            self._persistent_conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            
            # Load extensions
            self._persistent_conn.enable_load_extension(True)
            
            # Load AI extension
            ai_ext_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                "extensions", 
                f"ai{self._extension_suffix}"
            )
            if os.path.exists(ai_ext_path):
                self._persistent_conn.load_extension(ai_ext_path)
                print(f"Loaded AI extension: {ai_ext_path}")
            else:
                print(f"AI extension not found: {ai_ext_path}")
            
            # Load vector extension
            vector_ext_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                "extensions", 
                f"vector{self._extension_suffix}"
            )
            if os.path.exists(vector_ext_path):
                self._persistent_conn.load_extension(vector_ext_path)
                print(f"Loaded vector extension: {vector_ext_path}")
            else:
                print(f"Vector extension not found: {vector_ext_path}")
            
            self._persistent_conn.enable_load_extension(False)
            
        except Exception as e:
            print(f"Warning: Failed to initialize connection with extensions: {e}")
            # Create basic connection without extensions as fallback
            self._persistent_conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get the persistent database connection with thread safety and transaction handling"""
        with self._conn_lock:
            if self._persistent_conn is None:
                self._initialize_connection()
            
            # Reload model if one was previously loaded and not already loaded on this connection
            if self._model_loaded and self._model_path:
                try:
                    # Check if model is already loaded by trying to use it
                    test_result = self._persistent_conn.execute("SELECT llm_model_loaded()").fetchone()
                    if not test_result or not test_result[0]:
                        self._persistent_conn.execute("SELECT llm_model_load(?, ?)", (self._model_path, self._model_options or ""))
                        self._persistent_conn.commit()
                except Exception as e:
                    # If llm_model_loaded doesn't exist, just try to load the model
                    try:
                        self._persistent_conn.execute("SELECT llm_model_load(?, ?)", (self._model_path, self._model_options or ""))
                        self._persistent_conn.commit()
                    except Exception as load_e:
                        print(f"Warning: Failed to reload model on connection: {load_e}")
            
            try:
                yield self._persistent_conn
            except Exception as e:
                try:
                    self._persistent_conn.rollback()
                except:
                    pass  # Ignore rollback errors
                raise e
    
    def set_model_info(self, model_path: str, options: str = "") -> None:
        """Store model information for reloading on new connections"""
        self._model_path = model_path
        self._model_options = options
        self._model_loaded = True
    
    def get_persistent_connection(self):
        """Get the persistent connection (same as get_connection context manager but without context management)"""
        with self._conn_lock:
            if self._persistent_conn is None:
                self._initialize_connection()
            
            # Reload model if one was previously loaded and not already loaded on this connection
            if self._model_loaded and self._model_path:
                try:
                    # Check if model is already loaded by trying to use it
                    test_result = self._persistent_conn.execute("SELECT llm_model_loaded()").fetchone()
                    if not test_result or not test_result[0]:
                        self._persistent_conn.execute("SELECT llm_model_load(?, ?)", (self._model_path, self._model_options or ""))
                        self._persistent_conn.commit()
                except Exception as e:
                    # If llm_model_loaded doesn't exist, just try to load the model
                    try:
                        self._persistent_conn.execute("SELECT llm_model_load(?, ?)", (self._model_path, self._model_options or ""))
                        self._persistent_conn.commit()
                    except Exception as load_e:
                        print(f"Warning: Failed to reload model on connection: {load_e}")
            
            return self._persistent_conn
    
    def execute_query(self, query: str, params: Optional[tuple] = None):
        """Execute a query and return results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                return cursor.fetchall()
            else:
                conn.commit()
                return cursor.rowcount
    
    def close(self):
        """Close the persistent connection"""
        with self._conn_lock:
            if self._persistent_conn:
                try:
                    self._persistent_conn.close()
                except Exception as e:
                    print(f"Warning: Error closing database connection: {e}")
                finally:
                    self._persistent_conn = None
    
    def __del__(self):
        """Cleanup when the object is destroyed"""
        self.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
