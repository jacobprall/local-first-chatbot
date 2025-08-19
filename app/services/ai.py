from typing import Dict, Any
from .database import DatabaseManager


class AIService:
    """Simplified AI service for demonstration purposes."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        
    def get_version(self) -> str:
        """Get the sqlite-ai extension version."""
        try:
            conn = self.db_manager.get_persistent_connection()
            result = conn.execute("SELECT ai_version()").fetchone()
            return result[0] if result else "unknown"
        except Exception as e:
            print(f"Failed to get AI version: {e}")
            return "unknown"
    
    def load_model(self, model_path: str, options: str = "") -> None:
        """Load a GGUF model with optional configuration."""
        try:
            conn = self.db_manager.get_persistent_connection()
            cursor = conn.execute("SELECT llm_model_load(?, ?)", (model_path, options))
            result = cursor.fetchall()
            conn.commit()
            print(f"Model load result: {result}")
            
            # Store model info in database manager
            self.db_manager.set_model_info(model_path, options)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_path}': {e}")
    
    def generate_text(self, prompt: str, options: str = "") -> str:
        """Generate text completion for the given prompt."""
        conn = self.db_manager.get_persistent_connection()
        result = conn.execute("SELECT llm_text_generate(?, ?)", (prompt, options)).fetchone()
        return result[0] if result else ""
    
    def generate_embedding(self, text: str, options: str = "") -> bytes:
        """Generate embedding for the given text."""
        conn = self.db_manager.get_persistent_connection()
        result = conn.execute("SELECT llm_embed_generate(?, ?)", (text, options)).fetchone()
        return result[0] if result else b""
    
    def configure_sampler_greedy(self) -> None:
        """Configure sampler for deterministic output."""
        try:
            conn = self.db_manager.get_persistent_connection()
            conn.execute("SELECT llm_sampler_init_greedy()")
            conn.commit()
        except Exception as e:
            print(f"Failed to configure greedy sampling: {e}")
            raise RuntimeError(f"Cannot configure sampling method: {e}")
    
    def configure_sampler_temperature(self, temperature: float = 0.8) -> None:
        """Configure sampling temperature for more creative output."""
        try:
            # Ensure valid temperature range
            temperature = max(0.1, min(2.0, temperature))
            conn = self.db_manager.get_persistent_connection()
            conn.execute("SELECT llm_sampler_init_temp(?)", (temperature,))
            conn.commit()
        except Exception as e:
            print(f"Failed to configure temperature: {e}")
            # Fallback to greedy
            self.configure_sampler_greedy()
    
    def configure_sampler_top_p(self, p: float = 0.9, min_keep: int = 1) -> None:
        """Configure nucleus (top-p) sampling for balanced creativity."""
        try:
            # Ensure valid parameters
            p = max(0.1, min(0.99, p))
            min_keep = max(1, min_keep)
            conn = self.db_manager.get_persistent_connection()
            conn.execute("SELECT llm_sampler_init_top_p(?, ?)", (p, min_keep))
            conn.commit()
        except Exception as e:
            print(f"Failed to configure top-p sampling: {e}")
            # Fallback to greedy
            self.configure_sampler_greedy()
    
    def configure_sampler_top_k(self, k: int = 40) -> None:
        """Configure top-k sampling for controlled randomness."""
        conn = self.db_manager.get_persistent_connection()
        conn.execute("SELECT llm_sampler_init_top_k(?)", (k,))
        conn.commit()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata for display purposes."""
        info_queries = [
            ("n_params", "SELECT llm_model_n_params()"),
            ("size", "SELECT llm_model_size()"),
            ("n_ctx_train", "SELECT llm_model_n_ctx_train()"),
            ("n_embd", "SELECT llm_model_n_embd()"),
            ("n_layer", "SELECT llm_model_n_layer()"),
            ("n_head", "SELECT llm_model_n_head()"),
            ("description", "SELECT llm_model_desc()"),
        ]
        
        model_info = {}
        conn = self.db_manager.get_persistent_connection()
        for key, query in info_queries:
            try:
                result = conn.execute(query).fetchone()
                model_info[key] = result[0] if result else None
            except Exception:
                model_info[key] = None
                
        return model_info