"""
Centralized model state management using the Singleton pattern.

Manages model loading, configuration, and lifecycle to ensure consistent
state across the application and prevent duplicate model loading.
"""

import os
from typing import Optional, Dict, Any
from .ai import AIService
from .model_configuration import ModelConfiguration, ModelConfigurations


class ModelManager:
    """
    Singleton for centralized model state management.

    Ensures only one model is loaded at a time and provides consistent
    access to model state across the application.
    """
    
    _instance: Optional['ModelManager'] = None
    _model_loaded: bool = False
    _model_path: Optional[str] = None
    _model_config: Optional[ModelConfiguration] = None
    _ai_service: Optional[AIService] = None
    
    def __new__(cls) -> 'ModelManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the model manager (singleton pattern)."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
    
    def initialize_ai_service(self, ai_service: AIService) -> None:
        """Initialize the AI service reference."""
        self._ai_service = ai_service
    
    def load_model(self, model_path: str, config: Optional[ModelConfiguration] = None, options: str = "") -> bool:
        """
        Load a GGUF model for inference.

        Args:
            model_path: Path to the GGUF model file
            config: Model configuration object (optional)
            options: Legacy options string (deprecated, use config instead)

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self._model_loaded and self._model_path == model_path:
            return True  # Already loaded

        if not self._ai_service:
            raise RuntimeError("AI service not initialized. Call initialize_ai_service() first.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Configuration resolution: prefer config object over legacy options string
        if config is None:
            if options:
                # Legacy support: use provided options string with default config
                config = ModelConfigurations.balanced()
                final_options = options
            else:
                # No config provided: use balanced defaults
                config = ModelConfigurations.balanced()
                final_options = config.to_options_string()
        else:
            # Use provided config after validation
            config.validate()
            final_options = config.to_options_string()

        try:
            print(f"Loading model: {model_path}")
            print(f"Model configuration: {config.to_dict()}")
            print(f"Model options: {final_options}")

            self._ai_service.load_model(model_path, final_options)

            # Configure sampling strategy based on temperature setting
            # Greedy sampling (temp ≤ 0.1) for deterministic output
            # Temperature sampling for creative/varied responses
            if config.is_greedy_sampling():
                print("Configuring greedy sampling...")
                self._ai_service.configure_sampler_greedy()
            else:
                print(f"Configuring sampling: temp={config.temperature}, top_p={config.top_p}")
                self._ai_service.configure_sampler_temperature(config.temperature)
                if config.top_p > 0:
                    self._ai_service.configure_sampler_top_p(config.top_p, 1)
                if config.top_k:
                    self._ai_service.configure_sampler_top_k(config.top_k)

            self._model_loaded = True
            self._model_path = model_path
            self._model_config = config
            print("Model loaded successfully!")
            return True

        except Exception as e:
            print(f"Failed to load model: {e}")
            self._model_loaded = False
            self._model_path = None
            self._model_config = None
            return False
    
    def unload_model(self) -> None:
        """Unload the current model."""
        self._model_loaded = False
        self._model_path = None
        self._model_config = None
        print("Model unloaded")
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model_loaded
    
    def get_model_path(self) -> Optional[str]:
        """Get the path of the currently loaded model."""
        return self._model_path

    def get_model_config(self) -> Optional[ModelConfiguration]:
        """Get the configuration of the currently loaded model."""
        return self._model_config
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dict containing model status and information
        """
        if not self._model_loaded:
            return {"status": "No model loaded", "model_path": None}
        
        try:
            if self._ai_service:
                info = self._ai_service.get_model_info()
                info.update({
                    "status": "Model loaded",
                    "model_path": self._model_path,
                    "model_config": self._model_config.to_dict() if self._model_config else None
                })
                return info
            else:
                return {
                    "status": "Model loaded (AI service not available)",
                    "model_path": self._model_path
                }
        except Exception as e:
            return {
                "status": f"Model error: {e}",
                "model_path": self._model_path
            }
    
    def validate_model_loaded(self) -> None:
        """
        Validate that a model is loaded, raise exception if not.
        
        Raises:
            RuntimeError: If no model is loaded
        """
        if not self._model_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")
    
    def reset(self) -> None:
        """Reset the model manager state (useful for testing)."""
        self._model_loaded = False
        self._model_path = None
        self._model_config = None
        self._ai_service = None
        print("Model manager state reset")


# Global model manager instance
model_manager = ModelManager()
