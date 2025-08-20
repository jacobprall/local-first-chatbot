"""
Model Manager for handling model state and lifecycle.

This module provides centralized model state management following the Singleton pattern
to ensure consistent model state across the application.
"""

import os
from typing import Optional, Dict, Any
from .ai import AIService


class ModelManager:
    """
    Singleton class for managing model state and lifecycle.
    
    This class centralizes model loading, state tracking, and configuration
    to follow the Single Responsibility Principle and avoid scattered state management.
    """
    
    _instance: Optional['ModelManager'] = None
    _model_loaded: bool = False
    _model_path: Optional[str] = None
    _ai_service: Optional[AIService] = None
    
    def __new__(cls) -> 'ModelManager':
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the model manager."""
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._initialized = True
    
    def initialize_ai_service(self, ai_service: AIService) -> None:
        """Initialize the AI service reference."""
        self._ai_service = ai_service
    
    def load_model(self, model_path: str, options: str = "") -> bool:
        """
        Load a GGUF model for inference.
        
        Args:
            model_path: Path to the GGUF model file
            options: Model loading options string
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        if self._model_loaded and self._model_path == model_path:
            return True  # Already loaded
        
        if not self._ai_service:
            raise RuntimeError("AI service not initialized. Call initialize_ai_service() first.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            print(f"Loading model: {model_path}")
            print(f"Model options: {options}")
            
            self._ai_service.load_model(model_path, options)
            
            # Configure greedy sampling by default
            print("Configuring greedy sampling...")
            self._ai_service.configure_sampler_greedy()
            
            self._model_loaded = True
            self._model_path = model_path
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self._model_loaded = False
            self._model_path = None
            return False
    
    def unload_model(self) -> None:
        """Unload the current model."""
        self._model_loaded = False
        self._model_path = None
        print("Model unloaded")
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model_loaded
    
    def get_model_path(self) -> Optional[str]:
        """Get the path of the currently loaded model."""
        return self._model_path
    
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
                    "model_path": self._model_path
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
        self._ai_service = None
        print("Model manager state reset")


# Global model manager instance
model_manager = ModelManager()
