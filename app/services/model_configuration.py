"""
Model Configuration classes for encapsulating model parameters.

This module provides parameter objects to reduce primitive obsession and
improve the maintainability of model configuration.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelConfiguration:
    """
    Parameter object for model configuration settings.
    
    This class encapsulates all model loading and inference parameters
    to reduce parameter lists and improve maintainability.
    """
    
    # Core model settings
    prediction_tokens: int = 16
    context_size: int = 128
    gpu_layers: int = 0
    
    # Sampling settings
    temperature: float = 0.0  # 0.0 means greedy sampling
    top_p: float = 0.9
    top_k: Optional[int] = None
    
    # Performance settings
    threads: Optional[int] = None
    batch_size: int = 1
    
    # Advanced settings
    rope_freq_base: Optional[float] = None
    rope_freq_scale: Optional[float] = None
    
    def to_options_string(self) -> str:
        """
        Convert configuration to options string for model loading.
        
        Returns:
            str: Formatted options string for sqlite-ai
        """
        options = [
            f"n_predict={self.prediction_tokens}",
            f"n_ctx={self.context_size}",
            f"n_gpu_layers={self.gpu_layers}",
            f"n_batch={self.batch_size}"
        ]
        
        if self.threads is not None:
            options.append(f"n_threads={self.threads}")
        
        if self.rope_freq_base is not None:
            options.append(f"rope_freq_base={self.rope_freq_base}")
            
        if self.rope_freq_scale is not None:
            options.append(f"rope_freq_scale={self.rope_freq_scale}")
        
        return ",".join(options)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "prediction_tokens": self.prediction_tokens,
            "context_size": self.context_size,
            "gpu_layers": self.gpu_layers,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "threads": self.threads,
            "batch_size": self.batch_size,
            "rope_freq_base": self.rope_freq_base,
            "rope_freq_scale": self.rope_freq_scale
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfiguration':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if v is not None})
    
    def is_greedy_sampling(self) -> bool:
        """Check if configuration uses greedy sampling."""
        return self.temperature <= 0.1
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.prediction_tokens <= 0:
            raise ValueError("prediction_tokens must be positive")
        
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")
            
        if self.gpu_layers < 0:
            raise ValueError("gpu_layers cannot be negative")
            
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
            
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
            
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive if specified")
            
        if self.threads is not None and self.threads <= 0:
            raise ValueError("threads must be positive if specified")
            
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass
class SamplingConfiguration:
    """
    Parameter object for sampling configuration.
    
    Separate from ModelConfiguration to allow runtime sampling changes
    without reloading the model.
    """
    
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: Optional[int] = None
    min_keep: int = 1
    
    def is_greedy(self) -> bool:
        """Check if this is greedy sampling configuration."""
        return self.temperature <= 0.1
    
    def validate(self) -> None:
        """Validate sampling parameters."""
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
            
        if self.top_p <= 0.0 or self.top_p > 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
            
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive if specified")
            
        if self.min_keep <= 0:
            raise ValueError("min_keep must be positive")


# Predefined configurations for common use cases
class ModelConfigurations:
    """Predefined model configurations for common scenarios."""
    
    @staticmethod
    def fast_inference() -> ModelConfiguration:
        """Configuration optimized for fast inference."""
        return ModelConfiguration(
            prediction_tokens=8,
            context_size=64,
            gpu_layers=0,
            temperature=0.0,  # Greedy sampling
            batch_size=1
        )
    
    @staticmethod
    def balanced() -> ModelConfiguration:
        """Balanced configuration for general use."""
        return ModelConfiguration(
            prediction_tokens=16,
            context_size=128,
            gpu_layers=0,
            temperature=0.0,
            batch_size=1
        )
    
    @staticmethod
    def high_quality() -> ModelConfiguration:
        """Configuration for high-quality responses."""
        return ModelConfiguration(
            prediction_tokens=32,
            context_size=256,
            gpu_layers=0,
            temperature=0.7,
            top_p=0.9,
            batch_size=1
        )
    
    @staticmethod
    def creative() -> ModelConfiguration:
        """Configuration for creative text generation."""
        return ModelConfiguration(
            prediction_tokens=64,
            context_size=512,
            gpu_layers=0,
            temperature=1.0,
            top_p=0.95,
            top_k=40,
            batch_size=1
        )
    
    @staticmethod
    def gpu_accelerated() -> ModelConfiguration:
        """Configuration for GPU acceleration."""
        return ModelConfiguration(
            prediction_tokens=32,
            context_size=256,
            gpu_layers=32,  # Adjust based on GPU memory
            temperature=0.7,
            top_p=0.9,
            batch_size=4
        )


# Predefined sampling configurations
class SamplingConfigurations:
    """Predefined sampling configurations."""
    
    @staticmethod
    def greedy() -> SamplingConfiguration:
        """Greedy sampling for deterministic output."""
        return SamplingConfiguration(temperature=0.0)
    
    @staticmethod
    def balanced() -> SamplingConfiguration:
        """Balanced sampling for good quality."""
        return SamplingConfiguration(
            temperature=0.7,
            top_p=0.9,
            min_keep=1
        )
    
    @staticmethod
    def creative() -> SamplingConfiguration:
        """Creative sampling for diverse output."""
        return SamplingConfiguration(
            temperature=1.0,
            top_p=0.95,
            top_k=40,
            min_keep=1
        )
    
    @staticmethod
    def focused() -> SamplingConfiguration:
        """Focused sampling for coherent output."""
        return SamplingConfiguration(
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            min_keep=1
        )
