"""
Centralized error recovery with multiple fallback strategies.

Provides robust error handling through configurable recovery strategies,
improving application resilience and user experience.
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, Optional, List, Dict
from enum import Enum
import traceback


class ErrorSeverity(Enum):
    """Enumeration of error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(ABC):
    """Abstract base class for recovery strategies."""
    
    @abstractmethod
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this strategy can handle the given error."""
        pass
    
    @abstractmethod
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Attempt to recover from the error."""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this recovery strategy."""
        pass


class SimpleRetryStrategy(RecoveryStrategy):
    """Recovery strategy that simply retries the operation."""
    
    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Can handle most exceptions except critical ones."""
        return not isinstance(error, (RuntimeError, FileNotFoundError))
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Retry the original operation."""
        original_func = context.get("original_function")
        original_args = context.get("original_args", ())
        original_kwargs = context.get("original_kwargs", {})
        
        if not original_func:
            raise error
        
        for attempt in range(self.max_retries):
            try:
                print(f"   🔄 Retry attempt {attempt + 1}/{self.max_retries}")
                return original_func(*original_args, **original_kwargs)
            except Exception as retry_error:
                if attempt == self.max_retries - 1:
                    raise retry_error
                continue
    
    def get_strategy_name(self) -> str:
        return f"simple_retry_{self.max_retries}"


class FallbackPromptStrategy(RecoveryStrategy):
    """Recovery strategy for text generation failures using simpler prompts."""
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle text generation related errors."""
        return context.get("operation_type") == "text_generation"
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Try with a simpler prompt format."""
        ai_service = context.get("ai_service")
        message = context.get("message")
        reset_function = context.get("reset_function")
        
        if not all([ai_service, message]):
            raise error
        
        try:
            print(f"   🔄 Attempting recovery with simpler prompt...")
            if reset_function:
                reset_function()
            
            # Use simpler Q&A format
            simple_prompt = f"Q: {message}\nA:"
            response = ai_service.generate_text(simple_prompt, "n_predict=8")
            
            # Clean response
            if "Q:" in response:
                response = response.split("Q:")[0]
            
            return response.strip()
            
        except Exception as recovery_error:
            print(f"   ❌ Recovery attempt failed: {recovery_error}")
            # Final fallback: echo the message
            return f"I received your message: {message}"
    
    def get_strategy_name(self) -> str:
        return "fallback_prompt"


class GracefulDegradationStrategy(RecoveryStrategy):
    """Recovery strategy that provides graceful degradation of functionality."""
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Can handle most errors by providing degraded functionality."""
        return True
    
    def recover(self, error: Exception, context: Dict[str, Any]) -> Any:
        """Provide a graceful degradation response."""
        operation_type = context.get("operation_type", "unknown")
        
        if operation_type == "text_generation":
            message = context.get("message", "")
            return f"I apologize, but I'm having trouble processing your request: '{message[:50]}...'. Please try again."
        
        elif operation_type == "search":
            return []
        
        elif operation_type == "embedding":
            return b""
        
        else:
            return None
    
    def get_strategy_name(self) -> str:
        return "graceful_degradation"


class ErrorRecoveryHandler:
    """
    Centralized error recovery handler that manages fallback strategies.
    
    This class provides a clean interface for handling errors with
    multiple recovery strategies and consistent logging.
    """
    
    def __init__(self):
        self.strategies: List[RecoveryStrategy] = []
        self.error_log: List[Dict[str, Any]] = []
    
    def add_strategy(self, strategy: RecoveryStrategy) -> None:
        """Add a recovery strategy to the handler."""
        self.strategies.append(strategy)
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a recovery strategy by name."""
        for i, strategy in enumerate(self.strategies):
            if strategy.get_strategy_name() == strategy_name:
                del self.strategies[i]
                return True
        return False
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Any:
        """
        Handle an error using available recovery strategies.
        
        Args:
            error: The exception that occurred
            context: Context information for recovery
            
        Returns:
            Recovery result or re-raises the error if no recovery possible
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": self._get_timestamp(),
            "recovery_attempted": False,
            "recovery_successful": False,
            "recovery_strategy": None
        }
        
        try:
            # Try each strategy in order
            for strategy in self.strategies:
                if strategy.can_handle(error, context):
                    try:
                        print(f"   🔧 Attempting recovery with strategy: {strategy.get_strategy_name()}")
                        result = strategy.recover(error, context)
                        
                        error_info.update({
                            "recovery_attempted": True,
                            "recovery_successful": True,
                            "recovery_strategy": strategy.get_strategy_name()
                        })
                        
                        print(f"   ✅ Recovery successful with {strategy.get_strategy_name()}")
                        self.error_log.append(error_info)
                        return result
                        
                    except Exception as recovery_error:
                        print(f"   ❌ Recovery strategy {strategy.get_strategy_name()} failed: {recovery_error}")
                        continue
            
            # No strategy could handle the error
            error_info["recovery_attempted"] = True
            self.error_log.append(error_info)
            print(f"   ❌ No recovery strategy could handle the error")
            raise error
            
        except Exception as e:
            self.error_log.append(error_info)
            raise e
    
    def execute_with_recovery(self, func: Callable, *args, 
                            operation_type: str = "unknown", **kwargs) -> Any:
        """
        Execute a function with automatic error recovery.
        
        Args:
            func: Function to execute
            *args: Function arguments
            operation_type: Type of operation for context
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or recovery result
        """
        context = {
            "operation_type": operation_type,
            "original_function": func,
            "original_args": args,
            "original_kwargs": kwargs
        }
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return self.handle_error(e, context)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get statistics about handled errors."""
        total_errors = len(self.error_log)
        if total_errors == 0:
            return {"total_errors": 0}
        
        successful_recoveries = sum(1 for log in self.error_log if log["recovery_successful"])
        
        error_types = {}
        recovery_strategies = {}
        
        for log in self.error_log:
            error_type = log["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if log["recovery_strategy"]:
                strategy = log["recovery_strategy"]
                recovery_strategies[strategy] = recovery_strategies.get(strategy, 0) + 1
        
        return {
            "total_errors": total_errors,
            "successful_recoveries": successful_recoveries,
            "recovery_rate": successful_recoveries / total_errors if total_errors > 0 else 0,
            "error_types": error_types,
            "recovery_strategies_used": recovery_strategies
        }
    
    def clear_error_log(self) -> None:
        """Clear the error log."""
        self.error_log.clear()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.now().isoformat()


# Predefined error recovery configurations
class ErrorRecoveryConfigurations:
    """Factory for creating common error recovery configurations."""
    
    @staticmethod
    def basic() -> ErrorRecoveryHandler:
        """Create a basic error recovery handler."""
        handler = ErrorRecoveryHandler()
        handler.add_strategy(SimpleRetryStrategy(max_retries=1))
        handler.add_strategy(GracefulDegradationStrategy())
        return handler
    
    @staticmethod
    def robust() -> ErrorRecoveryHandler:
        """Create a robust error recovery handler with multiple strategies."""
        handler = ErrorRecoveryHandler()
        handler.add_strategy(SimpleRetryStrategy(max_retries=2))
        handler.add_strategy(FallbackPromptStrategy())
        handler.add_strategy(GracefulDegradationStrategy())
        return handler
    
    @staticmethod
    def text_generation_focused() -> ErrorRecoveryHandler:
        """Create an error recovery handler optimized for text generation."""
        handler = ErrorRecoveryHandler()
        handler.add_strategy(FallbackPromptStrategy())
        handler.add_strategy(SimpleRetryStrategy(max_retries=1))
        handler.add_strategy(GracefulDegradationStrategy())
        return handler
