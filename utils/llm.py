"""Centralized LLM configuration and utilities."""
from typing import Dict, Any, Optional
from dashscope import Generation
from dashscope.api.generation import Generation as GenerationType

class LLMConfig:
    """Configuration for LLM interactions."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "qwen-max"
    
    def call(
        self,
        messages: list,
        result_format: str = 'message'
    ) -> Optional[str]:
        """Make an LLM API call.
        
        Args:
            messages: List of message dictionaries
            result_format: Format for the result
            
        Returns:
            Generated text or None if generation fails
        """
        response = Generation.call(
            model=self.model,
            messages=messages,
            result_format=result_format,
            api_key=self.api_key
        )
        
        if isinstance(response, dict) and 'output' in response:
            output = response['output']
            if 'choices' in output:
                for choice in output['choices']:
                    if 'message' in choice and 'content' in choice['message']:
                        return choice['message']['content']
        return None
