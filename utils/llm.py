"""Centralized LLM configuration and utilities."""
from typing import Dict, Any, Optional
from dashscope import Generation

class LLMConfig:
    """Configuration for LLM interactions."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = "qwen-max"
    
    def call(
        self,
        messages: list,
        result_format: str = 'message',
        timeout: int = 30
    ) -> Optional[str]:
        """Make an LLM API call with timeout.
        
        Args:
            messages: List of message dictionaries
            result_format: Format for the result
            timeout: Timeout in seconds (default: 30)
            
        Returns:
            Generated text or None if generation fails
        """
        try:
            response = Generation.call(
                model=self.model,
                messages=messages,
                result_format=result_format,
                api_key=self.api_key,
                timeout=timeout
            )
        
            if isinstance(response, dict) and 'output' in response:
                output = response['output']
                if 'choices' in output:
                    for choice in output['choices']:
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content'].strip()
                            if content and content != '[]':
                                return content
            return None
        except Exception as e:
            print(f"Error in LLM call: {str(e)}")
            return None
