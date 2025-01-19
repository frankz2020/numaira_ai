from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
import os
from dotenv import load_dotenv
from anthropic import Anthropic
import requests
import json
import logging
from config import Config

load_dotenv()

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def analyze_text(self, text: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Analyze text using LLM.
        
        Args:
            text: Text to analyze
            timeout: Optional timeout in seconds
            
        Returns:
            Dict with analysis results
        """
        pass
        
    @abstractmethod
    def compare_text(self, text1: str, text2: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Compare two pieces of text.
        
        Args:
            text1: First text
            text2: Second text
            timeout: Optional timeout in seconds
            
        Returns:
            Dict with comparison results including confidence score
        """
        pass
        
    @abstractmethod
    def update_text(self, text: str, metric: str, values: Dict[str, float], timeout: Optional[int] = None) -> str:
        """Update text with new values.
        
        Args:
            text: Original text
            metric: Metric to update
            values: New values to insert
            timeout: Optional timeout in seconds
            
        Returns:
            Updated text
        """
        pass

class ClaudeProvider(LLMProvider):
    def __init__(self):
        api_key = os.getenv('CLAUDE_API_KEY')
        if not api_key:
            raise ValueError("CLAUDE_API_KEY environment variable not found")
        self.client = Anthropic(api_key=api_key)
    
    def analyze_text(self, text: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Analyze text using Claude."""
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": text
                }]
            )
            return {"analysis": response.content[0].text}
        except Exception as e:
            raise RuntimeError(f"Error calling Claude API: {str(e)}")
            
    def compare_text(self, text1: str, text2: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Compare two pieces of text using Claude."""
        prompt = f"""Compare these two pieces of text semantically:

Text 1: {text1}
Text 2: {text2}

Are they referring to the same thing? Output format:
{{"match": true/false, "confidence": 0.0-1.0}}"""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Error calling Claude API: {str(e)}")
            
    def update_text(self, text: str, metric: str, values: Dict[str, float], timeout: Optional[int] = None) -> str:
        """Update text with new values using Claude."""
        values_str = ", ".join(f"{k}: {v}" for k, v in values.items())
        prompt = f"""Update this text by replacing the values for the metric "{metric}" with these new values:
{values_str}

Text: {text}

Output only the updated text."""

        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise RuntimeError(f"Error calling Claude API: {str(e)}")

class QwenProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv('QWEN_API_KEY')
        if not self.api_key:
            raise ValueError("QWEN_API_KEY environment variable not found")
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _call_qwen(self, prompt: str, max_tokens: int = 100, timeout: int = 30) -> str:
        """Call Qwen API with proper error handling and logging.
        
        Args:
            prompt: Prompt text to send
            max_tokens: Maximum tokens to generate
            timeout: Timeout in seconds (default: 30)
            
        Returns:
            Generated text or "none" on error/timeout
        """
        logger = logging.getLogger(__name__)
        
        payload = {
            "model": "qwen2.5-72b-instruct",  # Using specified model version
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a rigorous financial analyst."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": 0,
                "result_format": "text"
            }
        }
        
        try:
            logger.debug(f"Calling Qwen API with prompt: {prompt[:100]}...")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Received API response: {result}")
            
            if 'output' in result and 'text' in result['output']:
                text = result['output']['text'].strip()
                # Clean up the response by removing [] and extra whitespace
                text = text.replace('[]', '').strip()
                if text:
                    logger.debug(f"Processed response text: {text}")
                    return text
            
            logger.warning("Empty or invalid response from API")
            return "none"
            
        except requests.Timeout:
            logger.warning(f"Timeout ({timeout}s) exceeded calling Qwen API")
            return "none"
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error calling Qwen API: {str(e)}")
            return "none"
        except ValueError as e:
            logger.error(f"JSON decode error in Qwen API response: {str(e)}")
            return "none"
        except Exception as e:
            logger.error(f"Unexpected error calling Qwen API: {str(e)}")
            return "none"
    
    def analyze_text(self, text: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Analyze text using Qwen."""
        response = self._call_qwen(text, max_tokens=200, timeout=timeout)
        return {"analysis": response} if response else {}
        
    def compare_text(self, text1: str, text2: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Compare two pieces of text using Qwen."""
        prompt = f"""Compare these two pieces of text semantically:

Text 1: {text1}
Text 2: {text2}

Are they referring to the same thing? Output format:
{{"match": true/false, "confidence": 0.0-1.0}}"""

        response = self._call_qwen(prompt, max_tokens=200, timeout=timeout)
        return response if response else {"match": False, "confidence": 0.0}
        
    def update_text(self, text: str, metric: str, values: Dict[str, float], timeout: Optional[int] = None) -> str:
        """Update text with new values using Qwen."""
        values_str = ", ".join(f"{k}: {v}" for k, v in values.items())
        prompt = f"""Update this text by replacing the values for the metric "{metric}" with these new values:
{values_str}

Text: {text}

Output only the updated text."""

        response = self._call_qwen(prompt, max_tokens=200, timeout=timeout)
        return response if response else text

class LLMProvider:
    def __init__(self, provider: str = None):
        config = Config()
        llm_config = config.get_model_config()["llm"]
        self.provider = provider or llm_config["provider"]
        self.model = llm_config["model"]
        self.default_timeout = llm_config["timeout"]
        
        if self.provider == "claude":
            self._provider = ClaudeProvider()
        elif self.provider == "qwen":
            self._provider = QwenProvider()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def analyze_text(self, text: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Analyze text using the configured provider."""
        if timeout is None:
            timeout = self.default_timeout
        return self._provider.analyze_text(text, timeout)
    
    def compare_text(self, text1: str, text2: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Compare two pieces of text using the configured provider."""
        if timeout is None:
            timeout = self.default_timeout
        return self._provider.compare_text(text1, text2, timeout)
    
    def update_text(self, text: str, metric: str, values: Dict[str, float], timeout: Optional[int] = None) -> str:
        """Update text with new values using the configured provider."""
        if timeout is None:
            timeout = self.default_timeout
        return self._provider.update_text(text, metric, values, timeout)
    
    def batch_check_metrics(
        self,
        sentence: str,
        target_metrics: List[str],
        timeout: int = None
    ) -> List[str]:
        """Check if target metrics are present in the sentence.
        
        Args:
            sentence: The sentence to check
            target_metrics: List of metric names to look for
            timeout: Optional timeout in seconds (defaults to config value)
            
        Returns:
            List of metrics found in the sentence
        """
        if timeout is None:
            timeout = self.default_timeout
            
        prompt = f"""Check which metrics from this list are present in the sentence:

Sentence: {sentence}

Target Metrics:
{chr(10).join(f"- {metric}" for metric in target_metrics)}

Output only the matching metrics as a comma-separated list. If no matches, output: none"""

        try:
            result = self._provider.analyze_text(prompt, timeout)
            text_response = result.get("analysis", "none").strip().lower()
            
            if text_response == "none":
                return []
                
            return [m.strip() for m in text_response.split(",")]
            
        except Exception as e:
            logger.error(f"Error in batch_check_metrics: {str(e)}")
            return []

def get_llm_provider(provider: str = None) -> LLMProvider:
    """Get an LLM provider instance.
    
    Args:
        provider: Optional provider name (defaults to config value)
        
    Returns:
        LLMProvider instance
    """
    return LLMProvider(provider)                  