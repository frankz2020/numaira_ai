from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import os
from dotenv import load_dotenv
from anthropic import Anthropic
import requests
import json

load_dotenv()

class LLMProvider(ABC):
    @abstractmethod
    def batch_check_metrics(self, sentence: str, target_metrics: List[str]) -> List[str]:
        """Check which metrics from the list match the sentence."""
        pass
        
    @abstractmethod
    def get_updated_numbers(self, sentence: str, target_metrics: List[str], target_values: List[str]) -> Dict[str, Tuple[str, str]]:
        """Get updated numbers for multiple metrics in one call."""
        pass

class ClaudeProvider(LLMProvider):
    def __init__(self):
        api_key = os.getenv('CLAUDE_API_KEY')
        if not api_key:
            raise ValueError("CLAUDE_API_KEY environment variable not found")
        self.client = Anthropic(api_key=api_key)
    
    def batch_check_metrics(self, sentence: str, target_metrics: List[str]) -> List[str]:
        prompt = f"""You are a financial expert. Given a sentence and a list of target financial metrics:

Sentence: {sentence}

Target Metrics:
{chr(10).join(f"- {metric}" for metric in target_metrics)}

Task: Return a list of metrics that this sentence is reporting.
Consider:
- Exact matches (e.g., "total revenue" matches "total revenue")
- Equivalent terms (e.g., "net income attributable to common stockholders" matches "net income to stockholders")
- Hierarchical relationships (e.g., "automotive revenue" is a subset of "total revenue")

Output only the matching metrics as a comma-separated list. No other text.
Example: total revenue, net income
If no matches, output: none
"""
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=100,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text_response = response.content[0].text.strip().lower()
            if text_response == "none":
                return []
                
            return [m.strip() for m in text_response.split(",")]
            
        except Exception as e:
            raise RuntimeError(f"Error calling Claude API: {str(e)}")
    
    def get_updated_numbers(self, sentence: str, target_metrics: List[str], target_values: List[str]) -> Dict[str, Tuple[str, str]]:
        metrics_info = "\n".join(f"Metric {i+1}: {metric}\nTarget Values: {value}" 
                               for i, (metric, value) in enumerate(zip(target_metrics, target_values)))
        
        prompt = f"""You are a financial expert. Given a sentence and multiple target metrics with their values:

Sentence: {sentence}

{metrics_info}

For each metric that matches the sentence, extract the three month and six month values that should be used to update the sentence.
Output each result on a new line in the format:
metric_name: three_month_value,six_month_value

Example:
total revenue: 24.93,48.26
net income: 2.61,5.15

Only include metrics that match. Use 'none,none' for metrics that don't match.
All numbers should be in billions with 2 decimal places.
"""
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            results = {}
            text_response = response.content[0].text.strip()
            
            for line in text_response.split("\n"):
                if ":" not in line:
                    continue
                    
                metric, values = line.split(":", 1)
                metric = metric.strip().lower()
                values = values.strip()
                
                if values == "none,none":
                    continue
                    
                try:
                    three_month, six_month = values.split(",")
                    three_month = float(three_month)
                    six_month = float(six_month)
                    results[metric] = (f"{three_month:.2f} billion", f"{six_month:.2f} billion")
                except (ValueError, IndexError):
                    continue
            
            return results
            
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
        import logging
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
    
    def batch_check_metrics(self, sentence: str, target_metrics: List[str], timeout: int = 30) -> List[str]:
        """Check which metrics from the list match the sentence.
        
        Args:
            sentence: Text to analyze
            target_metrics: List of metrics to check
            timeout: Timeout in seconds (default: 30)
            
        Returns:
            List of matching metrics
        """
        prompt = f"""Given a sentence and a list of target financial metrics:

Sentence: {sentence}

Target Metrics:
{chr(10).join(f"- {metric}" for metric in target_metrics)}

Task: Return a list of metrics that this sentence is reporting.
Consider:
- Exact matches (e.g., "total revenue" matches "total revenue")
- Equivalent terms (e.g., "net income attributable to common stockholders" matches "net income to stockholders")
- Hierarchical relationships (e.g., "automotive revenue" is a subset of "total revenue")

Output only the matching metrics as a comma-separated list. No other text.
Example: total revenue, net income
If no matches, output: none
"""
        response = self._call_qwen(prompt, timeout=timeout)
        if not response or response == "none":
            return []
            
        return [m.strip() for m in response.lower().split(",")]
    
    def get_updated_numbers(
        self,
        sentence: str,
        target_metrics: List[str],
        target_values: List[str],
        timeout: int = 30
    ) -> Dict[str, Tuple[str, str]]:
        """Get updated numbers for multiple metrics in one call.
        
        Args:
            sentence: Text to analyze
            target_metrics: List of metrics to check
            target_values: List of target values
            timeout: Timeout in seconds (default: 30)
            
        Returns:
            Dictionary mapping metrics to (three_month, six_month) value pairs
        """
        metrics_info = "\n".join(f"Metric {i+1}: {metric}\nTarget Values: {value}" 
                               for i, (metric, value) in enumerate(zip(target_metrics, target_values)))
        
        prompt = f"""Given a sentence and multiple target metrics with their values:

Sentence: {sentence}

{metrics_info}

For each metric that matches the sentence, extract the three month and six month values that should be used to update the sentence.
Output each result on a new line in the format:
metric_name: three_month_value,six_month_value

Example:
total revenue: 24.93,48.26
net income: 2.61,5.15

Only include metrics that match. Use 'none,none' for metrics that don't match.
All numbers should be in billions with 2 decimal places.
"""
        response = self._call_qwen(prompt, max_tokens=200, timeout=timeout)
        if not response:
            return {}
            
        results = {}
        
        for line in response.split("\n"):
            if ":" not in line:
                continue
                
            metric, values = line.split(":", 1)
            metric = metric.strip().lower()
            values = values.strip()
            
            if values == "none,none":
                continue
                
            try:
                three_month, six_month = values.split(",")
                three_month = float(three_month)
                six_month = float(six_month)
                results[metric] = (f"{three_month:.2f} billion", f"{six_month:.2f} billion")
            except (ValueError, IndexError):
                continue
        
        return results

def get_llm_provider(provider_name: str | None = None) -> LLMProvider:
    """Factory function to get the configured LLM provider."""
    # Get provider from environment if not specified
    if not provider_name:
        provider_name = os.getenv('LLM_PROVIDER', 'claude').lower()
    
    if provider_name == 'claude':
        return ClaudeProvider()
    elif provider_name == 'qwen':
        return QwenProvider()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")               