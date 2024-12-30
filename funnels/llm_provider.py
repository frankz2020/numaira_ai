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
                model="claude-3-opus-20240229",
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
                model="claude-3-opus-20240229",
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
    
    def _call_qwen(self, prompt: str, max_tokens: int = 100) -> str:
        payload = {
            "model": "qwen-max",
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
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if 'output' in result and 'text' in result['output']:
                return result['output']['text'].strip()
            raise RuntimeError("Unexpected API response format")
            
        except Exception as e:
            raise RuntimeError(f"Error calling Qwen API: {str(e)}")
    
    def batch_check_metrics(self, sentence: str, target_metrics: List[str]) -> List[str]:
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
        response = self._call_qwen(prompt)
        response = response.lower()
        
        if response == "none":
            return []
            
        return [m.strip() for m in response.split(",")]
    
    def get_updated_numbers(self, sentence: str, target_metrics: List[str], target_values: List[str]) -> Dict[str, Tuple[str, str]]:
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
        response = self._call_qwen(prompt, max_tokens=200)
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

def get_llm_provider(provider_name: str = None) -> LLMProvider:
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