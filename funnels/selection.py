from funnels.document_processing import read_docx
import re
from typing import Dict, List, Tuple, Set
import logging
from anthropic import Anthropic
import os
from dotenv import load_dotenv
from tqdm import tqdm
import sys

# Load environment variables
load_dotenv()

# Configure logging - suppress all loggers except errors
logging.getLogger().setLevel(logging.WARNING)  # Root logger
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("funnels").setLevel(logging.WARNING)
logging.getLogger("RAG").setLevel(logging.WARNING)

# Only show our specific formatted output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a custom formatter that only shows the message
class MinimalFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            return record.getMessage()
        return f"{record.levelname}: {record.getMessage()}"

# Create console handler with minimal output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(MinimalFormatter())
logger.addHandler(console_handler)

# Disable tqdm.write() output when not in progress bar context
tqdm.write = lambda x, file=None: None

class SemanticMatcher:
    def __init__(self):
        """Initialize the matcher with Claude."""
        api_key = os.getenv('CLAUDE_API_KEY')
        if not api_key:
            raise ValueError("CLAUDE_API_KEY environment variable not found")
        self.client = Anthropic(api_key=api_key)
    
    def batch_check_metrics(self, sentence: str, target_metrics: List[str]) -> List[str]:
        """Check which metrics from the list match the sentence."""
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
            logger.error(f"Error calling Claude API: {str(e)}")
            return []
    
    def get_updated_numbers(self, sentence: str, target_metrics: List[str], target_values: List[str]) -> Dict[str, Tuple[str, str]]:
        """Get updated numbers for multiple metrics in one call."""
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
                    logger.error(f"Failed to parse numbers for metric {metric}: {values}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}")
            return {}

def clean_text(text: str) -> str:
    """Clean text by removing special characters and extra whitespace."""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return ' '.join(text.split())

def extract_date_info(text: str) -> List[Tuple[str, str, str]]:
    """Extract period, month, and year from text."""
    text = text.lower()
    
    # Handle combined periods (e.g., "three and six months ended June 30, 2023")
    periods = []
    
    # Try to match combined format first
    combined = re.search(r'((?:three|3|six|6))\s+and\s+((?:three|3|six|6))\s+months\s+ended\s+((?:january|february|march|april|may|june|july|august|september|october|november|december))\s+\d{1,2}\s*,\s*(\d{4})', text)
    if combined:
        period1, period2, month, year = combined.groups()
        # Normalize periods
        period1 = '3' if period1 in ['three', '3'] else '6'
        period2 = '3' if period2 in ['three', '3'] else '6'
        periods.extend([(period1, month, year), (period2, month, year)])
    else:
        # Try single period format
        single = re.search(r'((?:three|3|six|6))\s+months\s+ended\s+((?:january|february|march|april|may|june|july|august|september|october|november|december))\s+\d{1,2}\s*,\s*(\d{4})', text)
        if single:
            period, month, year = single.groups()
            period = '3' if period in ['three', '3'] else '6'
            periods.append((period, month, year))
    
    return periods

def dates_match(sentence_dates: List[Tuple[str, str, str]], target_dates: List[Tuple[str, str, str]]) -> bool:
    """Check if dates match between sentence and target."""
    if not sentence_dates or not target_dates:
        return False
        
    # Convert to sets of (period, month, year) for comparison
    sentence_set = {(p, m, y) for p, m, y in sentence_dates}
    target_set = {(p, m, y) for p, m, y in target_dates}
    
    # Check if any dates match
    return bool(sentence_set & target_set)

def selection(changed_sentences: Dict[int, List[Tuple[List[str], str]]], sentences: Dict[int, str]) -> Dict[int, List[Tuple[List[str], str]]]:
    """Filter and match sentences with their corresponding metrics using exact matching and number verification."""
    filtered_sentences = {}
    matcher = SemanticMatcher()
    
    print("\n🔍 Starting financial data analysis...")  # Changed to print for cleaner output
    
    # Create progress bar for sentence processing
    with tqdm(total=len(changed_sentences), desc="Processing sentences", unit="sent", leave=True) as pbar:
        for key, values in changed_sentences.items():
            sentence = sentences[key]
            
            # Extract sentence dates
            sentence_dates = extract_date_info(sentence)
            if not sentence_dates:
                pbar.update(1)
                continue
            
            # Group metrics by date
            date_metrics = {}
            for value in values:
                target = ' '.join(value[0])
                target_dates = extract_date_info(target)
                if not target_dates:
                    continue
                    
                # Only process if dates match
                if dates_match(sentence_dates, target_dates):
                    date_key = '_'.join(f"{p}_{m}_{y}" for p, m, y in target_dates)
                    if date_key not in date_metrics:
                        date_metrics[date_key] = []
                    date_metrics[date_key].append((value[0][0], target, value))
            
            # Process each date group
            matches = []
            for date_key, metrics in date_metrics.items():
                target_metrics = [m[0] for m in metrics]
                
                # Get matching metrics in one call
                matching_metrics = matcher.batch_check_metrics(sentence, target_metrics)
                if matching_metrics:
                    # Get updated numbers for matching metrics in one call
                    matching_values = [m[1] for m in metrics if m[0].lower() in matching_metrics]
                    number_updates = matcher.get_updated_numbers(sentence, matching_metrics, matching_values)
                    
                    # Add matches with updated numbers
                    for metric, numbers in number_updates.items():
                        for m in metrics:
                            if m[0].lower() == metric:
                                matches.append((m[2], 1.0))
                                tqdm.write(f"✨ Updated {metric}")  # Use tqdm.write instead of print/logger
            
            # Keep all matches
            if matches:
                filtered_sentences[key] = [m[0] for m in matches]
            
            pbar.update(1)
    
    # Final summary
    total_updates = sum(len(matches) for matches in filtered_sentences.values())
    if total_updates > 0:
        print(f"\n✅ Successfully processed {total_updates} financial updates")  # Changed to print for cleaner output
    else:
        print("\n❌ No matching financial data found")  # Changed to print for cleaner output
    
    return filtered_sentences


