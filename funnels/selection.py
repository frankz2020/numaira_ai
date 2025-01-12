from funnels.document_processing import read_docx
import re
from typing import Dict, List, Tuple, Set
import logging
import os
from dotenv import load_dotenv
from tqdm import tqdm
import sys
from funnels.llm_provider import get_llm_provider

# Load environment variables
load_dotenv()

from utils.logging import setup_logging

# Set up logging
logger = setup_logging(level=logging.INFO)

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

from typing import Dict, List, Tuple, Optional
from datetime import datetime

def selection(
    changed_sentences: Dict[int, List[Tuple[List[str], str, float]]],
    sentences: Dict[int, str]
) -> Dict[int, List[Tuple[List[str], str, float]]]:
    """Filter and match sentences with their corresponding metrics using exact matching.
    
    Pipeline Steps:
    1. Text Preprocessing
       - Convert to lowercase
       - Remove special characters
       - Normalize whitespace
    2. Date Information Extraction
       - Parse period (three/six months)
       - Extract month and year
       - Handle combined periods
    3. Metric Matching
       - Verify exact word presence
       - Match financial metrics
       - Validate temporal context
    4. Confidence Scoring
       - Propagate similarity scores
       - Filter low confidence matches
    
    Data Structures:
        changed_sentences: {
            sentence_index: [
                (
                    [target_words],  # List of words to match (e.g., ["Revenue"])
                    new_value,       # Updated value (e.g., "1234.56")
                    confidence       # Similarity score (0.0-1.0)
                )
            ]
        }
        
        sentences: {
            sentence_index: str  # Original sentence text
        }
    
    Args:
        changed_sentences: Dictionary mapping sentence indices to lists of potential changes
        sentences: Dictionary mapping indices to original sentences
        
    Returns:
        Dict[int, List[Tuple[List[str], str, float]]]: Filtered dictionary of confirmed changes
        
    Example:
        >>> changes = {0: [(['Revenue'], '1234.56', 0.85)]}
        >>> sentences = {0: 'Revenue was $1,000.00 million'}
        >>> filtered = selection(changes, sentences)
        >>> print(filtered[0][0])  # First change for sentence 0
        (['Revenue'], '1234.56', 0.85)
    """
    filtered_sentences = {}
    
    print("\n🔍 Starting financial data analysis...")
    
    # Create progress bar for sentence processing
    with tqdm(total=len(changed_sentences), desc="Processing sentences", unit="sent", leave=True) as pbar:
        for key, values in changed_sentences.items():
            sentence = sentences[key].lower()
            if sentence == "financial report":  # Skip title
                pbar.update(1)
                continue
                
            matches = []
            for value in values:
                target_words = value[0]  # List of target words
                new_value = value[1]     # New value to update
                confidence = value[2]     # Confidence score
                
                # Balanced validation with exact metric matching
                try:
                    if not isinstance(target_words, list) or not target_words:
                        print(f"Warning: Invalid target words format: {target_words}")
                        continue
                        
                    metric_type = str(target_words[0]).lower()  # e.g., 'revenue', 'net income', 'ebitda'
                    sentence_lower = sentence.lower()
                    
                    # Define metric mappings with exact matches
                    metric_mappings = {
                        'revenue': ['revenue'],
                        'net income': ['net income'],
                        'operating income': ['operating income'],
                        'ebitda': ['ebitda']
                    }
                    
                    # Check for exact metric match
                    metric_match = False
                    for metric_key, terms in metric_mappings.items():
                        if metric_type == metric_key:
                            # Check if any term appears in sentence
                            for term in terms:
                                if term in sentence_lower:
                                    metric_match = True
                                    print(f"✓ Found metric '{metric_key}' in sentence")
                                    break
                            if metric_match:
                                break
                    
                    # Extract dates for confidence boosting
                    date_info = extract_date_info(sentence_lower)
                    has_date = bool(date_info)
                    
                    # Check period match for confidence boosting
                    has_period = False
                    if any(period in sentence_lower for period in 
                          ['three months', 'three-month', '3 months',
                           'six months', 'six-month', '6 months']):
                        has_period = True
                except (IndexError, TypeError, AttributeError) as e:
                    print(f"Warning: Error processing metric: {str(e)}")
                    continue
                
                # Basic validation with confidence boosting
                has_financial_units = any(term in sentence_lower for term in ['million', 'billion'])
                has_currency = '$' in sentence_lower
                has_verb = any(term in sentence_lower for term in ['was', 'increased to', 'decreased to'])
                
                # Require metric match and financial context
                if metric_match and has_financial_units:
                    # Start with base confidence
                    context_score = 1.0
                    
                    # Boost for exact metric match at start
                    if metric_type == sentence_lower.split()[0]:
                        context_score *= 1.2
                    
                    # Boost for period match
                    if has_period:
                        context_score *= 1.15
                        
                    # Boost for date match
                    if has_date:
                        context_score *= 1.15
                        
                    # Boost for currency and verb presence
                    if has_currency:
                        context_score *= 1.1
                    if has_verb:
                        context_score *= 1.1
                    
                    scaled_confidence = confidence * context_score
                    if scaled_confidence >= 0.35:  # Lower threshold with balanced validation
                        matches.append((target_words, new_value, min(1.0, max(0.0, scaled_confidence))))
                    metric = target_words[0]  # First word is usually the metric
                    tqdm.write(f"✨ Matched {metric} (confidence: {confidence:.1%})")
            
            # Keep all matches with full tuple structure
            if matches:
                filtered_sentences[key] = matches
            
            pbar.update(1)
    
    # Final summary
    total_updates = sum(len(matches) for matches in filtered_sentences.values())
    if total_updates > 0:
        print(f"\n✅ Successfully processed {total_updates} financial updates")
    else:
        print("\n❌ No matching financial data found")
    
    return filtered_sentences


