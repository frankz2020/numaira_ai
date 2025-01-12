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
    changed_sentences: Dict[int, List[Tuple[List[str], str]]],
    sentences: Dict[int, str]
) -> Dict[int, List[Tuple[List[str], str]]]:
    """Filter and match sentences with their corresponding metrics using exact matching and number verification.
    
    Args:
        changed_sentences: Dictionary mapping sentence indices to lists of potential changes
                         Each change is a tuple of (target_words, new_value)
        sentences: Dictionary mapping indices to original sentences
        
    Returns:
        Dictionary mapping sentence indices to confirmed changes
        Each confirmed change is a tuple of (target_words, new_value)
    """
    filtered_sentences = {}
    llm = get_llm_provider()  # Get configured LLM provider
    
    print("\n🔍 Starting financial data analysis...")
    
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
                matching_metrics = llm.batch_check_metrics(sentence, target_metrics)
                if matching_metrics:
                    # Get updated numbers for matching metrics in one call
                    matching_values = [m[1] for m in metrics if m[0].lower() in matching_metrics]
                    number_updates = llm.get_updated_numbers(sentence, matching_metrics, matching_values)
                    
                    # Add matches with updated numbers
                    for metric, numbers in number_updates.items():
                        for m in metrics:
                            if m[0].lower() == metric:
                                matches.append((m[2], 1.0))
                                tqdm.write(f"✨ Updated {metric}")
            
            # Keep all matches
            if matches:
                filtered_sentences[key] = [m[0] for m in matches]
            
            pbar.update(1)
    
    # Final summary
    total_updates = sum(len(matches) for matches in filtered_sentences.values())
    if total_updates > 0:
        print(f"\n✅ Successfully processed {total_updates} financial updates")
    else:
        print("\n❌ No matching financial data found")
    
    return filtered_sentences


