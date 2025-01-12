"""Utility functions for processing financial metrics."""
import re
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from funnels.metric_config import (
    METRIC_DEFINITIONS,
    PERIOD_PATTERNS,
    DATE_PATTERNS,
    FINANCIAL_UNITS
)

def normalize_text(text: str) -> str:
    """Normalize text for comparison.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text (lowercase, clean whitespace)
    """
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_date_info(text: str) -> List[Tuple[str, str, str]]:
    """Extract period, month, and year from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of (period, month, year) tuples
    """
    text = normalize_text(text)
    periods = []
    
    # Try combined format first (e.g., "three and six months")
    combined = re.search(
        r'((?:three|3|six|6))\s+and\s+((?:three|3|six|6))\s+months\s+ended\s+'
        r'((?:january|february|march|april|may|june|july|august|september|october|november|december))'
        r'\s+\d{1,2}\s*,\s*(\d{4})',
        text
    )
    
    if combined:
        period1, period2, month, year = combined.groups()
        # Normalize periods
        period1 = '3' if period1 in ['three', '3'] else '6'
        period2 = '3' if period2 in ['three', '3'] else '6'
        periods.extend([(period1, month, year), (period2, month, year)])
    else:
        # Try single period format
        for pattern in DATE_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    date_str = match.group()
                    # Parse date using multiple formats
                    for fmt in ['%B %d, %Y', '%b %d, %Y', '%m/%d/%Y', '%Y-%m-%d']:
                        try:
                            date = datetime.strptime(date_str, fmt)
                            # Look for period before date
                            period_text = text[:match.start()].strip()
                            for period_key, patterns in PERIOD_PATTERNS.items():
                                if any(p in period_text for p in patterns):
                                    period = '3' if period_key == 'three_month' else '6'
                                    periods.append((
                                        period,
                                        date.strftime('%B').lower(),
                                        date.strftime('%Y')
                                    ))
                            break
                        except ValueError:
                            continue
                except Exception:
                    continue
    
    return periods

def extract_financial_values(text: str) -> List[Tuple[float, str]]:
    """Extract financial values and their units.
    
    Args:
        text: Text to analyze
        
    Returns:
        List of (value, unit) tuples
    """
    text = normalize_text(text)
    values = []
    
    # Match patterns like "$1,234.56 million" or "1,234.56 billion"
    pattern = r'\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|mn|bn|k|thousand)'
    matches = re.finditer(pattern, text)
    
    for match in matches:
        try:
            value_str, unit = match.groups()
            value = float(value_str.replace(',', ''))
            
            # Normalize units
            if unit in ['billion', 'bn']:
                value *= 1000  # Convert to millions
            elif unit in ['thousand', 'k']:
                value /= 1000  # Convert to millions
                
            values.append((value, 'million'))
        except ValueError:
            continue
            
    return values

def find_metric_in_text(text: str, metric_key: str) -> Optional[Dict[str, any]]:
    """Find a specific metric in text.
    
    Args:
        text: Text to analyze
        metric_key: Key of metric to find
        
    Returns:
        Dict with match info or None if not found
    """
    if metric_key not in METRIC_DEFINITIONS:
        return None
        
    text = normalize_text(text)
    metric_info = METRIC_DEFINITIONS[metric_key]
    
    # Check variations
    for variation in metric_info['variations']:
        if variation in text:
            # Get surrounding context
            context_start = max(0, text.find(variation) - 50)
            context_end = min(len(text), text.find(variation) + len(variation) + 50)
            context = text[context_start:context_end]
            
            return {
                'metric': metric_info['name'],
                'matched_variation': variation,
                'context': context,
                'requires_exact': metric_info['requires_exact_match'],
                'confidence_threshold': metric_info['confidence_threshold']
            }
            
    return None

def validate_periods_and_dates(text: str) -> Tuple[bool, float, bool]:
    """Validate period references and dates in text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Tuple of (has_valid_periods, confidence_boost, has_date)
    """
    text = normalize_text(text)
    
    # Extract date information
    date_info = extract_date_info(text)
    has_date = bool(date_info)
    
    # Check for exact period format
    if 'three and six months ended june 30, 2023' in text:
        return True, 2.0, True  # High confidence for exact match
        
    # Check for combined periods
    combined_patterns = [
        'three and six months',
        '3 and 6 months',
        'three and 6 months',
        '3 and six months'
    ]
    has_combined = any(pattern in text for pattern in combined_patterns)
    if has_combined:
        return True, 1.5, has_date
        
    # Check individual periods
    has_three = any(pattern in text for pattern in PERIOD_PATTERNS['three_month'])
    has_six = any(pattern in text for pattern in PERIOD_PATTERNS['six_month'])
    
    if has_three and has_six:
        return True, 1.25, has_date
        
    return False, 0.0, has_date
