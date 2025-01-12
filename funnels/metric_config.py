"""Centralized configuration for financial metrics and their variations."""
from typing import Dict, List, Set

# Core metric definitions with variations and metadata
METRIC_DEFINITIONS: Dict[str, Dict[str, any]] = {
    'total_revenues': {
        'name': 'Total revenues',
        'variations': [
            'total revenues',
            'total revenue',
            'revenues',
            'revenue',
            'consolidated revenue',
            'consolidated revenues'
        ],
        'period_forms': {
            'three_month': 'Total revenues (3 months)',
            'six_month': 'Total revenues (6 months)'
        },
        'confidence_threshold': 0.60,  # Higher threshold for common terms
        'requires_exact_match': False
    },
    'net_income': {
        'name': 'Net income attributable to common stockholders',
        'variations': [
            'net income attributable to common stockholders',
            'net income',
            'income attributable to stockholders',
            'net income to stockholders',
            'our net income attributable to common stockholders'
        ],
        'period_forms': {
            'three_month': 'Net income (3 months)',
            'six_month': 'Net income (6 months)'
        },
        'confidence_threshold': 0.55,  # Higher for compound terms
        'requires_exact_match': True
    },
    'operating_income': {
        'name': 'Operating income',
        'variations': [
            'operating income',
            'operating profit',
            'income from operations',
            'operating earnings'
        ],
        'period_forms': {
            'three_month': 'Operating income (3 months)',
            'six_month': 'Operating income (6 months)'
        },
        'confidence_threshold': 0.55,
        'requires_exact_match': False
    },
    'ebitda': {
        'name': 'EBITDA',
        'variations': [
            'ebitda',
            'earnings before interest taxes depreciation and amortization',
            'earnings before interest, taxes, depreciation and amortization'
        ],
        'period_forms': {
            'three_month': 'EBITDA (3 months)',
            'six_month': 'EBITDA (6 months)'
        },
        'confidence_threshold': 0.65,  # Highest for unique terms
        'requires_exact_match': True
    }
}

# Period definitions and variations
PERIOD_PATTERNS: Dict[str, List[str]] = {
    'three_month': [
        'three months',
        '3 months',
        'three-month',
        '3-month',
        'quarter',
        'three month period'
    ],
    'six_month': [
        'six months',
        '6 months',
        'six-month',
        '6-month',
        'two quarters',
        'six month period'
    ]
}

# Date format patterns
DATE_PATTERNS: List[str] = [
    r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\s*,?\s*\d{4}',
    r'\d{1,2}/\d{1,2}/\d{4}',
    r'\d{4}-\d{2}-\d{2}'
]

# Financial unit patterns
FINANCIAL_UNITS: Set[str] = {
    'million',
    'billion',
    'thousands',
    'mn',
    'bn',
    'k'
}

def get_metric_variations(metric_key: str) -> List[str]:
    """Get all variations for a given metric key."""
    if metric_key not in METRIC_DEFINITIONS:
        return []
    return METRIC_DEFINITIONS[metric_key]['variations']

def get_metric_threshold(metric_key: str) -> float:
    """Get confidence threshold for a given metric key."""
    if metric_key not in METRIC_DEFINITIONS:
        return 0.50  # Default threshold
    return METRIC_DEFINITIONS[metric_key]['confidence_threshold']

def get_period_patterns(period_key: str) -> List[str]:
    """Get all patterns for a given period key."""
    if period_key not in PERIOD_PATTERNS:
        return []
    return PERIOD_PATTERNS[period_key]

def requires_exact_match(metric_key: str) -> bool:
    """Check if metric requires exact matching."""
    if metric_key not in METRIC_DEFINITIONS:
        return False
    return METRIC_DEFINITIONS[metric_key]['requires_exact_match']
