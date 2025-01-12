"""Confidence scoring model for document updates."""
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from funnels.metric_config import (
    METRIC_DEFINITIONS,
    get_metric_threshold,
    requires_exact_match
)
from utils.metrics import (
    normalize_text,
    extract_date_info,
    validate_periods_and_dates,
    find_metric_in_text
)

def compute_confidence(
    sentence: str,
    metric_key: str,
    embedding_similarity: float,
    llm_verified: bool,
    model: Optional[SentenceTransformer] = None
) -> float:
    """Compute confidence score using multiple signals.
    
    Features used:
    1. Embedding similarity (base signal)
    2. LLM verification boost
    3. Date/period match
    4. Exact metric match requirements
    5. Financial value presence
    6. Context signals
    
    Args:
        sentence: Original sentence text
        metric_key: Key of metric being matched
        embedding_similarity: Base similarity score
        llm_verified: Whether LLM verified the match
        model: Optional SentenceTransformer model
        
    Returns:
        float: Confidence score between 0.0-1.0
    """
    # Start with base similarity score
    confidence = embedding_similarity
    
    # Get metric-specific requirements
    metric_threshold = get_metric_threshold(metric_key)
    needs_exact = requires_exact_match(metric_key)
    
    # Check if similarity meets metric threshold
    if confidence < metric_threshold:
        return 0.0
    
    # Validate periods and dates
    has_valid_periods, period_boost, has_date = validate_periods_and_dates(sentence)
    if not has_valid_periods or not has_date:
        return 0.0
        
    # Apply period/date boost
    confidence *= period_boost
    
    # Check for exact match if required
    if needs_exact:
        metric_info = find_metric_in_text(sentence, metric_key)
        if not metric_info:
            return 0.0
            
    # Apply LLM verification boost
    if llm_verified:
        confidence *= 1.2  # 20% boost for LLM verification
        
    # Ensure confidence is bounded between 0 and 1
    return min(1.0, max(0.0, confidence))

def aggregate_confidence_scores(scores: List[float]) -> float:
    """Aggregate multiple confidence scores.
    
    Uses a weighted geometric mean to combine scores,
    giving more weight to lower scores to be conservative.
    
    Args:
        scores: List of confidence scores
        
    Returns:
        float: Aggregated score between 0.0-1.0
    """
    if not scores:
        return 0.0
        
    # Convert scores to numpy array
    score_array = np.array([float(s) for s in scores])
    
    # Calculate weights (lower scores get higher weight)
    weights = 1.0 - score_array
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    # Weighted geometric mean with epsilon to avoid log(0)
    epsilon = 1e-10
    log_scores = np.log(score_array + epsilon)
    weighted_sum = float(np.exp(np.sum(weights * log_scores)))
    
    # Ensure output is bounded between 0 and 1
    return min(1.0, max(0.0, weighted_sum))
