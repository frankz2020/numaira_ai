"""NER-based metric detection for financial documents."""
from typing import List, Dict, Optional, Tuple, Any, Union
import re
import logging
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np
from utils.metrics import normalize_text, extract_financial_values
from funnels.metric_config import (
    METRIC_DEFINITIONS,
    get_metric_variations,
    get_metric_threshold
)
from funnels.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)

def detect_metrics(
    text: str,
    model: Optional[SentenceTransformer] = None,
    threshold: float = 0.75
) -> List[Dict[str, Any]]:
    """Detect financial metrics in text using NER and embedding similarity.
    
    Uses a layered approach:
    1. Initial filtering with sentence embeddings
    2. LLM verification for high-confidence matches
    3. Financial value extraction and validation
    
    Args:
        text: Input text to analyze
        model: Optional SentenceTransformer model (loads default if None)
        threshold: Similarity threshold (default: 0.75)
        
    Returns:
        List of detected metrics with structure:
        [
            {
                'metric': str,          # Metric name
                'variation': str,       # Matched variation
                'values': List[float],  # Extracted values
                'confidence': float     # Detection confidence (0.0-1.0)
            }
        ]
    """
    # Normalize input text
    text = normalize_text(text)
    
    detected_metrics: List[Dict[str, Any]] = []
    
    # Load default model if none provided
    if model is None:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Failed to load default model: {str(e)}")
            return detected_metrics
    
    if not isinstance(model, SentenceTransformer):
        logger.error("Invalid model type provided")
        return detected_metrics
        
    # Get text embedding
    try:
        text_embedding = model.encode(text, convert_to_numpy=True)
        if not isinstance(text_embedding, np.ndarray):
            text_embedding = np.array(text_embedding)
    except Exception as e:
        logger.error(f"Failed to encode text: {str(e)}")
        return detected_metrics
        
        detected_metrics = []
        for metric_key, metric_info in METRIC_DEFINITIONS.items():
            # Get variations and encode
            variations = get_metric_variations(metric_key)
            # Encode variations with type validation
            try:
                variation_embeddings = model.encode(variations, convert_to_numpy=True)
                if not isinstance(variation_embeddings, np.ndarray):
                    variation_embeddings = np.array(variation_embeddings)
            except Exception as e:
                logger.error(f"Failed to encode variations for {metric_key}: {str(e)}")
                continue
            
            # Find best matching variation
            best_similarity: float = 0.0
            best_variation: Optional[str] = None
            
            for variation, embedding in zip(variations, variation_embeddings):
                try:
                    similarity = float(1 - cosine(text_embedding, embedding))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_variation = variation
                except Exception as e:
                    logger.error(f"Failed to calculate similarity: {str(e)}")
                    continue
        
        # Check against threshold
        metric_threshold = get_metric_threshold(metric_key)
        if best_similarity >= metric_threshold:
            # Extract financial values
            values = extract_financial_values(text)
            
            # Verify with LLM for high-confidence matches
            # Process best match if found
            if best_variation and isinstance(best_variation, str):
                # Extract financial values
                values = extract_financial_values(text)
                if values:
                    # Verify with LLM
                    llm_match = False
                    try:
                        llm_provider = get_llm_provider('qwen')
                        matched_metrics = llm_provider.batch_check_metrics(
                            sentence=text,
                            target_metrics=[best_variation]
                        )
                        llm_match = bool(matched_metrics)
                    except Exception as e:
                        logger.error(f"Failed LLM verification for {metric_key}: {str(e)}")
                    
                    if llm_match:
                        # Calculate final confidence score
                        try:
                            confidence = float(min(1.0, float(best_similarity) * 1.2))  # Boost for LLM verification
                            float_values = [float(v[0]) for v in values]
                            
                            # Add validated metric
                            detected_metrics.append({
                                'metric': str(metric_info['name']),
                                'variation': best_variation,
                                'values': float_values,
                                'confidence': confidence
                            })
                            logger.info(f"Added metric {metric_info['name']} with confidence {confidence:.2%}")
                        except (ValueError, TypeError, IndexError) as e:
                            logger.error(f"Failed to process values for {metric_key}: {str(e)}")
                else:
                    logger.warning(f"No financial values found for {metric_key}")
    
    return detected_metrics
