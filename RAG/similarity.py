from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from utils.llm import LLMConfig

# Load environment variables
load_dotenv()

# Initialize LLM configuration
llm = LLMConfig(os.getenv("QWEN_API_KEY"))

async def find_changes(
    excel_value: List[str],
    sentences: Dict[int, str],
    model: SentenceTransformer,
    threshold: float = 0.75,  # Higher threshold to filter irrelevant sentences before LLM
    metric_specific_thresholds: Optional[Dict[str, float]] = None,  # Optional metric-specific thresholds
    timeout: int = 30  # Timeout in seconds
) -> Tuple[Dict[int, List[Tuple[List[str], str, float]]], None, None]:
    """Find changes between excel values and sentences using semantic similarity.
    
    Args:
        excel_value (List[str]): List of values from Excel file
        sentences (Dict[int, str]): Dictionary mapping indices to sentences
        model (SentenceTransformer): Model for encoding text
        threshold (float, optional): Default similarity threshold. Defaults to 0.40.
        metric_specific_thresholds (Dict[str, float], optional): Metric-specific thresholds.
            Defaults to None.
        timeout (int, optional): Timeout in seconds. Defaults to 30.
        
    Returns:
        Tuple[Dict[int, List[Tuple[List[str], str, float]]], None, None]:
            - Dict mapping sentence indices to list of (target_words, new_value, confidence)
            - None (reserved for future metadata)
            - None (reserved for future metadata)
    """
    # Default metric-specific thresholds if none provided
    if metric_specific_thresholds is None:
        metric_specific_thresholds = {
            'revenue': 0.60,      # Higher threshold for common terms
            'net income': 0.55,   # Higher for compound terms
            'operating income': 0.55,
            'ebitda': 0.65       # Highest for unique terms
        }
    relevant_clips = []
    try:
        import asyncio
        for value in excel_value:
            # Wrap encode() calls in asyncio.wait_for()
            value_embedding = await asyncio.wait_for(
                asyncio.to_thread(model.encode, value),
                timeout=timeout
            )
            
            for idx, sentence in sentences.items():
                sentence_embedding = await asyncio.wait_for(
                    asyncio.to_thread(model.encode, sentence),
                    timeout=timeout
                )
                
                # Clean and normalize text for comparison
                value_lower = value.lower().strip()
                sentence_lower = sentence.lower().strip()
                
                # Calculate base similarity score
                similarity = float(1 - cosine(value_embedding, sentence_embedding))
                
                # Get metric type and threshold
                effective_threshold = 0.05  # Very low base threshold for initial matching
                for metric, metric_threshold in metric_specific_thresholds.items():
                    if metric in value_lower:
                        effective_threshold = 0.10  # Still low threshold for known metrics
                        break
                
                # Check if similarity meets threshold
                if similarity >= effective_threshold:
                    # Apply context-based boosts with focus on exact matches
                    final_similarity = similarity
                    
                    # Check for exact metric phrases
                    exact_phrases = [
                        'total revenues',
                        'net income attributable to common stockholders'
                    ]
                    if any(phrase in value_lower and phrase in sentence_lower for phrase in exact_phrases):
                        final_similarity *= 2.0  # High boost for exact metric match
                    elif value_lower in sentence_lower:
                        final_similarity *= 1.5  # Good boost for exact content match
                    
                    # Period and date validation
                    if 'three and six months ended june 30, 2023' in sentence_lower:
                        final_similarity *= 2.0  # High boost for exact date format
                    elif ('three months' in value_lower and 'three months' in sentence_lower) or \
                         ('six months' in value_lower and 'six months' in sentence_lower):
                        final_similarity *= 1.5  # Good boost for period match
                    
                    # Store with bounded confidence score
                    relevant_clips.append((idx, [([value], sentence, min(1.0, final_similarity))]))
    except asyncio.TimeoutError:
        print(f"\n⚠️ Timeout ({timeout}s) exceeded during sentence encoding")
        return {}, None, None
    except Exception as e:
        print(f"\n❌ Error during sentence encoding: {str(e)}")
        return {}, None, None
    return dict(relevant_clips), None, None

def identify_exact_words(relevant_clips, revenue_number, api_key):
    clips_text = "\n".join([clip for clip, _ in relevant_clips])
    prompt = (
        f"Given the following text segments:\n{clips_text}\n\n"
        f"Find segments that are semantically equivalent to the number {revenue_number}. "
        f"Extract the relevant numbers from these segments, whether they are expressed as integers "
        f"or in millions/billions format. Ignore any non-matching segments.\n\n"
        f"Output only the numbers in a nested list format, like [[123456], [123 million]]. "
        f"Do not include any additional reasoning."
    )

    messages = [
        {'role': 'system', 'content': 'You are a precise financial number extractor. Output only the relevant numbers in a nested list format without any additional information.'},
        {'role': 'user', 'content': prompt}
    ]


    response = llm.call(messages)

    exact_words = None
    if not response:
        return None
        
    # Clean and validate response
    try:
        # Remove any surrounding brackets and split into parts
        text = response.strip('[]').strip()
        if not text:
            return None
            
        # Split into number components
        parts = text.split(',')
        if not parts:
            return None
            
        # Extract first valid number
        for part in parts:
            # Remove non-numeric characters except decimal point
            clean_num = ''.join(c for c in part if c.isdigit() or c == '.')
            if clean_num:
                return clean_num
    except Exception as e:
        print(f"Error parsing LLM response: {str(e)}")
        return None
        
    return None
