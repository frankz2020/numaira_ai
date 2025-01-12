
import asyncio
from typing import Optional, Dict, Any, Tuple
from utils.llm import LLMConfig

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LLM configuration
llm = LLMConfig(os.getenv("QWEN_API_KEY"))

async def format_maps(
    old_excel_value: str,
    old_doc_value: str,
    new_excel_value: str,
    confidence: float = 0.3  # Default to threshold value
) -> Tuple[Optional[str], float]:
    """Format text based on pattern matching using LLM.
    
    LLM Integration:
        Provider: Qwen-max model
        Input Format:
            - System: Role definition and output format instructions
            - User: Pattern analysis and formatting request
        Error Handling:
            - Timeout: 30 seconds default
            - Response Validation: Strip whitespace, check for empty/invalid
            - Confidence Bounds: Clamp to 0.0-1.0 range
    
    Pattern Analysis:
        1. Identify formatting pattern between old Excel value and document text
        2. Extract number format (e.g., "$X,XXX.XX million")
        3. Preserve surrounding context and temporal terms
        4. Apply same pattern to new Excel value
    
    Args:
        old_excel_value: Original value from Excel (e.g., "1234.56")
        old_doc_value: Original value from document (e.g., "$1,234.56 million")
        new_excel_value: New value from Excel to be formatted
        confidence: Confidence score from similarity matching (0.0-1.0)
        
    Returns:
        Tuple[Optional[str], float]: Tuple containing:
            - Formatted text based on the pattern, or None if formatting fails
            - Confidence score from similarity matching (bounded between 0.0-1.0)
            
    Example:
        >>> text, conf = await format_maps("1234.56", "$1,234.56 million", "5678.90")
        >>> print(f"Formatted: {text}")
        >>> print(f"Confidence: {conf:.2%}")
        Formatted: $5,678.90 million
        Confidence: 85.00%
    """
    # Validate confidence score bounds
    confidence = max(0.0, min(1.0, confidence))
    prompt = (
        f"Given the formatting pattern between:\n"
        f"Original Excel value: '{old_excel_value}'\n"
        f"Original document text: '{old_doc_value}'\n\n"
        f"Apply the same formatting pattern to generate text for new Excel value: '{new_excel_value}'\n"
        f"Only output the formatted text without any additional information."
    )

    messages = [
        {'role': 'system', 'content': 'You are a precise financial text formatter. Only output the formatted text without any additional information.'},
        {'role': 'user', 'content': prompt}
    ]
    try:
        # Make async LLM call
        response = await asyncio.to_thread(llm.call, messages)
        
        if not response:
            return None, confidence
            
        # Clean and validate response
        text = response.strip()
        if not text or text == '[]':
            return None, confidence
            
        return text, confidence
            
    except Exception as e:
        print(f"Error in format_maps: {str(e)}")
        return None, confidence
