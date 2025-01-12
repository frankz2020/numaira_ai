import asyncio
import logging
import re
from typing import List, Tuple, Optional
import os
from dotenv import load_dotenv
from utils.llm import LLMConfig
from funnels.document_processing import values_are_equal

# Load environment variables
load_dotenv()

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize LLM configuration
llm = LLMConfig(os.getenv("QWEN_API_KEY"))

def extract_numbers(text: str) -> List[Tuple[str, int, int]]:
    """Extract numbers and their positions from text."""
    pattern = r'\$?\d+\.?\d*\s*(?:billion|million|thousand)?'
    matches = list(re.finditer(pattern, text))
    return [(m.group(), m.start(), m.end()) for m in matches]

def format_changed_sentence(original: str, new_values: List[str]) -> str:
    """Format a sentence with changes, preserving exact formatting.
    
    Args:
        original: Original sentence text
        new_values: List of new values to insert
        
    Returns:
        Updated sentence with new values inserted
    """
    print(f"\nFormatting sentence:")
    print(f"Original: {original}")
    print(f"New values: {new_values}")
    
    # Extract all numbers with their formatting
    pattern = r'\$(\d+\.\d+)\s*billion'
    matches = list(re.finditer(pattern, original))
    
    if not matches:
        print("No numbers found in original text")
        return original
        
    # Handle both single and paired number updates
    if len(matches) == 2 and len(new_values) == 1:
        # Split the new value if it contains two numbers
        parts = new_values[0].split(' and ')
        if len(parts) == 2:
            new_values = [p.strip('$').strip(' billion') for p in parts]
    
    if len(matches) != len(new_values):
        print(f"Mismatch in number count: found {len(matches)}, have {len(new_values)} new values")
        return original
        
    # Replace numbers while preserving formatting
    result = original
    offset = 0
    for i, match in enumerate(matches):
        if i < len(new_values):
            # Get the new value and ensure it's formatted correctly
            new_val = new_values[i]
            if isinstance(new_val, (int, float)):
                new_val = f"{float(new_val):.2f}"
            elif isinstance(new_val, str):
                # Clean and convert string to float
                clean_val = new_val.replace('$', '').replace(',', '').replace('billion', '').strip()
                try:
                    new_val = f"{float(clean_val):.2f}"
                except ValueError:
                    print(f"Invalid number format: {new_val}")
                    continue
            
            print(f"Replacing {match.group(1)} with {new_val}")
            
            # Replace while preserving $ and billion
            start = match.start(1) + offset
            end = match.end(1) + offset
            result = result[:start] + new_val + result[end:]
            offset += len(new_val) - (match.end(1) - match.start(1))
    
    print(f"Result: {result}")
    return result

def format_changes(changed_sentences, original_sentences):
    """Format all changed sentences."""
    logger.info("Formatting changed sentences")
    formatted_sentences = []
    
    for key, changes in changed_sentences.items():
        original = original_sentences[key]
        formatted = format_changed_sentence(original, changes)
        formatted_sentences.append(formatted)
    
    return formatted_sentences

async def request_llm(old_doc_value, values):
    try:
        prompt = ""
        for value in values:
            excel_value_1 = value[0]
            excel_value_2 = value[1]
            sentence = (
                f"Original value: '{excel_value_1}'\n"
                f"Document text: '{old_doc_value}'\n"
                f"New value: '{excel_value_2}'\n\n"
                f"Task: Update the document text with the new value, preserving the exact format and time-related terms. "
                f"Output only the modified text without any additional information."
            )
            prompt += sentence + "\n\n"

        prompt += (
            "Important:\n"
            "- Carefully match financial metrics and their context\n"
            "- Preserve all formatting and surrounding text\n"
            "- If values already match or no valid update is possible, return []\n"
            "- Output only the modified text, no explanations"
        )
        
        logger.info(f"Sending prompt for value: {excel_value_1}")

        messages = [
            {'role': 'system',
             'content': 'You are a precise financial text formatter. Output only the modified text, preserving exact formatting and numerical context. '
                      'Return [] if no valid update is possible or if values already match.'},
            {'role': 'user', 'content': prompt}
        ]

        # Async call to LLM
        response = await asyncio.to_thread(
            llm.call,
            messages=messages
        )
        
        if response and 'output' in response:
            logger.info("Received valid response from LLM")
        else:
            logger.warning("Received invalid response format from LLM")
            
        return response
    except Exception as e:
        logger.error(f"Error in request_llm: {str(e)}")
        return None


def get_exact_words(response):
    try:
        if not response:
            logger.warning("Empty response received")
            return None
            
        if isinstance(response, dict) and 'output' in response:
            output = response['output']
            if isinstance(output, dict) and 'choices' in output:
                for choice in output['choices']:
                    if 'message' in choice and 'content' in choice['message']:
                        content = choice['message']['content']
                        # Only strip brackets if content isn't empty and contains brackets
                        if content and content.strip() != '[]':
                            return content.strip()
                        elif content.strip() == '[]':
                            logger.info("LLM indicated no changes needed")
                            return None
        
        logger.warning(f"Unexpected response format: {response}")
        return None
    except Exception as e:
        logger.error(f"Error in get_exact_words: {str(e)}")
        return None
    
async def format_maps(old_excel_value: str, old_doc_value: str, new_excel_value: str, confidence: float = 0.3) -> Tuple[Optional[str], float]:
    """Format text based on pattern matching using LLM.
    
    Args:
        old_excel_value: Original value from Excel
        old_doc_value: Original value from document
        new_excel_value: New value from Excel to be formatted
        confidence: Confidence score from similarity matching (0.0-1.0)
        
    Returns:
        Tuple containing:
        - Formatted text based on the pattern, or None if formatting fails
        - Confidence score from similarity matching (bounded between 0.0-1.0)
    """
    # Validate confidence score bounds
    confidence = max(0.0, min(1.0, confidence))
    
    try:
        # Extract numbers from Excel values
        pattern = r'(\d+\.?\d+)\s*billion'
        old_numbers = re.findall(pattern, old_excel_value)
        new_numbers = re.findall(pattern, new_excel_value)
        
        print(f"\nExtracting numbers:")
        print(f"From old Excel value: {old_excel_value} -> {old_numbers}")
        print(f"From new Excel value: {new_excel_value} -> {new_numbers}")
        
        if not old_numbers or not new_numbers:
            print("No valid numbers found in Excel values")
            return None, confidence
            
        # Format the sentence with new numbers
        formatted = format_changed_sentence(old_doc_value, new_numbers)
        if formatted and formatted != old_doc_value:
            return formatted, confidence
        
        return None, confidence
        
    except Exception as e:
        logger.error(f"Error in format_maps: {str(e)}")
        return None, confidence
    