import asyncio
import logging
from dashscope import Generation
import re
from typing import List, Tuple
from funnels.document_processing import values_are_equal

logger = logging.getLogger(__name__)

api_key = "sk-1fc2f2739d444a1690d390e9cfdd8b0c"

def extract_numbers(text: str) -> List[Tuple[str, int, int]]:
    """Extract numbers and their positions from text."""
    pattern = r'\$?\d+\.?\d*\s*(?:billion|million|thousand)?'
    matches = list(re.finditer(pattern, text))
    return [(m.group(), m.start(), m.end()) for m in matches]

def format_changed_sentence(original: str, changes: List[Tuple[List[str], str]]) -> str:
    """Format a sentence with changes, only marking actual value changes."""
    result = original
    offset = 0
    
    # Extract original numbers and their positions
    original_numbers = extract_numbers(original)
    
    # Process each change
    for change in changes:
        new_value = change[1]
        
        # Find matching number in original text
        for orig_num, start, end in original_numbers:
            # Compare the values
            if not values_are_equal(orig_num, new_value):
                # Replace the number only if it's different
                replacement = f'<span class="changed-text"><span class="original">{orig_num}</span><span class="modified">{new_value}</span></span>'
                result = result[:start + offset] + replacement + result[end + offset:]
                offset += len(replacement) - (end - start)
            else:
                # If values are equal, keep the original number without any markup
                result = result[:start + offset] + orig_num + result[end + offset:]
    
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
            sentence = f"Given the infomation and time of '{excel_value_1}', change this sentence corresponding value '{old_doc_value} ' to '{excel_value_2}', be sensitive to time-related term; and output the modified text, directly without any other information."
            prompt += sentence + "\n"

        prompt += "Please carefully identify nouns to ensure there is a semantic correspondence in the sentence. Note that the rest of the text content should remain unchanged. If it already corresponds, please return an empty list, [], and don't do anything else. If no corresponding data is found, please return without modification."
        
        logger.info(f"Sending prompt for value: {excel_value_1}")

        messages = [
            {'role': 'system',
             'content': 'You are a rigorous financial analyst. Only respond with the modified text.'
                      'Please be sensitive to time-related keywords. If there is no match or no need for modification, please return an empty list []'},
            {'role': 'user', 'content': prompt}
        ]

        # Async call to Generation.call
        response = await asyncio.to_thread(
            Generation.call,
            model="qwen2.5-72b-instruct",
            messages=messages,
            result_format='message',
            api_key=api_key
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
    
async def format_maps(changed_sentences, sentences):
    """Format and update sentences with changes."""
    logger.info("Starting to format changes")
    formatted_sentences = sentences.copy()
    
    for key, values in changed_sentences.items():
        old_doc_value = sentences[key]
        response = await request_llm(old_doc_value, values)
        
        if response:
            words = get_exact_words(response)
            if words and words != '[]':
                formatted_sentences[key] = words
            else:
                formatted_sentences[key] = old_doc_value
        else:
            formatted_sentences[key] = old_doc_value
            
    return formatted_sentences
    