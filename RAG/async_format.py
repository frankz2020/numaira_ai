import asyncio
import logging
from dashscope import Generation
import re

logger = logging.getLogger(__name__)

api_key = "sk-1fc2f2739d444a1690d390e9cfdd8b0c"

async def format_maps(changed_sentences, sentences):
    """Format the changed sentences with the new values."""
    logger.info(f"Formatting {len(changed_sentences)} changed sentences")
    
    for key, values in changed_sentences.items():
        sentence = sentences[key]
        logger.info(f"Processing sentence: {sentence}")
        
        # Group values by category and period
        value_groups = {}
        for value in values:
            category = value[0][0]  # First element is the category
            period = value[0][1]  # Second element is the period
            if category not in value_groups:
                value_groups[category] = {'three_month': [], 'six_month': []}
            
            if 'Three Months' in period:
                value_groups[category]['three_month'].append(value[1])
            elif 'Six Months' in period:
                value_groups[category]['six_month'].append(value[1])
        
        # Find all dollar amounts with billion/million
        pattern = r'\$\d+\.?\d*\s*(billion|million)'
        matches = list(re.finditer(pattern, sentence, re.IGNORECASE))
        
        if len(matches) >= 2:
            # Track the best matching category
            best_match = None
            best_match_score = 0
            best_values = None
            
            # For each category, find the most relevant values
            for category, periods in value_groups.items():
                three_month_values = periods['three_month']
                six_month_values = periods['six_month']
                
                # Sort values to ensure consistent order
                three_month_values.sort(reverse=True)
                six_month_values.sort(reverse=True)
                
                # Get the first value that's not "nan" for each period
                three_month_value = next((v for v in three_month_values if isinstance(v, str) and v.lower() != 'nan'), None)
                six_month_value = next((v for v in six_month_values if isinstance(v, str) and v.lower() != 'nan'), None)
                
                if three_month_value and six_month_value:
                    # Check if this category's values should be used for this sentence
                    category_words = category.lower().split()
                    sentence_lower = sentence.lower()
                    
                    # Calculate match score based on word presence and position
                    match_score = 0
                    last_pos = -1
                    for word in category_words:
                        if word in sentence_lower:
                            pos = sentence_lower.find(word)
                            if pos > last_pos:  # Words appear in the same order
                                match_score += 1
                                if last_pos != -1 and pos - last_pos < 20:  # Words are close together
                                    match_score += 0.5
                            last_pos = pos
                    
                    # If this is the best match so far, store it
                    if match_score > best_match_score:
                        best_match = category
                        best_match_score = match_score
                        best_values = (three_month_value, six_month_value)
            
            # If we found a good match, update the sentence
            if best_match and best_match_score > 0:
                three_month_value, six_month_value = best_values
                sentence = (
                    sentence[:matches[0].start()] + 
                    f"${three_month_value}" +
                    sentence[matches[0].end():matches[1].start()] +
                    f"${six_month_value}" +
                    sentence[matches[1].end():]  # Keep the rest of the sentence
                )
                logger.info(f"Updated sentence with {best_match} values: {sentence}")
                logger.info(f"Used values for {best_match}: 3-month: {three_month_value}, 6-month: {six_month_value}")
        
        sentences[key] = sentence
    
    return sentences


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
    