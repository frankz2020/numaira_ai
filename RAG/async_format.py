import asyncio
import logging
from dashscope import Generation
import re

logger = logging.getLogger(__name__)

api_key = "sk-1fc2f2739d444a1690d390e9cfdd8b0c"

async def format_maps(changed_sentences, sentences):
    """Format the changed sentences using LLM."""
    logger = logging.getLogger(__name__)
    logger.info(f"Formatting {len(changed_sentences)} changed sentences")
    
    for key in changed_sentences:
        sentence = sentences[key]
        logger.info(f"Processing sentence: {sentence}")
        
        # Get all changes for this sentence
        changes = changed_sentences[key]
        
        # Group changes by period
        three_month_changes = []
        six_month_changes = []
        
        for change in changes:
            target = change[0] if isinstance(change[0], str) else change[0][0]
            value = change[1]
            period = change[0][1] if isinstance(change[0], list) else change[0].split(',')[1].strip()
            
            if 'Three Months' in period:
                three_month_changes.append((target, value))
            elif 'Six Months' in period:
                six_month_changes.append((target, value))
        
        # Find the values to replace
        values = re.findall(r'\$\d+\.\d+\s*billion', sentence)
        if len(values) >= 2:
            # Get the values based on what's in the sentence
            three_month_value = None
            six_month_value = None
            
            if 'total revenues' in sentence.lower():
                # Handle revenue sentence
                for target, value in three_month_changes:
                    if 'Total revenues' in target:
                        three_month_value = value
                for target, value in six_month_changes:
                    if 'Total revenues' in target:
                        six_month_value = value
            elif 'net income attributable to common stockholders' in sentence.lower():
                # Handle income sentence
                for target, value in three_month_changes:
                    if 'Net income attributable to common stockholders' in target:
                        three_month_value = value
                for target, value in six_month_changes:
                    if 'Net income attributable to common stockholders' in target:
                        six_month_value = value
            
            if three_month_value and six_month_value:
                # Replace the values in order
                modified = sentence
                modified = modified.replace(values[0], f"${three_month_value}")
                modified = modified.replace(values[1], f"${six_month_value}")
                sentences[key] = modified
                logger.info(f"Updated sentence to: {modified}")
            else:
                logger.warning(f"Could not find matching values for sentence {key}")
        else:
            logger.warning(f"Could not find values to replace in sentence {key}")
    
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
    