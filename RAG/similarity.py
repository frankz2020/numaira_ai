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

def find_changes(
    excel_value: List[str],
    sentences: Dict[int, str],
    model: SentenceTransformer,
    threshold: float = 0.3
) -> Tuple[Dict[int, List[Tuple[List[str], str, float]]], None, None]:
    """Find changes between excel values and sentences using semantic similarity.
    
    Args:
        excel_value: List of values from Excel file
        sentences: Dictionary mapping indices to sentences from Word document
        model: SentenceTransformer model for encoding text
        threshold: Minimum similarity threshold for matching
        
    Returns:
        Tuple containing:
        - Dictionary mapping sentence indices to list of (target_words, new_value, confidence) pairs
          where target_words is a list containing the original value and confidence is similarity score
        - None (reserved for future metadata)
        - None (reserved for future metadata)
    """
    relevant_clips = []
    for value in excel_value:
        value_embedding = model.encode(value)
        for idx, sentence in sentences.items():
            sentence_embedding = model.encode(sentence)
            # Calculate similarity score (already bounded 0-1 by cosine distance)
            similarity = float(1 - cosine(value_embedding, sentence_embedding))
            if similarity >= threshold:
                # Store similarity score as confidence
                relevant_clips.append((idx, [([value], sentence, similarity)]))
    return dict(relevant_clips), None, None

def identify_exact_words(relevant_clips, revenue_number, api_key):
    clips_text = "\n".join([clip for clip, _ in relevant_clips])
    prompt = (
        f"给定以下文字片段：\n{clips_text}\n\n"
        f"请找出其中与这个数字 {revenue_number} 在意义上相等的文字片段，并从这些文字片段中提取出相关的数字，不论它们是用整数还是百万的形式表示。如果没有意义相等的文字片段，请忽略它们。"
        f"只需回答相关的数字，并使用嵌套列表格式，如 [[123456], [123百万]]。不要包含多余的推理信息。"
    )

    messages = [
        {'role': 'system', 'content': '只回答相关的数字，用嵌套列表装起来，不要包含多余信息'},
        {'role': 'user', 'content': prompt}
    ]


    response = llm.call(messages)

    exact_words = None
    if isinstance(response, dict) and 'output' in response:
        output = response['output']
        if 'choices' in output:
            for choice in output['choices']:
                if 'message' in choice and 'content' in choice['message']:
                    exact_words = choice['message']['content']
                    break
    return exact_words
