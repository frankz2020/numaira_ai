
from typing import Optional, Dict, Any
from utils.llm import LLMConfig

# Initialize LLM configuration
llm = LLMConfig('YOUR_API_KEY')

def format_maps(
    old_excel_value: str,
    old_doc_value: str,
    new_excel_value: str
) -> Optional[str]:
    """Format text based on pattern matching using LLM.
    
    Args:
        old_excel_value: Original value from Excel
        old_doc_value: Original value from document
        new_excel_value: New value from Excel to be formatted
        
    Returns:
        Formatted text based on the pattern, or None if formatting fails
    """
    prompt = (
        f"请根据\n'{old_excel_value}'到'{old_doc_value}'的格式变化逻辑"
        f"给出'{new_excel_value}'根据同样的变换逻辑会生成的字段。"
    )

    messages = [
        {'role': 'system', 'content': '只回答相关的数字，不要包含多余信息'},
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

def parse_nested_list(s):
    # 去掉最外层的方括号
    s = s[1:-1]
    
    result = []
    temp = ""
    nested_level = 0
    
    for char in s:
        if char == '[':
            if nested_level > 0:
                temp += char
            nested_level += 1
        elif char == ']':
            nested_level -= 1
            temp+="]"
            if nested_level == 0:
                result.append(temp.strip())
                temp = ""
            else:
                temp += char
        if nested_level > 0:
            temp += char
    
    return result

nested_list_str = "[[10,811,255], [10,811百万]]"
print(parse_nested_list(nested_list_str))
