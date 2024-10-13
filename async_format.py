import asyncio
from dashscope import Generation

api_key = "sk-1fc2f2739d444a1690d390e9cfdd8b0c"


async def format_maps(changed_sentences, sentences):
    tasks=[]
    sentences_index=[]
    for key, values in changed_sentences.items():
        temp = sentences[key]
        # change_name=[]
        # change_value=[]
        # print(key,values,temp)
        # for value in values :
        #     change_name.append(value[0])
        #     change_value.append(value[1])
        # 使用 asyncio.gather() 并发执行所有 request_llm
        
        
        tasks.append(request_llm(temp, values))
        sentences_index.append(key) 
    responses = await asyncio.gather(*tasks)  # 并发运行任务

    # 处理每个响应，获取确切的词语
    for response, key in zip(responses, sentences_index):
        words = get_exact_words(response)
        sentences[key] = words
    return sentences


async def request_llm(old_doc_value, values):
    prompt = ""
    for value in values:
        excel_value_1 = value[0]
        excel_value_2 = value[1]
        sentence = f"Please change the data corresponding to '{excel_value_1}' in this sentence '{old_doc_value} ' to '{excel_value_2}' and output the modified text."
        prompt += sentence + "\n"

    prompt += "Please carefully identify nouns to ensure there is a semantic correspondence in the sentence. Note that the rest of the text content should remain unchanged. If it already corresponds, please return an empty list and don't do anything else. If no corresponding data is found, please return without modification. Go through this process for each sentence, and finally return one sentence."
    # print(prompt)

    messages = [
        {'role': 'system',
         'content': 'You are a rigorous financial analyst. When modifying reports, you need to update paragraphs based on new information. Only respond with the modified text.'
                    'Please be sensitive to time-related keywords. If there is no match or no need for modification, please return an empty list []'},
        {'role': 'user', 'content': prompt}
    ]

    # 异步调用 Generation.call
    response = await asyncio.to_thread(
        Generation.call,
        model="qwen-max",
        messages=messages,
        result_format='message',
        api_key=api_key
    )

    return response


def get_exact_words(response):
    exact_words = None
    if response and isinstance(response, dict) and 'output' in response:
        output = response['output']
        if isinstance(output, dict) and 'choices' in output:
            for choice in output['choices']:
                if 'message' in choice and 'content' in choice['message']:
                    exact_words = choice['message']['content']
                    break
    return exact_words
