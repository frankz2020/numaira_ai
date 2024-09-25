import asyncio
from dashscope import Generation

api_key = "sk-1fc2f2739d444a1690d390e9cfdd8b0c"


async def format_maps(changed_sentences, sentences):
    for key, values in changed_sentences.items():
        temp = sentences[key]
        # 使用 asyncio.gather() 并发执行所有 request_llm
        tasks = [request_llm(temp, value[0], value[1]) for value in values]
        responses = await asyncio.gather(*tasks)  # 并发运行任务

        # 处理每个响应，获取确切的词语
        for response in responses:
            words = get_exact_words(response)
            sentences[key] = temp
    return words


async def request_llm(old_doc_value, excel_value_1, excel_value_2):
    prompt = (
        f"请将'{excel_value_1}'在{old_doc_value}中对应的数据改为'{excel_value_2}'并输出修改后文字。"
        f"注意其余文字内容要原封不动。若本身就对应，请返回一个空的列表，不要做任何别的事"
    )

    messages = [
        {'role': 'system',
         'content': '你是一个严谨的金融分析员，你在修改报告的时候需要根据新的信息来修改文段，只回答修改后的文字。'
                    '请对时间关键词保持敏感，若不匹配或无需修改，请返回一个空的列表[]'},
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
