from dashscope import Generation
from document_processing import read_docx

api_key = "sk-1fc2f2739d444a1690d390e9cfdd8b0c"

def selection(changed_cell, sentences):
    for key, values in changed_cell.items():
        selected_list = []
        to_be_selected = set()
        for value in values:
            items = value[0].split(',')
            first_item = items[0].strip()
            to_be_selected.add(first_item)
        selected_item = llm_select(sentences[key], to_be_selected)
        selected_list.append(selected_item[2:-2])
        changed_cell[key] = process_changed_list(changed_cell[key], selected_list)

    return changed_cell

    

def llm_select(sen, to_be_selected):
    prompt = (
        f"Please read this sentence \n'{sen}', and decide which of the following term is this sentence describing: \n{to_be_selected}, and output the selected term, if multiple, return a list of term like ['a', 'b', 'c']. Don't do anything else."
    )
    messages = [
        {'role': 'system', 'content': 'You are a rigorous financial analyst. '},
        {'role': 'user', 'content': prompt}
    ]
    response = Generation.call(
        model="qwen-max",
        messages=messages,
        result_format='message',
        api_key=api_key
    )
    
    exact_words = None
    if isinstance(response, dict) and 'output' in response:
        output = response['output']
        if 'choices' in output:
            for choice in output['choices']:
                if 'message' in choice and 'content' in choice['message']:
                    exact_words = choice['message']['content']
                    break
    return exact_words

def process_changed_list(changed_list, selected_list):
    i = 0
    while i < len(changed_list):
        temp = changed_list[i][0].split(',')[0].strip()
        if temp not in selected_list:
            del changed_list[i]
        else:
            i += 1
    return changed_list

