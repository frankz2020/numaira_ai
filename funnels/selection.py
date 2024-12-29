from dashscope import Generation
from funnels.document_processing import read_docx

api_key = "sk-1fc2f2739d444a1690d390e9cfdd8b0c"

def selection(changed_sentences, sentences):
    """Filter out irrelevant sentences."""
    filtered_changes = {}
    
    for key, values in changed_sentences.items():
        filtered_values = []
        sentence = sentences[key].lower()
        print(f"\nChecking sentence: {sentence}")
        
        for value in values:
            # Handle both string and list inputs for target
            if isinstance(value[0], list):
                target = ' '.join(value[0])
            else:
                target = value[0]
            
            # Clean and normalize the target
            target = target.lower().strip()
            print(f"Checking target: {target}")
            
            # Split into components and check if enough components match
            components = [comp.strip() for comp in target.split(',') if comp.strip()]
            matching_components = [comp for comp in components if comp in sentence]
            
            # If more than half of the components match, consider it valid
            if len(matching_components) >= len(components) * 0.5:
                print(f"Match found! {len(matching_components)}/{len(components)} components match")
                filtered_values.append(value)
            else:
                print(f"No match: only {len(matching_components)}/{len(components)} components match")
        
        if filtered_values:
            filtered_changes[key] = filtered_values
            print(f"Kept sentence {key} with {len(filtered_values)} matches")
    
    return filtered_changes

    

def llm_select(sen, to_be_selected):
    prompt = (
        f"Please read this sentence \n'{sen}', and decide which of the following term is this sentence describing: \n{to_be_selected}, and output the selected term, if multiple, return a list of term like ['a', 'b', 'c']. Don't do anything else."
    )
    messages = [
        {'role': 'system', 'content': 'You are a rigorous financial analyst. '},
        {'role': 'user', 'content': prompt}
    ]
    response = Generation.call(
        model="qwen2.5-72b-instruct",
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

