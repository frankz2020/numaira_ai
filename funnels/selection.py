from dashscope import Generation
from funnels.document_processing import read_docx

api_key = "sk-1fc2f2739d444a1690d390e9cfdd8b0c"

def selection(changed_sentences, sentences):
    """Filter out irrelevant sentences."""
    filtered_sentences = {}
    
    for key, values in changed_sentences.items():
        sentence = sentences[key].lower()
        print(f"\nChecking sentence: {sentence}")
        
        matches = 0
        for value in values:
            target = ' '.join(value[0]).lower()
            print(f"Checking target: {target}")
            
            # Check if any component of the target appears in the sentence
            target_components = target.split()
            
            # Count matching components, but require key terms to match
            key_terms = ['months', 'ended', 'june', '30', '2023', '2022']
            matching_components = 0
            key_terms_matched = 0
            
            for comp in target_components:
                if comp in sentence:
                    matching_components += 1
                    if comp in key_terms:
                        key_terms_matched += 1
            
            # Require at least one key term to match and a minimum percentage of components
            min_match_percentage = 0.5
            if key_terms_matched > 0 and matching_components / len(target_components) >= min_match_percentage:
                print(f"Match found! {matching_components}/{len(target_components)} components match")
                matches += 1
            else:
                print(f"No match: insufficient component matches or no key terms matched")
        
        # If we found any matches, keep this sentence
        if matches > 0:
            filtered_sentences[key] = values
            print(f"Kept sentence {key} with {matches} matches")
    
    return filtered_sentences

    

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

