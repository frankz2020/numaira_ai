import re
from document_processing import embed_text
from sklearn.metrics.pairwise import cosine_similarity
def find_relevant_sentences(sentences, target_text, model, threshold):
    # Embed the target text
    target_embedding = embed_text(target_text, model)
    # Embed all sentences
    sentence_embeddings = [embed_text(sentence, model) for sentence in sentences]
    # Calculate cosine similarities
    similarities = cosine_similarity([target_embedding], sentence_embeddings)[0]
    # Collect sentences with similarity above threshold
    relevant_sentences = []
    for sentence, similarity in zip(sentences, similarities):
        if similarity >= threshold:
            relevant_sentences.append((sentence, similarity))
    # Sort relevant sentences by similarity in descending order
    return sorted(relevant_sentences, key=lambda x: x[1], reverse=True)
def check_values(old_doc_value, target):
    # Function to preprocess text by removing spaces, commas, and converting to lowercase
    def preprocess(text):
        return text.replace(" ", "").lower()
    
    # Preprocess the old_doc_value
    old_doc_processed = preprocess(old_doc_value)
    
    # Split the target into substrings and preprocess each one
    substrings = [preprocess(s) for s in re.split(r'[,\s]+', target) if s]
    # Check if all substrings are in the processed old_doc_value
    if all(substring in old_doc_processed for substring in substrings):
        return False 
    else:
        return True
    # If not all substrings are found, the function returns None by default

def find_changes(excel_value, sentences, model, threshold):
    changed_sentences = {}

    for target_text in excel_value:
        target = str(target_text[0]) + "," + str(target_text[1])
        
        # Find similar sentences
        similar_sentences = find_relevant_sentences(sentences, target, model, threshold)
        
        # If similar_sentences is empty, skip this loop iteration
        if not similar_sentences:
            print("No similar sentences found for target:", target)
            continue
        
        for sentence, similarity in similar_sentences:
            if check_values(sentence, target):
                continue
            try:
                sentence_index = sentences.index(sentence)
                if sentence_index not in changed_sentences:
                    changed_sentences[sentence_index] = []

                # Append the [target, target_text[2]] pair to the list associated with the sentence index
                changed_sentences[sentence_index].append([target, target_text[2]])
            except ValueError:
                print("valueError")
                # If the sentence is not found, skip to the next one
                continue
    return changed_sentences