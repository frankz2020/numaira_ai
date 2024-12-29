import re
import time
import logging
from sklearn.metrics.pairwise import cosine_similarity
from funnels.document_processing import embed_text
from tqdm import tqdm

logger = logging.getLogger(__name__)

def find_relevant_sentences(sentences, target_text, model, threshold):
    target_embedding = embed_text(target_text, model)
    sentence_embeddings = [embed_text(sentence, model) for sentence in sentences]
    similarities = cosine_similarity([target_embedding], sentence_embeddings)[0]
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

    old_doc_processed = preprocess(old_doc_value)

    # Split the target into substrings and preprocess each one
    substrings = [preprocess(s) for s in re.split(r'[,\s]+', target) if s]
    if all(substring in old_doc_processed for substring in substrings):
        return False
    else:
        return True


def find_changes(excel_value, sentences, model, threshold=0.3):
    """Find changes in sentences based on Excel values."""
    changed_sentences = {}
    total_find_sentences_time = 0
    total_input_change_time = 0

    # Create embeddings for all sentences at once
    sentence_list = list(sentences.values()) if isinstance(sentences, dict) else sentences
    sentence_embeddings = model.encode(sentence_list, show_progress_bar=False)
    
    # Group Excel values by their categories
    value_groups = {}
    for value in excel_value:
        if isinstance(value[0], list):
            category = value[0][0]  # First element is the category
            if category not in value_groups:
                value_groups[category] = []
            value_groups[category].append(value)
    
    # Process each category of values
    with tqdm(value_groups.items(), desc="Processing Excel values", ncols=100, position=0) as pbar:
        for category, values in pbar:
            # Create target text that includes the category
            target = f"{category} for the three and six months ended June 30, 2023"
            
            # Create embedding for target
            target_embedding = model.encode([target], show_progress_bar=False)[0]
            
            # Calculate similarities with all sentences at once
            similarities = cosine_similarity([target_embedding], sentence_embeddings)[0]
            
            # Find matches above threshold
            matches = [(i, sim) for i, sim in enumerate(similarities) if sim > threshold]
            
            if matches:
                max_sim_idx, max_sim = max(matches, key=lambda x: x[1])
                # Add all values for this category to the sentence
                if max_sim_idx in changed_sentences:
                    changed_sentences[max_sim_idx].extend(values)
                else:
                    changed_sentences[max_sim_idx] = values
    
    return changed_sentences, total_find_sentences_time, total_input_change_time
