import re
import time
from sklearn.metrics.pairwise import cosine_similarity
from funnels.document_processing import embed_text


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


def find_changes(excel_value, sentences, model, threshold):
    changed_sentences = {}
    totel_find_sentences_time = 0
    total_input_change_time = 0
    for target_text in excel_value:
        one_find_sentences_time = time.time()
        target = ",".join(target_text[0])
        similar_sentences = find_relevant_sentences(sentences, target, model, threshold)
        one_find_sentences_time = time.time() - one_find_sentences_time
        totel_find_sentences_time = totel_find_sentences_time + one_find_sentences_time

        if not similar_sentences:
            print("No similar sentences found for target:", target)
            continue

        # 该步骤不耗时
        one_input_change_time = time.time()
        for sentence, similarity in similar_sentences:
            if check_values(sentence, target):
                continue
            try:
                sentence_index = sentences.index(sentence)
                if sentence_index not in changed_sentences:
                    changed_sentences[sentence_index] = []

                changed_sentences[sentence_index].append([target, target_text[1]])
            except ValueError:
                print("valueError")
                continue
        one_input_change_time = time.time() - one_input_change_time
        total_input_change_time = total_input_change_time + one_input_change_time

    return changed_sentences, totel_find_sentences_time, total_input_change_time
