import spacy
import re

# Load the spaCy model for "Chinese" only
nlp = spacy.load("zh_core_web_sm")

def split_sentence(doc):
    chunks = []
    current_chunk = []

    for token in doc:
        # This match save the format of any large number seperated by ',' each three digits.
        if re.match(r'^\d{1,3}(,\d{3})*$', token.text):
            current_chunk.append(token.text)
        else:
            current_chunk.append(token.text)
            if len(token.text) > 1:
                chunks.append(''.join(current_chunk))
                current_chunk = []

    if current_chunk:
        chunks.append(''.join(current_chunk))

    return chunks

def split_text_into_clips(text):
    # Process the text with spaCy
    doc = nlp(text)
    sentences = split_sentence(doc)
    clips = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Save only clips that contain numbers to further filter
        if re.search(r'\d', sentence):
            clips.append(sentence)
    # Remove any empty clips
    clips = [clip.strip() for clip in clips if clip.strip()]
    return clips

