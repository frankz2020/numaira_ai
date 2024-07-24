import docx

def extract_text_from_word(word_file):
    doc = docx.Document(word_file)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)
