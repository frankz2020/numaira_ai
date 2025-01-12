import docx

def read_docx(word_file):
    """Read a Word document and return a dictionary of sentences."""
    doc = docx.Document(word_file)
    sentences = {}
    idx = 0
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            sentences[idx] = paragraph.text.strip()
            idx += 1
    return sentences
