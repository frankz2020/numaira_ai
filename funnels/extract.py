import docx
import logging

logger = logging.getLogger(__name__)

def read_docx(word_file):
    """Read a Word document and return a dictionary of sentences."""
    logger.info(f"Reading Word document: {word_file}")
    doc = docx.Document(word_file)
    
    sentences = {}
    idx = 0
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text:
            logger.info(f"Found sentence {idx}: {text}")
            sentences[idx] = text
            idx += 1
        else:
            logger.debug("Skipping empty paragraph")
            
    logger.info(f"Total sentences extracted: {len(sentences)}")
    return sentences
