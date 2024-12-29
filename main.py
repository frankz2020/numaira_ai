import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import asyncio
import logging
from sentence_transformers import SentenceTransformer
from RAG import format_maps, find_changes
from funnels import read_docx, excel_to_list, selection
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from tqdm import tqdm

# Set up logging - only show warnings and errors
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configure progress bar
tqdm.pandas(desc="Processing", ncols=100, position=0, leave=True)

def process_files(docx_path, excel_path):
    try:
        print("\n=== Debug Information ===")
        
        model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
        excel_value = excel_to_list(excel_path)
        sentences = read_docx(docx_path)
        
        print(f"\nExcel values (first 2):")
        for val in excel_value[:2]:
            print(f"  {val}")
            
        print(f"\nSentences (first 2):")
        for i, sent in enumerate(sentences[:2]):
            print(f"  {i}: {sent}")
        
        if not sentences:
            raise ValueError("No text found in the Word document")
        
        # Convert sentences list to dict if it's not already
        if isinstance(sentences, list):
            sentences = {i: s for i, s in enumerate(sentences)}
        
        # Process the documents
        changed_sentences = {}
        threshold = 0.3
        
        changed_sentences, _, _ = find_changes(excel_value, sentences, model, threshold)
        print(f"\nFound changes in sentences:")
        for key, values in changed_sentences.items():
            print(f"\nSentence {key}: {sentences[key]}")
            print("Changes to make:")
            for v in values:  # Show all changes
                print(f"  Target: {v[0]}")
                print(f"  New value: {v[1]}")
                print()
        
        # Store original sentences before modification
        original_sentences = {}
        for key in changed_sentences:
            original_sentences[key] = sentences[key]
        
        # Filter irrelevant sentences
        print("\nBefore filtering:", len(changed_sentences), "sentences")
        changed_sentences = selection(changed_sentences, sentences)
        print("After filtering:", len(changed_sentences), "sentences")
        
        # Format maps
        print("\nApplying changes...")
        modified_sentences = asyncio.run(format_maps(changed_sentences, sentences.copy()))
        
        # Return results
        results = []
        for key in changed_sentences:
            original = original_sentences[key]
            modified = modified_sentences[key]
            if original != modified:  # Only include if there's an actual change
                results.append((original, modified))
                print(f"\nAdded result:")
                print(f"Original: {original}")
                print(f"Modified: {modified}")
        
        print(f"\nFinal results count: {len(results)}")
        
        if not results:
            results = [("No matching sentences found.", 
                       "No matching sentences found in the documents. Please process a document first.")]
        
        print("\n=== End Debug Information ===\n")
        return results
    except Exception as e:
        logger.error(f"Error in process_files: {str(e)}", exc_info=True)
        raise Exception(f"Error processing files: {str(e)}")

def update_document(docx_path, results, output_path):
    """Update the document with the modified sentences and save to a new file."""
    try:
        # Load the original document
        doc = Document(docx_path)
        
        # Create a mapping of original to modified sentences
        updates = {result[0].strip(): result[1].strip() for result in results if isinstance(result, tuple)}
        print(f"\nUpdates to make:")
        for orig, mod in updates.items():
            print(f"Original: {orig}")
            print(f"Modified: {mod}\n")
        
        # Keep track of changes made
        changes_made = 0
        
        # Update paragraphs
        for paragraph in doc.paragraphs:
            # Get the text without extra whitespace
            original_text = paragraph.text.strip()
            
            # Check if this paragraph needs to be updated
            for orig_text, mod_text in updates.items():
                if orig_text in original_text:  # Changed from exact match to contains
                    print(f"\nUpdating paragraph:")
                    print(f"From: {original_text}")
                    print(f"To: {mod_text}")
                    
                    # Store formatting information
                    font_name = None
                    font_size = None
                    is_bold = None
                    is_italic = None
                    
                    if paragraph.runs:
                        first_run = paragraph.runs[0]
                        font_name = first_run.font.name
                        font_size = first_run.font.size
                        is_bold = first_run.font.bold
                        is_italic = first_run.font.italic
                    
                    # Clear the paragraph
                    p = paragraph._p
                    p.clear_content()
                    
                    # Add new run with modified text
                    new_run = paragraph.add_run(mod_text)
                    
                    # Apply stored formatting
                    if font_name:
                        new_run.font.name = font_name
                    if font_size:
                        new_run.font.size = font_size
                    if is_bold:
                        new_run.font.bold = is_bold
                    if is_italic:
                        new_run.font.italic = is_italic
                    
                    changes_made += 1
                    print(f"Successfully updated paragraph")
                    break  # Only apply first matching update
        
        print(f"\nMade {changes_made} changes to the document")
        
        # Save the modified document
        doc.save(output_path)
        print(f"Saved updated document to: {output_path}")
        
        return changes_made
        
    except Exception as e:
        logger.error(f"Error updating document: {str(e)}", exc_info=True)
        raise Exception(f"Failed to update document: {str(e)}")

if __name__ == '__main__':
    # For command line usage
    file_path = 'test.docx'
    filename = 'test_new_rag.xlsx'
    results = process_files(file_path, filename)
    for result in results:
        print(result)
