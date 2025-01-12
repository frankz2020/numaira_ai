import os
import logging
import asyncio
from typing import Dict, List, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
from RAG.format_mapping import format_maps
from RAG.similarity import find_changes
from funnels.extract import read_docx
from funnels.excel_utils import excel_to_list
from funnels.selection import selection
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from tqdm import tqdm
from utils.logging import setup_logging

# Set up logging
logger = setup_logging(level=logging.WARNING)

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
                print(f"  Confidence: {v[2]:.2%}")
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
        # Format maps with confidence scores
        modified_sentences = {}
        for key, values in changed_sentences.items():
            for value in values:
                formatted_text, confidence = asyncio.run(format_maps(
                    value[0][0],  # old_excel_value
                    sentences[key],  # old_doc_value
                    value[1],  # new_excel_value
                    value[2]  # confidence
                ))
                if formatted_text:
                    modified_sentences[key] = (formatted_text, confidence)
        
        # Return results with confidence scores
        results = []
        for key in changed_sentences:
            original = original_sentences[key]
            modified, confidence = modified_sentences[key]
            if original != modified:  # Only include if there's an actual change
                results.append((original, modified, confidence))
                print(f"\nAdded result:")
                print(f"Original: {original}")
                print(f"Modified: {modified}")
                print(f"Confidence: {confidence:.2%}")
        
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
                    
                    # Store formatting information from all runs
                    runs_info = []
                    if paragraph.runs:
                        for run in paragraph.runs:
                            runs_info.append({
                                'text': run.text,
                                'font_name': run.font.name,
                                'font_size': run.font.size,
                                'bold': run.font.bold,
                                'italic': run.font.italic,
                                'underline': run.font.underline,
                                'color': run.font.color and run.font.color.rgb,
                                'highlight_color': run.font.highlight_color,
                            })
                    
                    # Clear the paragraph
                    p = paragraph._p
                    p.clear_content()
                    
                    # Replace the text in the original paragraph
                    new_text = original_text.replace(orig_text, mod_text)
                    
                    # If we have formatting information, try to preserve it
                    if runs_info:
                        # Calculate the relative position of each run in the original text
                        total_length = sum(len(run_info['text']) for run_info in runs_info)
                        ratios = [len(run_info['text']) / total_length for run_info in runs_info]
                        
                        # Split the new text into parts based on the original ratios
                        current_pos = 0
                        for i, ratio in enumerate(ratios):
                            # Calculate the length of this run's text
                            length = int(len(new_text) * ratio)
                            if i == len(ratios) - 1:  # Last run gets the remainder
                                run_text = new_text[current_pos:]
                            else:
                                run_text = new_text[current_pos:current_pos + length]
                            current_pos += length
                            
                            # Create new run with preserved formatting
                            new_run = paragraph.add_run(run_text)
                            run_info = runs_info[i]
                            
                            # Apply stored formatting
                            if run_info['font_name']:
                                new_run.font.name = run_info['font_name']
                            if run_info['font_size']:
                                new_run.font.size = run_info['font_size']
                            if run_info['bold']:
                                new_run.font.bold = run_info['bold']
                            if run_info['italic']:
                                new_run.font.italic = run_info['italic']
                            if run_info['underline']:
                                new_run.font.underline = run_info['underline']
                            if run_info['color']:
                                new_run.font.color.rgb = run_info['color']
                            if run_info['highlight_color']:
                                new_run.font.highlight_color = run_info['highlight_color']
                    else:
                        # If no formatting info, just add the text as a single run
                        paragraph.add_run(new_text)
                    
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
    import sys
    if len(sys.argv) != 3:
        print("Usage: python main.py <docx_file> <excel_file>")
        sys.exit(1)
    
    docx_path = sys.argv[1]
    excel_path = sys.argv[2]
    results = process_files(docx_path, excel_path)
    for result in results:
        print(result)
