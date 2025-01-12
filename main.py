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
logger = setup_logging(level=logging.INFO)

# Configure progress bar
tqdm.pandas(desc="Processing", ncols=100, position=0, leave=True)

async def process_files(docx_path, excel_path, timeout: int = 30):
    """Process Word and Excel files to find and update matching content.
    
    Pipeline Steps:
    1. Load and parse input files
       - Word document text extraction
       - Excel data parsing
    2. Find matching sentences using semantic similarity
       - Sentence transformer embeddings
       - Cosine similarity matching
    3. Filter matches based on exact word presence
       - Metric name verification
       - Date/period matching
    4. Format changes using LLM
       - Pattern analysis
       - Text formatting
    5. Return results with confidence scores
    
    Args:
        docx_path: Path to Word document containing text to update
        excel_path: Path to Excel file containing new values
        timeout: Timeout in seconds for LLM calls (default: 30)
        
    Returns:
        List[Tuple[str, str, float]]: List of tuples containing:
            - original: Original sentence from document
            - modified: Updated sentence with new values
            - confidence: Update confidence score (0.0-1.0)
            
    Raises:
        ValueError: If no text found in Word document
        RuntimeError: If model initialization fails
        Exception: For file processing or LLM errors
        
    Example:
        >>> results = await process_files("report.docx", "updates.xlsx")
        >>> for orig, mod, conf in results:
        ...     print(f"Original: {orig}")
        ...     print(f"Modified: {mod}")
        ...     print(f"Confidence: {conf:.2%}")
    """
    try:
        print("\n=== Debug Information ===")
        
        # Initialize model (synchronous operation)
        print("\nInitializing sentence transformer model...")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Model initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Error initializing model: {str(e)}")
            
        # Load input files
        print("\nLoading input files...")
        excel_value = excel_to_list(excel_path)
        sentences = read_docx(docx_path)
        
        print(f"\nExcel values (first 2):")
        for val in excel_value[:2]:
            print(f"  {val}")
            
        print(f"\nSentences (first 2):")
        for i, sent in list(sentences.items())[:2]:
            print(f"  {i}: {sent}")
        
        if not sentences:
            raise ValueError("No text found in the Word document")
        
        # Convert sentences list to dict if it's not already
        if isinstance(sentences, list):
            sentences = {i: s for i, s in enumerate(sentences)}
        
        # Process the documents
        changed_sentences = {}
        threshold = 0.3
        
        changed_sentences, _, _ = await find_changes(
            excel_value=excel_value,
            sentences=sentences,
            model=model,
            threshold=threshold,
            timeout=timeout
        )
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
        for sentence_idx, values in changed_sentences.items():
            if not isinstance(sentence_idx, int):
                print(f"Warning: Invalid sentence index type: {type(sentence_idx)}")
                continue
                
            if sentence_idx not in sentences:
                print(f"Warning: Sentence index {sentence_idx} not found")
                continue
                
            for value in values:
                try:
                    if not isinstance(value, (list, tuple)) or len(value) != 3:
                        print(f"Warning: Invalid value format: {value}")
                        continue
                        
                    target_words, new_value, confidence = value
                    if not isinstance(target_words, list) or not target_words:
                        print(f"Warning: Invalid target words: {target_words}")
                        continue
                        
                    formatted_text, confidence = await format_maps(
                        old_excel_value=target_words[0],  # Original Excel value
                        old_doc_value=sentences[sentence_idx],  # Original document text
                        new_excel_value=new_value,  # Updated Excel value
                        confidence=confidence  # Confidence score from similarity matching
                    )
                    if formatted_text:
                        modified_sentences[sentence_idx] = (formatted_text, confidence)
                        print(f"\nFormatted text for sentence {sentence_idx}:")
                        print(f"Original: {sentences[sentence_idx]}")
                        print(f"Modified: {formatted_text}")
                        print(f"Confidence: {confidence:.2%}")
                except (IndexError, TypeError, ValueError) as e:
                    print(f"Warning: Error processing value: {str(e)}")
        
        # Return results with confidence scores
        results = []
        for key in changed_sentences:
            original = original_sentences[key]
            if key in modified_sentences:
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
                       "No matching sentences found in the documents. Please process a document first.",
                       0.0)]  # Include confidence score
        
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

async def main():
    """Main async entry point."""
    if len(sys.argv) != 3:
        print("Usage: python main.py <docx_file> <excel_file>")
        sys.exit(1)
    
    docx_path = sys.argv[1]
    excel_path = sys.argv[2]
    
    try:
        results = await process_files(docx_path, excel_path)
        for result in results:
            print(result)
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    import sys
    import asyncio
    asyncio.run(main())
