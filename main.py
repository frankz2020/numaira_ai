import os
import sys
import logging
from config import Config
from utils.logging import setup_logging
from utils.document_processing.processor import process_files, update_document

# Load configuration
config = Config()
for key, value in config.get_env_vars().items():
    os.environ[key] = value

# Set up logging
logger = setup_logging(level=logging.ERROR)  # Only show ERROR logs

def main():
    """Main entry point."""
    if len(sys.argv) != 3:
        print("Usage: python main.py <docx_file> <excel_file>")
        sys.exit(1)
    
    docx_path = sys.argv[1]
    excel_path = sys.argv[2]
    
    try:
        # Process files to find matches
        results = process_files(docx_path, excel_path)
        
        # Print results
        if results:
            print(f"\nFound {len(results)} changes:")
            for i, (orig, mod, conf) in enumerate(results, 1):
                print(f"\nChange {i} (confidence: {conf:.1%}):")
                print("Original:", orig)
                print("Modified:", mod)
        
        # Update document with changes
        output_path = docx_path.replace('.docx', '_updated.docx')
        update_document(docx_path, results, output_path)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
