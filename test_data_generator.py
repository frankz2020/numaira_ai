"""Generate test data for document analysis system."""
import random
from datetime import datetime, timedelta
from docx import Document
import pandas as pd
import os

def generate_random_amount():
    """Generate a random dollar amount in millions."""
    return random.randint(100, 9999)

def generate_date():
    """Generate a random date within the last year."""
    today = datetime.now()
    days_ago = random.randint(0, 365)
    date = today - timedelta(days=days_ago)
    return date.strftime("%B %d, %Y")

def generate_period():
    """Generate a random period (3 or 6 months)."""
    return random.choice(["three", "six"])

def generate_financial_sentence():
    """Generate a realistic financial sentence."""
    templates = [
        "For the {period} months ended {date}, revenue was ${amount} million.",
        "Net income for the {period}-month period ended {date} was ${amount} million.",
        "Operating income for the {period} months ended {date} increased to ${amount} million.",
        "For the {period}-month period ended {date}, EBITDA was ${amount} million.",
    ]
    template = random.choice(templates)
    return template.format(
        period=generate_period(),
        date=generate_date(),
        amount=generate_random_amount()
    )

def extract_amounts(text):
    """Extract dollar amounts from text."""
    import re
    amounts = re.findall(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion)?', text)
    return [int(amount.replace(',', '')) for amount in amounts]

def generate_docx(path="test.docx", num_sentences=10):
    """Generate a test Word document with financial statements."""
    doc = Document()
    sentences = []
    
    # Add title
    doc.add_heading('Financial Report', 0)
    doc.add_paragraph()
    
    # Generate sentences and store them
    for _ in range(num_sentences):
        sentence = generate_financial_sentence()
        sentences.append(sentence)
        doc.add_paragraph(sentence)
    
    # Save document
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc.save(path)
    return sentences

def generate_xlsx(path="test.xlsx", sentences=None, variation_factor=0.1):
    """Generate a test Excel file with financial data.
    
    Args:
        path: Path to save Excel file
        sentences: List of sentences to extract amounts from
        variation_factor: How much to vary the amounts (0.1 = 10% variation)
    """
    if not sentences:
        return
    
    # Extract amounts and dates from sentences
    data = []
    for sentence in sentences:
        amounts = extract_amounts(sentence)
        if amounts:
            # Add some variation to create "new" numbers
            varied_amount = amounts[0] * (1 + random.uniform(-variation_factor, variation_factor))
            data.append({
                'Original Amount': amounts[0],
                'Updated Amount': round(varied_amount)
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False)

def generate_test_files(output_dir="test_data", num_files=5, sentences_per_file=10):
    """Generate multiple test file pairs."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        docx_path = os.path.join(output_dir, f"test_{i+1}.docx")
        xlsx_path = os.path.join(output_dir, f"test_{i+1}.xlsx")
        
        # Generate docx and get sentences
        sentences = generate_docx(docx_path, sentences_per_file)
        
        # Generate corresponding xlsx with variations
        generate_xlsx(xlsx_path, sentences)
        
        print(f"Generated test pair {i+1}: {docx_path}, {xlsx_path}")

if __name__ == '__main__':
    # Generate test files in test_data directory
    generate_test_files()
