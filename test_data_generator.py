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

def generate_financial_sentence(metric=None, period=None, amount=None):
    """Generate a realistic financial sentence.
    
    Args:
        metric: Specific metric to use (Revenue, Net Income, etc.)
        period: Specific period to use (three/six months)
        amount: Specific amount to use (if None, generates random amount)
    """
    templates = {
        'Revenue': "For the {period} months ended {date}, revenue was ${amount:.2f} million",
        'Net Income': "Net income for the {period}-month period ended {date} was ${amount:.2f} million",
        'Operating Income': "Operating income for the {period} months ended {date} increased to ${amount:.2f} million",
        'EBITDA': "For the {period}-month period ended {date}, EBITDA was ${amount:.2f} million"
    }
    
    # Use provided metric or choose randomly
    if not metric:
        metric = random.choice(list(templates.keys()))
    template = templates[metric]
    
    # Use provided period or choose randomly
    if not period:
        period = generate_period()
        
    # Use provided amount or generate random
    if amount is None:
        amount = generate_random_amount()
        
    return template.format(
        period=period,
        date=generate_date(),
        amount=amount
    ), metric, amount

def extract_amounts(text):
    """Extract dollar amounts from text."""
    import re
    amounts = re.findall(r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:million|billion)?', text)
    return [int(amount.replace(',', '')) for amount in amounts]

def generate_docx(path="test.docx"):
    """Generate a test Word document with financial statements."""
    doc = Document()
    data = []  # Store (sentence, metric, period, amount) tuples
    
    # Add title
    doc.add_heading('Financial Report', 0)
    doc.add_paragraph()
    
    # Generate one sentence for each metric and period combination
    metrics = ['Revenue', 'Net Income', 'Operating Income', 'EBITDA']
    periods = ['three', 'six']
    
    for metric in metrics:
        for period in periods:
            amount = generate_random_amount()
            sentence, _, _ = generate_financial_sentence(metric=metric, period=period, amount=amount)
            data.append((sentence, metric, period, amount))
            doc.add_paragraph(sentence)
    
    # Save document
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc.save(path)
    return data

def generate_xlsx(path="test.xlsx", data=None, variation_factor=0.1):
    """Generate a test Excel file with financial data.
    
    Args:
        path: Path to save Excel file
        data: List of (sentence, metric, period, amount) tuples
        variation_factor: How much to vary the amounts (0.1 = 10% variation)
    """
    if not data:
        return
        
    # Create DataFrame with proper structure
    records = []
    for sentence, metric, period, amount in data:
        # Convert period to standard format
        period_str = "3 Months" if period == "three" else "6 Months"
        
        # Add variation to create "new" numbers
        varied_amount = amount * (1 + random.uniform(-variation_factor, variation_factor))
        
        records.append({
            'Metric': metric,
            'Period': period_str,
            'Original Value': f"{amount:.2f}",  # Keep 2 decimal places
            'Updated Value': f"{varied_amount:.2f}"  # Keep 2 decimal places
        })
    
    # Define metric order
    metric_order = ['Revenue', 'Operating Income', 'Net Income', 'EBITDA']
    
    # Create DataFrame with proper structure and index
    columns = []
    for period in ['3 Months', '6 Months']:
        columns.extend([f'Original Value ({period})', f'Updated Value ({period})'])
    
    # Initialize DataFrame with zeros and proper index
    result_df = pd.DataFrame(0.0, 
                           index=pd.Index(metric_order, name='Metric'),
                           columns=columns)
    
    # Fill values with proper decimal formatting
    for record in records:
        metric = record['Metric']
        period = record['Period']
        orig_val = float(record['Original Value'])
        updated_val = float(record['Updated Value'])
        result_df.at[metric, f'Original Value ({period})'] = orig_val
        result_df.at[metric, f'Updated Value ({period})'] = updated_val
    
    # Ensure consistent decimal places
    result_df = result_df.round(2)
    
    # Ensure all values are properly formatted
    for col in result_df.columns:
        result_df[col] = pd.to_numeric(result_df[col], errors='coerce').round(2)
    
    # Save with proper formatting
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        # Save with proper index and number formatting
        # Reset index to make Metric a regular column, then set it as index again
        # This ensures proper display of metric names
        temp_df = result_df.reset_index()
        temp_df['Metric'] = pd.Categorical(temp_df['Metric'], categories=metric_order, ordered=True)
        temp_df = temp_df.set_index('Metric')
        
        # Convert to string format with 2 decimal places
        for col in temp_df.columns:
            temp_df[col] = temp_df[col].apply(lambda x: '{:.2f}'.format(x))
            
        # Save with proper formatting
        temp_df.to_excel(
            writer,
            index=True,
            index_label='Metric'
        )
        
        # Access the worksheet to adjust column widths and format
        worksheet = writer.sheets['Sheet1']
        worksheet.column_dimensions['A'].width = 20  # Metric column
        for col in range(1, len(result_df.columns) + 2):
            col_letter = chr(65 + col)
            worksheet.column_dimensions[col_letter].width = 15  # Data columns

def generate_test_files(output_dir="test_data", num_files=5):
    """Generate multiple test file pairs."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_files):
        docx_path = os.path.join(output_dir, f"test_{i+1}.docx")
        xlsx_path = os.path.join(output_dir, f"test_{i+1}.xlsx")
        
        # Generate docx and get data
        data = generate_docx(docx_path)
        
        # Generate corresponding xlsx with variations
        generate_xlsx(xlsx_path, data)
        
        print(f"Generated test pair {i+1}: {docx_path}, {xlsx_path}")

if __name__ == '__main__':
    # Generate test files in test_data directory
    generate_test_files()
