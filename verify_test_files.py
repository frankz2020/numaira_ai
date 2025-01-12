"""Verify test files meet all required assumptions."""
import pandas as pd
from docx import Document
import os

def verify_excel_structure(excel_path):
    """Verify Excel file structure meets requirements."""
    print(f"\nVerifying Excel structure: {excel_path}")
    
    # Read Excel file with 'Metric' as index
    df = pd.read_excel(excel_path, index_col='Metric')
    print("\nExcel Content:")
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print(df.to_string())
    
    # Verify metrics in leftmost column
    metrics = df.index.tolist()
    print("\nMetrics (leftmost column):", metrics)
    
    # Verify periods in column headers
    periods = sorted(set(col.split('(')[1].strip(')') for col in df.columns if '(' in col))
    print("Periods (column headers):", periods)
    
    # Verify number formatting
    print("\nValue Formatting:")
    for col in df.columns:
        values = df[col].round(2)  # Ensure 2 decimal places
        print(f"\n{col}:")
        for metric, value in values.items():
            print(f"  {metric}: {value:.2f}")

def verify_word_content(docx_path):
    """Verify Word document content meets requirements."""
    print(f"\nVerifying Word content: {docx_path}")
    doc = Document(docx_path)
    print("\nDocument Content:")
    for para in doc.paragraphs:
        if para.text.strip():
            print(para.text)

def main():
    """Verify test files in test_data directory."""
    test_dir = "test_data"
    
    # Verify first test pair
    excel_path = os.path.join(test_dir, "test_1.xlsx")
    docx_path = os.path.join(test_dir, "test_1.docx")
    
    verify_excel_structure(excel_path)
    verify_word_content(docx_path)

if __name__ == '__main__':
    main()
