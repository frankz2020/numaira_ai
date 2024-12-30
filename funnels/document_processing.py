import pandas as pd
from docx import Document
import re
import logging

# Suppress all logging from this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def read_docx(file_path):
    doc = Document(file_path)
    full_text = ' '.join(para.text for para in doc.paragraphs)
    text_segments = re.split(r'(?<!\d)\.(?!\d)', full_text)
    text_segments = [segment.strip() for segment in text_segments if segment.strip()]
    return text_segments

def format_value(value):
    """Format a numeric value with appropriate units."""
    try:
        num = float(value)
        abs_num = abs(num)
        if abs_num >= 1e9:
            formatted_num = num / 1e9
            unit = 'billion'
        elif abs_num >= 1e6:
            formatted_num = num / 1e6
            unit = 'million'
        elif abs_num >= 1e3:
            formatted_num = num / 1e3
            unit = 'thousand'
        else:
            formatted_num = num
            unit = ''
        formatted_num = round(formatted_num, 2)
        if unit:
            value_str = f"{formatted_num:.2f} {unit}"
        else:
            value_str = f"{formatted_num:.2f}"
    except (ValueError, TypeError):
        value_str = str(value)  # Keep as is if not a number
    return value_str

def normalize_number(value_str: str) -> float:
    """Convert a formatted number string back to float for comparison."""
    try:
        # Remove any currency symbols and whitespace
        clean_str = value_str.replace('$', '').strip()
        
        # Extract the numeric part and unit
        parts = clean_str.split()
        if len(parts) != 2:
            return float(clean_str)
            
        number, unit = parts
        number = float(number)
        
        # Convert to base unit based on suffix
        if 'billion' in unit.lower():
            return number
        elif 'million' in unit.lower():
            return number / 1000
        elif 'thousand' in unit.lower():
            return number / 1000000
        return number
    except (ValueError, IndexError):
        return None

def values_are_equal(value1: str, value2: str, tolerance: float = 0.01) -> bool:
    """Compare two formatted number strings for equality within tolerance."""
    num1 = normalize_number(value1)
    num2 = normalize_number(value2)
    
    if num1 is None or num2 is None:
        return value1 == value2
        
    # Compare with tolerance for floating point
    return abs(num1 - num2) < tolerance

def dict_to_list(dictionary):
    """Convert dictionary to list format with value comparison."""
    result = []
    for keys, value in dictionary.items():
        keys = list(keys)
        formatted_value = format_value(value)
        # Store original value for comparison
        result.append([keys, formatted_value, value])  # Include raw value for comparison
    return result

def excel_to_list(filename):
    try:
        logger.info(f"Reading Excel file: {filename}")
        df = pd.read_excel(filename, header=None, engine='openpyxl')
        
        if df.empty:
            raise ValueError("Excel file is empty")
            
        logger.info(f"Excel file shape: {df.shape}")
        
        # Extract row names and column names
        row_names = df.iloc[1:, 0].tolist()
        col_names = df.iloc[0, 1:].tolist()
        data = df.iloc[1:, 1:].values
        
        logger.info(f"Found {len(row_names)} rows and {len(col_names)} columns")
        
        if not row_names or not col_names:
            raise ValueError("Excel file must have both row names and column names")

        result = []
        for i, row_name in enumerate(row_names):
            for j, col_name in enumerate(col_names):
                if pd.isna(row_name) or pd.isna(col_name):
                    continue
                value = data[i, j]
                formatted_value = format_value(value)
                result.append([[str(row_name).strip(), str(col_name).strip()], formatted_value, value])  # Include raw value

        logger.info(f"Processed {len(result)} data points")
        return result
        
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        raise ValueError(f"Error processing Excel file: {str(e)}")

def embed_text(text, model):
    return model.encode(text)
