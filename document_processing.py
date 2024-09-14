import pandas as pd
from docx import Document
import re
def read_docx(file_path):
    # Load the Word document
    doc = Document(file_path)
    
    # Concatenate all paragraphs into one string
    full_text = ' '.join(para.text for para in doc.paragraphs)
    
    # Use regex to split on periods not between digits
    text_segments = re.split(r'(?<!\d)\.(?!\d)', full_text)
    
    # Remove any leading/trailing whitespace from each segment
    text_segments = [segment.strip() for segment in text_segments if segment.strip()]
    
    return text_segments

def excel_to_list(filename):
    # Read the Excel file without setting headers or index columns
    df = pd.read_excel(filename, header=None)

    # Extract row names (from the first column, excluding the first cell)
    row_names = df.iloc[1:, 0].tolist()

    # Extract column names (from the first row, excluding the first cell)
    col_names = df.iloc[0, 1:].tolist()

    # Extract the data excluding the first row and first column
    data = df.iloc[1:, 1:].values

    # Function to convert numeric value to string with unit
    def format_value(value):
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

            # Round the formatted number to 2 decimal places
            formatted_num = round(formatted_num, 2)

            # Format the number to always show two decimal places
            if unit:
                value_str = f"{formatted_num:.2f} {unit}"
            else:
                value_str = f"{formatted_num:.2f}"
        except (ValueError, TypeError):
            value_str = str(value)  # Keep as is if not a number
        return value_str

    result = []

    for i, row_name in enumerate(row_names):
        for j, col_name in enumerate(col_names):
            value = data[i, j]
            value_str = format_value(value)
            result.append([row_name, col_name, value_str])

    return result

def embed_text(text, model):
    return model.encode(text)