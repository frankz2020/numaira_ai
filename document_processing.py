import pandas as pd
from docx import Document
import re
def read_docx(file_path):
    doc = Document(file_path)
    full_text = ' '.join(para.text for para in doc.paragraphs)
    text_segments = re.split(r'(?<!\d)\.(?!\d)', full_text)
    text_segments = [segment.strip() for segment in text_segments if segment.strip()]
    return text_segments

def excel_to_list(filename):
    # to be coordinated
    df = pd.read_excel(filename, header=None)
    row_names = df.iloc[1:, 0].tolist()
    col_names = df.iloc[0, 1:].tolist()
    data = df.iloc[1:, 1:].values
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
            formatted_num = round(formatted_num, 2)
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