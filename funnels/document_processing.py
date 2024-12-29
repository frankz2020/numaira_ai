import pandas as pd
from docx import Document
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_docx(file_path):
    doc = Document(file_path)
    full_text = ' '.join(para.text for para in doc.paragraphs)
    text_segments = re.split(r'(?<!\d)\.(?!\d)', full_text)
    text_segments = [segment.strip() for segment in text_segments if segment.strip()]
    return text_segments

def dict_to_list(dictionary):
    dictionary = {
        ("Total revenues", "Six Months Ended June 30, 2023"): "48256000000",
        ("Total revenues", "Three Months Ended June 30, 2023"): "24927000000",
        ("Net income attributable to common stockholders", "Three Months Ended June 30, 2023"): "2703000000",
        ("Net income attributable to common stockholders", "Six Months Ended June 30, 2023"): "74873000000",
        ("Net income", "Six Months Ended June 30, 2023"): "5153000000",
        ("Net income", "Three Months Ended June 30, 2023"): "5153000000",
    }

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

    for keys,value in dictionary.items():
        keys=list(keys)
        value=format_value(value)
        result.append([keys, value])

    return result




def excel_to_list(filename):
    try:
        logger.info(f"Reading Excel file: {filename}")
        df = pd.read_excel(filename, header=None)
        
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
                if pd.isna(row_name) or pd.isna(col_name):
                    continue
                value = data[i, j]
                value_str = format_value(value)
                result.append([[str(row_name).strip(), str(col_name).strip()], value_str])

        logger.info(f"Processed {len(result)} data points")
        return result
        
    except Exception as e:
        logger.error(f"Error processing Excel file: {str(e)}")
        raise ValueError(f"Error processing Excel file: {str(e)}")


def embed_text(text, model):
    return model.encode(text)
