import pandas as pd

def excel_to_list(excel_path):
    """Convert Excel data to list format."""
    try:
        df = pd.read_excel(excel_path)
        # Convert all columns to string and concatenate
        values = []
        for col in df.columns:
            values.extend(df[col].astype(str).tolist())
        return [v.strip() for v in values if str(v).strip()]
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return []
