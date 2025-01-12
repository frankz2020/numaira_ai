import pandas as pd
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

def parse_period_column(col_name: str) -> Optional[Dict[str, Any]]:
    """Parse period information from column name.
    
    Args:
        col_name: Column name to parse
        
    Returns:
        Dict with period info or None if not a period column
        {
            'period': 'three'|'six',
            'date': datetime object,
            'original': original column name
        }
    """
    col_lower = str(col_name).lower().strip()
    
    # Match period (three/3/six/6 months)
    period_match = re.search(r'(?:three|3|six|6)\s*(?:months?|mo\.?|mos\.?|quarter)', col_lower)
    if not period_match:
        return None
        
    # Determine period type
    period = 'three' if any(x in period_match.group().lower() for x in ['three', '3']) else 'six'
    
    # Try to extract date
    date_patterns = [
        r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2}\s*,?\s*\d{4}',
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{4}-\d{2}-\d{2}'
    ]
    
    date_obj = None
    for pattern in date_patterns:
        date_match = re.search(pattern, col_lower)
        if date_match:
            try:
                date_str = date_match.group()
                # Try different date formats
                for fmt in ['%B %d, %Y', '%b %d, %Y', '%m/%d/%Y', '%Y-%m-%d']:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
                if date_obj:
                    break
            except ValueError:
                continue
    
    return {
        'period': period,
        'date': date_obj,
        'original': col_name
    }

def excel_to_list(excel_path):
    """Extract financial metrics from Excel file.
    
    Returns structured data with definition names and their corresponding values
    for three and six month periods.
    
    Returns:
        List[Dict]: List of dictionaries with structure:
            {
                "definition": str,  # e.g., "Total revenues"
                "values": List[str],  # e.g., ["24.93", "48.26"] (in billions)
                "periods": List[str]  # e.g., ["three months", "six months"]
            }
    """
    try:
        # Read Excel file and clean up column names
        df = pd.read_excel(excel_path)
        print("\nReading Excel file structure...")
        
        # Identify metric column (usually 'Unnamed: 0' or first column)
        metric_col = 'Unnamed: 0'
        if metric_col not in df.columns:
            metric_col = df.columns[0]
        print(f"Using metric column: {metric_col}")
        
        # Find period columns using flexible pattern matching
        print("\nAnalyzing column structure...")
        period_cols = []
        
        # Parse all columns for period information
        for col in df.columns:
            period_info = parse_period_column(col)
            if period_info:
                period_cols.append(period_info)
                print(f"Found {period_info['period']}-month column: {period_info['original']}")
                if period_info['date']:
                    print(f"  Date: {period_info['date'].strftime('%B %d, %Y')}")
        
        if len(period_cols) < 2:
            print("\nWarning: Could not find both three-month and six-month columns")
            print("Available columns:", df.columns.tolist())
            raise ValueError("Expected at least two period columns (three months and six months)")
            
        # Sort by period type to ensure consistent order
        period_cols.sort(key=lambda x: x['period'])
        
        # Get most recent columns for each period
        three_month_cols = [col for col in period_cols if col['period'] == 'three']
        six_month_cols = [col for col in period_cols if col['period'] == 'six']
        
        if not three_month_cols or not six_month_cols:
            raise ValueError("Missing required period columns")
            
        # Use most recent date if available, otherwise first column
        three_month_col = max(three_month_cols, key=lambda x: x['date'] if x['date'] else datetime.min)['original']
        six_month_col = max(six_month_cols, key=lambda x: x['date'] if x['date'] else datetime.min)['original']
        
        print(f"\nUsing period columns:\n  Three-month: {three_month_col}\n  Six-month: {six_month_col}")
        
        # Extract target metrics with structured data
        metrics_data = []
        
        # Display DataFrame structure for debugging
        print("\nDataFrame Structure:")
        print(df[[metric_col, three_month_col, six_month_col]].head())
        
        # Define target metrics and their variations with period-specific forms
        target_metrics = {
            'Total revenues': {
                'variations': ['total revenues', 'total revenue', 'revenues', 'revenue'],
                'period_forms': {
                    'three_month': 'Total revenues (3 months)',
                    'six_month': 'Total revenues (6 months)'
                }
            },
            'Net income attributable to common stockholders': {
                'variations': [
                    'net income attributable to common stockholders',
                    'net income',
                    'our net income attributable to common stockholders'
                ],
                'period_forms': {
                    'three_month': 'Net income attributable to common stockholders (3 months)',
                    'six_month': 'Net income attributable to common stockholders (6 months)'
                }
            },
            'Operating income': {
                'variations': ['operating income', 'operating profit', 'income from operations'],
                'period_forms': {
                    'three_month': 'Operating income (3 months)',
                    'six_month': 'Operating income (6 months)'
                }
            },
            'EBITDA': {
                'variations': ['ebitda', 'earnings before interest taxes depreciation and amortization'],
                'period_forms': {
                    'three_month': 'EBITDA (3 months)',
                    'six_month': 'EBITDA (6 months)'
                }
            }
        }
        
        # Process each target metric using leftmost column
        print("\nProcessing metrics from leftmost column...")
        for definition_name, metric_info in target_metrics.items():
            # Look for exact match or any variation in leftmost column
            variations = [definition_name.lower()] + [v.lower() for v in metric_info['variations']]
            
            # Clean and convert metric column values to lowercase for matching
            df[metric_col] = df[metric_col].astype(str).str.strip().str.lower()
            
            # Try exact match first
            matches = df[df[metric_col] == definition_name.lower()]
            
            # If no exact match, try variations
            if matches.empty:
                matches = df[df[metric_col].isin(variations)]
            
            if not matches.empty:
                # Get the first matching row
                row = matches.iloc[0]
                original_metric = row[metric_col]
                print(f"\nFound metric: {definition_name}")
                print(f"Matched variation: {original_metric}")
                
                try:
                    # Extract and convert values to billions with exact formatting
                    three_month_val = pd.to_numeric(row[three_month_col], errors='coerce') / 1e9
                    six_month_val = pd.to_numeric(row[six_month_col], errors='coerce') / 1e9
                    
                    # Verify values are valid
                    if pd.isna(three_month_val) or pd.isna(six_month_val):
                        print(f"Warning: Invalid values for {definition_name}")
                        continue
                    
                    # Format with exactly 2 decimal places
                    formatted_values = [f"{three_month_val:.2f}", f"{six_month_val:.2f}"]
                    
                    print(f"  {metric_info['period_forms']['three_month']}: ${formatted_values[0]} billion")
                    print(f"  {metric_info['period_forms']['six_month']}: ${formatted_values[1]} billion")
                    
                    metrics_data.append({
                        "definition": definition_name,
                        "values": formatted_values,
                        "periods": ["three months", "six months"],
                        "variations": metric_info['variations'],
                        "period_forms": metric_info['period_forms']
                    })
                    print(f"✓ Added {definition_name} with values: ${formatted_values[0]}B, ${formatted_values[1]}B")
                except (ValueError, AttributeError, IndexError) as e:
                    print(f"Error processing {definition_name}: {str(e)}")
                    continue
        
        return metrics_data
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return []
