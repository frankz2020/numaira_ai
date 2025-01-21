"""Evaluate accuracy of document update system."""
import os
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
from utils.document_processing.processor import process_files

def load_test_data(excel_path: str) -> Dict[str, Dict[str, float]]:
    """Load test data from Excel file.
    
    Args:
        excel_path: Path to Excel file
        
    Returns:
        Dict mapping row headers to their values
    """
    try:
        # Read Excel data
        df = pd.read_excel(excel_path)
        
        # Use first column as row headers
        row_header_col = df.columns[0]
        
        # Create mapping of row headers to values
        test_data = {}
        for _, row in df.iterrows():
            header = str(row[row_header_col]).strip()
            if not header or pd.isna(header):
                continue
                
            values = {}
            for col in df.columns[1:]:
                if not pd.isna(row[col]):
                    try:
                        values[str(col)] = float(row[col])
                    except (ValueError, TypeError):
                        continue
                        
            if values:
                test_data[header] = values
                
        return test_data
        
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return {}

def calculate_metrics(test_data: Dict[str, Dict[str, float]], predictions: List[Tuple[str, str, float]]) -> Dict:
    """Calculate accuracy metrics for predictions.
    
    Args:
        test_data: Ground truth data from Excel
        predictions: List of (original, modified, confidence) tuples
        
    Returns:
        Dict with accuracy metrics
    """
    true_positives = 0
    false_positives = 0
    false_negatives = len(test_data)  # Start with all as missed
    confidence_correlation = []
    
    for orig, mod, conf in predictions:
        # Find matching row in test data
        matched = False
        for header, values in test_data.items():
            # Compare values in modified text with Excel values
            matches = True
            for col, value in values.items():
                if str(value) not in mod:
                    matches = False
                    break
                    
            if matches:
                true_positives += 1
                false_negatives -= 1
                matched = True
                confidence_correlation.append((conf, 1.0))
                break
                
        if not matched:
            false_positives += 1
            confidence_correlation.append((conf, 0.0))
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate confidence correlation
    conf_correlation = np.corrcoef([x[0] for x in confidence_correlation], [x[1] for x in confidence_correlation])[0,1] if confidence_correlation else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confidence_correlation': conf_correlation,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

async def evaluate_test_files(test_dir='test_data'):
    """Evaluate accuracy using test files."""
    results = []
    
    # Get all test file pairs
    test_files = []
    for i in range(1, 6):  # We generated 5 test pairs
        docx_path = os.path.join(test_dir, f'test_{i}.docx')
        xlsx_path = os.path.join(test_dir, f'test_{i}.xlsx')
        if os.path.exists(docx_path) and os.path.exists(xlsx_path):
            test_files.append((docx_path, xlsx_path))
    
    print(f"\nEvaluating {len(test_files)} test file pairs...")
    
    for docx_path, xlsx_path in tqdm(test_files):
        try:
            # Load test data
            test_data = load_test_data(xlsx_path)
            
            # Process files and get predictions
            predictions = await process_files(docx_path, xlsx_path)
            
            # Calculate metrics
            metrics = calculate_metrics(test_data, predictions)
            metrics['test_pair'] = os.path.basename(docx_path)
            results.append(metrics)
            
        except Exception as e:
            print(f"\nError processing {docx_path}: {str(e)}")
            continue
    
    # Calculate average metrics
    avg_metrics = {
        'precision': np.mean([r['precision'] for r in results]),
        'recall': np.mean([r['recall'] for r in results]),
        'f1': np.mean([r['f1'] for r in results]),
        'confidence_correlation': np.mean([r['confidence_correlation'] for r in results])
    }
    
    print("\nResults by test pair:")
    for r in results:
        print(f"\n{r['test_pair']}:")
        print(f"Precision: {r['precision']:.2%}")
        print(f"Recall: {r['recall']:.2%}")
        print(f"F1 Score: {r['f1']:.2%}")
        print(f"Confidence Correlation: {r['confidence_correlation']:.3f}")
        print(f"True Positives: {r['true_positives']}")
        print(f"False Positives: {r['false_positives']}")
        print(f"False Negatives: {r['false_negatives']}")
    
    print("\nAverage Metrics:")
    print(f"Precision: {avg_metrics['precision']:.2%}")
    print(f"Recall: {avg_metrics['recall']:.2%}")
    print(f"F1 Score: {avg_metrics['f1']:.2%}")
    print(f"Confidence Correlation: {avg_metrics['confidence_correlation']:.3f}")
    
    return results, avg_metrics

if __name__ == '__main__':
    import asyncio
    asyncio.run(evaluate_test_files())
