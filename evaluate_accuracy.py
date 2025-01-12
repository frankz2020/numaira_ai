"""Evaluate accuracy of document update system."""
import os
import pandas as pd
from main import process_files
import numpy as np
from tqdm import tqdm

def load_test_data(excel_path):
    """Load test data from Excel file and extract original and updated amounts."""
    # Read Excel file with proper column names
    df = pd.read_excel(excel_path)
    
    # Get columns containing Original/Updated values
    orig_cols = [col for col in df.columns if 'Original Value' in col]
    update_cols = [col for col in df.columns if 'Updated Value' in col]
    
    # Create mapping of original to updated values
    changes = {}
    for orig_col, update_col in zip(orig_cols, update_cols):
        # Convert values to float and round to 2 decimal places
        orig_values = df[orig_col].astype(float).round(2)
        update_values = df[update_col].astype(float).round(2)
        
        # Add to changes dict, filtering out NaN values
        valid_pairs = zip(orig_values.dropna(), update_values.dropna())
        changes.update(dict(valid_pairs))
    
    return changes

def calculate_metrics(actual_changes, predicted_changes, confidence_scores):
    """Calculate accuracy metrics."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    confidence_correlation = []
    
    # Track which actual changes were found
    found_changes = set()
    
    print("\nDebug: Processing predictions...")
    for pred, conf in zip(predicted_changes, confidence_scores):
        # Case-insensitive text comparison
        pred = pred.lower()
        print(f"\nPredicted text: {pred}")
        
        # Extract financial values with flexible format
        import re
        financial_numbers = []
        
        # Extract numbers with flexible format
        patterns = [
            r'\$\s*([\d,]+(?:\.\d{2})?)\s*(?:million|billion)',  # With $
            r'(?:^|[^\d])([\d,]+(?:\.\d{2})?)\s+(?:million|billion)(?:$|[^\d])',  # Without $
            r'(?:increased|decreased|was)\s+(?:to\s+)?\$?\s*([\d,]+(?:\.\d{2})?)\s*(?:million|billion)',  # With verbs
            r'(?:of|to)\s+\$?\s*([\d,]+(?:\.\d{2})?)\s*(?:million|billion)'  # With prepositions
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, pred, re.IGNORECASE)
            if matches:
                financial_numbers = matches
                break
        if not financial_numbers:
            print("No financial values found in prediction")
            continue
            
        # Convert to float, handling million/billion multipliers
        pred_nums = []
        for num_str in financial_numbers:
            try:
                # Remove commas and convert to float
                num = float(num_str.replace(',', ''))
                # Convert billion to million if needed
                if 'billion' in pred:
                    print(f"Converting {num} billion to {num * 1000} million")
                    num *= 1000
                pred_nums.append(num)
                print(f"Extracted financial value: {num} million")
            except ValueError as e:
                print(f"Failed to parse number '{num_str}': {str(e)}")
                continue
                
        if not pred_nums:
            continue
            
        # Find closest actual change
        pred_num = pred_nums[0]  # Use first number found
        print(f"\nLooking for matches for {pred_num} million")
        
        closest_actual = None
        min_diff_pct = float('inf')  # Track percentage difference
        
        for orig, updated in actual_changes.items():
            # Calculate percentage difference with adaptive tolerance
            diff_pct = abs(pred_num - updated) / updated
            
            # Scale tolerance based on value magnitude with more granular bands
            base_tolerance = 0.02  # Base 2% tolerance
            magnitude_factor = 1.0
            
            # More granular magnitude bands with higher factors
            if updated >= 8000:  # Very large numbers (>$8B)
                magnitude_factor = 2.0  # 4% tolerance
            elif updated >= 5000:  # Large numbers ($5B-$8B)
                magnitude_factor = 1.8  # 3.6% tolerance
            elif updated >= 1000:  # Medium numbers ($1B-$5B)
                magnitude_factor = 1.5  # 3% tolerance
            elif updated <= 100:  # Small numbers (<$100M)
                magnitude_factor = 0.8  # 1.6% tolerance
            
            tolerance = base_tolerance * magnitude_factor
            print(f"Comparing with {updated} million (diff: {diff_pct:.2%}, tolerance: {tolerance:.2%}, band: ${updated/1000:.1f}B)")
            
            if diff_pct < min_diff_pct:
                min_diff_pct = diff_pct
                closest_actual = (orig, updated)
        
        # Check against adaptive tolerance
        if closest_actual:
            # Use same magnitude bands for final check
            value = closest_actual[1]
            final_factor = (1.5 if value >= 8000 else
                          1.3 if value >= 5000 else
                          1.1 if value >= 1000 else
                          0.8 if value <= 100 else 1.0)
            final_tolerance = base_tolerance * final_factor
            
            if min_diff_pct <= final_tolerance:
                true_positives += 1
                found_changes.add(closest_actual)
                # Improved match quality calculation
                # Scale quality based on how far under tolerance we are
                match_quality = max(0.0, min(1.0, 1.0 - (min_diff_pct / final_tolerance) ** 0.5))
                confidence_correlation.append((conf, match_quality))
                print(f"✅ Found match: {closest_actual[1]} million (diff: {min_diff_pct:.2%}, tolerance: {final_tolerance:.2%}, quality: {match_quality:.2%})")
            else:
                false_positives += 1
                confidence_correlation.append((conf, 0))  # Incorrect prediction
                print(f"❌ Near miss: {closest_actual[1]} million (diff: {min_diff_pct:.2%} > tolerance: {final_tolerance:.2%})")
        else:
            false_positives += 1
            confidence_correlation.append((conf, 0))  # Incorrect prediction
            print("❌ No close matches found")
    
    # Count false negatives (actual changes not found)
    false_negatives = len(actual_changes) - len(found_changes)
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate confidence correlation
    if len(confidence_correlation) >= 2:  # Need at least 2 points for correlation
        conf_scores = [x[0] for x in confidence_correlation]
        correct_vals = [x[1] for x in confidence_correlation]
        if len(set(conf_scores)) > 1 and len(set(correct_vals)) > 1:  # Need variation in both arrays
            conf_correlation = np.corrcoef(conf_scores, correct_vals)[0,1]
        else:
            conf_correlation = 0.0  # No correlation if no variation
    else:
        conf_correlation = 0.0  # Not enough data points
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'confidence_correlation': conf_correlation
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
            # Load actual changes from Excel
            actual_changes = load_test_data(xlsx_path)
            
            # Process files and get predictions (now awaiting the coroutine)
            predictions = await process_files(docx_path, xlsx_path)
            
            # Extract predicted changes and confidence scores
            predicted_changes = []
            confidence_scores = []
            for orig, mod, conf in predictions:
                predicted_changes.append(mod)
                confidence_scores.append(conf)
            
            # Calculate metrics
            metrics = calculate_metrics(actual_changes, predicted_changes, confidence_scores)
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
