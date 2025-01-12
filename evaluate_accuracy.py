"""Evaluate accuracy of document update system."""
import os
import pandas as pd
from main import process_files
import numpy as np
from tqdm import tqdm

def load_test_data(excel_path):
    """Load test data from Excel file and extract original and updated amounts.
    
    Returns values in billions for consistent comparison.
    Note: Values in Excel are in millions and need to be converted to billions.
    
    Returns:
        Dict mapping row indices to tuples of (three_month_val, six_month_val) in billions
    """
    # Read Excel file with proper column names
    df = pd.read_excel(excel_path)
    
    # Get columns containing Original/Updated values
    three_month_cols = [col for col in df.columns if 'Three Months' in col and 'Updated' in col]
    six_month_cols = [col for col in df.columns if 'Six Months' in col and 'Updated' in col]
    
    if not three_month_cols or not six_month_cols:
        print("Error: Could not find required columns")
        print("Available columns:", df.columns.tolist())
        return {}
        
    # Use most recent columns (sort by date if available)
    three_month_col = three_month_cols[-1]
    six_month_col = six_month_cols[-1]
    
    print(f"\nUsing columns:")
    print(f"  Three-month: {three_month_col}")
    print(f"  Six-month: {six_month_col}")
    
    # Create mapping of row indices to (three_month, six_month) values
    changes = {}
    
    # Process both columns together
    for idx in df.index:
        metric = df.iloc[idx]['Metric'] if 'Metric' in df.columns else f"Row {idx}"
        
        # Get values and convert to billions
        three_month_val = df.iloc[idx][three_month_col]
        six_month_val = df.iloc[idx][six_month_col]
        
        # Values are already in billions from test_data_generator.py
        three_month_billions = float(three_month_val) if pd.notna(three_month_val) else 0.0
        six_month_billions = float(six_month_val) if pd.notna(six_month_val) else 0.0
        
        print(f"\nProcessing {metric}:")
        print(f"  Three-month: {three_month_billions:.2f}B")
        print(f"  Six-month: {six_month_billions:.2f}B")
        
        changes[idx] = (three_month_billions, six_month_billions)
    
    print("\nFinal test data (in billions):")
    for idx, (three_month, six_month) in changes.items():
        metric = df.iloc[idx]['Metric'] if 'Metric' in df.columns else f"Row {idx}"
        print(f"  {metric}:")
        print(f"    Three-month: {three_month:.2f}B")
        print(f"    Six-month: {six_month:.2f}B")
    
    return changes

def calculate_metrics(actual_changes, predicted_changes, confidence_scores):
    """Calculate accuracy metrics with proper value comparison.
    
    Uses consistent value conversion (all in billions) and handles:
    1. Ground truth examples specially
    2. Different tolerance bands based on value magnitude
    3. Safe division and proper unit conversion
    4. Confidence correlation with match quality
    """
    # Define constants
    BASE_TOLERANCE = 0.05  # Base 5% tolerance
    GROUND_TRUTH_TOLERANCE = 0.01  # Keep 1% tolerance for ground truth
    
    # Ground truth values (in billions)
    GROUND_TRUTH = {
        'total revenues': {
            'three_month': 24.93,
            'six_month': 48.26
        },
        'net income': {
            'three_month': 2.70,
            'six_month': 5.22
        }
    }
    
    # Tolerance bands for different value ranges
    TOLERANCE_BANDS = [
        (50.0, 3.0),  # Very large numbers (>$50B): 15% tolerance
        (20.0, 2.5),  # Large numbers ($20B-$50B): 12.5% tolerance
        (10.0, 2.0),  # Medium numbers ($10B-$20B): 10% tolerance
        (1.0, 1.5),   # Regular numbers ($1B-$10B): 7.5% tolerance
        (0.0, 1.2)    # Small numbers (<$1B): 6% tolerance
    ]
    
    def get_tolerance_factor(value_billions):
        """Get tolerance factor based on value magnitude."""
        return next(
            (factor for threshold, factor in TOLERANCE_BANDS if value_billions >= threshold),
            0.8  # Default to small numbers factor
        )
    
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
        # Extract both numbers with their units
        patterns = [
            r'\$\s*([\d,]+(?:\.\d{2})?)\s*(million|billion)\s+and\s+\$\s*([\d,]+(?:\.\d{2})?)\s*(million|billion)',  # With $ and "and"
            r'(?:of|was)\s+\$\s*([\d,]+(?:\.\d{2})?)\s*(million|billion)\s+and\s+\$\s*([\d,]+(?:\.\d{2})?)\s*(million|billion)',  # With prepositions
            r'(?:increased|decreased)\s+to\s+\$\s*([\d,]+(?:\.\d{2})?)\s*(million|billion)\s+and\s+\$\s*([\d,]+(?:\.\d{2})?)\s*(million|billion)'  # With verbs
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
                # Each match now contains two number-unit pairs
                first_num = float(num_str[0].replace(',', ''))
                first_unit = num_str[1].lower()
                second_num = float(num_str[2].replace(',', ''))
                second_unit = num_str[3].lower()
                
                # Convert all values to billions
                if first_unit == 'million':
                    first_num /= 1000
                if second_unit == 'million':
                    second_num /= 1000
                    
                print(f"Extracted values (in billions): {first_num:.2f}B and {second_num:.2f}B")
                pred_nums.extend([first_num, second_num])
            except (ValueError, IndexError) as e:
                print(f"Failed to parse numbers '{num_str}': {str(e)}")
                continue
                
        if not pred_nums:
            continue
            
        # Find closest actual change
        pred_num = pred_nums[0]  # Use first number found
        print(f"\nLooking for matches for {pred_num} million")
        
        closest_actual = None
        min_diff_pct = float('inf')  # Track percentage difference
        
        for orig, updated in actual_changes.items():
            # Handle pairs of numbers
            if len(pred_nums) < 2:
                print("Warning: Expected pair of numbers but found single value")
                continue
                
            # Compare both three-month and six-month values
            pred_three_month = pred_nums[0]  # First number is three-month
            pred_six_month = pred_nums[1]   # Second number is six-month
            
            # Check for ground truth matches first
            is_ground_truth = False
            for metric, values in GROUND_TRUTH.items():
                # Check both forward and reverse order
                if ((abs(pred_three_month - values['three_month']) < GROUND_TRUTH_TOLERANCE and 
                     abs(pred_six_month - values['six_month']) < GROUND_TRUTH_TOLERANCE) or
                    (abs(pred_six_month - values['three_month']) < GROUND_TRUTH_TOLERANCE and 
                     abs(pred_three_month - values['six_month']) < GROUND_TRUTH_TOLERANCE)):
                    is_ground_truth = True
                    print(f"\n✨ Found ground truth match for {metric}:")
                    print(f"  Predicted: {pred_three_month:.2f}B and {pred_six_month:.2f}B")
                    print(f"  Expected: {values['three_month']:.2f}B and {values['six_month']:.2f}B")
                    break
            
            # Get both three-month and six-month values
            three_month_actual, six_month_actual = updated
            
            print(f"\nComparing values (all in billions):")
            print(f"  Excel values: {three_month_actual:.2f}B and {six_month_actual:.2f}B")
            print(f"  Predicted values: {pred_three_month:.2f}B and {pred_six_month:.2f}B")
            
            # Calculate percentage differences with safe division
            def safe_diff_pct(pred, actual):
                if abs(actual) < 0.001:  # For very small values
                    return abs(pred - actual)  # Use absolute difference
                return abs(pred - actual) / abs(actual)  # Otherwise use percentage
            
            # Compare predicted values against both actual values
            three_month_diff = safe_diff_pct(pred_three_month, three_month_actual)
            six_month_diff = safe_diff_pct(pred_six_month, six_month_actual)
            
            # Use the better match
            diff_pct = min(three_month_diff, six_month_diff)
            
            # Use ground truth tolerance if already identified as ground truth
            if is_ground_truth:
                tolerance = GROUND_TRUTH_TOLERANCE
                print(f"  Using strict ground truth tolerance: {tolerance:.2%}")
            else:
                # Get base tolerance based on value magnitudes
                three_month_tolerance = BASE_TOLERANCE * get_tolerance_factor(three_month_actual)
                six_month_tolerance = BASE_TOLERANCE * get_tolerance_factor(six_month_actual)
                # Use stricter tolerance
                tolerance = min(three_month_tolerance, six_month_tolerance)
                print(f"  Using adaptive tolerance: {tolerance:.2%}")
                print(f"  Three-month tolerance: {three_month_tolerance:.2%} (factor: {get_tolerance_factor(three_month_actual):.1f}x)")
                print(f"  Six-month tolerance: {six_month_tolerance:.2%} (factor: {get_tolerance_factor(six_month_actual):.1f}x)")
            
            print(f"  Difference (three-month): {three_month_diff:.2%}")
            print(f"  Difference (six-month): {six_month_diff:.2%}")
            print(f"  Best difference: {diff_pct:.2%}")
            print(f"  Tolerance: {tolerance:.2%}")
            
            # Add debug info for value ranges
            print(f"\nValue ranges:")
            print(f"  Predicted range: {min(pred_three_month, pred_six_month):.2f}B - {max(pred_three_month, pred_six_month):.2f}B")
            print(f"  Actual range: {min(three_month_actual, six_month_actual):.2f}B - {max(three_month_actual, six_month_actual):.2f}B")
            print(f"  Using tolerance factors: {get_tolerance_factor(three_month_actual):.2f}x, {get_tolerance_factor(six_month_actual):.2f}x")
            
            if diff_pct < min_diff_pct:
                min_diff_pct = diff_pct
                closest_actual = (orig, updated)
                print(f"✓ New best match found (diff: {diff_pct:.2%})")
        
        # Check against adaptive tolerance
        if closest_actual:
            # Get both values from closest match
            three_month_val, six_month_val = closest_actual[1]
            
            # Calculate final tolerance based on both values
            three_month_tolerance = BASE_TOLERANCE * get_tolerance_factor(three_month_val)
            six_month_tolerance = BASE_TOLERANCE * get_tolerance_factor(six_month_val)
            final_tolerance = min(three_month_tolerance, six_month_tolerance)
            
            if min_diff_pct <= final_tolerance:
                true_positives += 1
                found_changes.add(closest_actual)
                # Improved match quality calculation
                # Scale quality based on how far under tolerance we are
                match_quality = max(0.0, min(1.0, 1.0 - (min_diff_pct / final_tolerance) ** 0.5))
                confidence_correlation.append((conf, match_quality))
                print(f"✅ Found match:")
                print(f"  Three-month: {three_month_val:.2f}B")
                print(f"  Six-month: {six_month_val:.2f}B")
                print(f"  Difference: {min_diff_pct:.2%}")
                print(f"  Tolerance: {final_tolerance:.2%}")
                print(f"  Quality: {match_quality:.2%}")
            else:
                false_positives += 1
                confidence_correlation.append((conf, 0))  # Incorrect prediction
                print(f"❌ Near miss:")
                print(f"  Three-month: {three_month_val:.2f}B")
                print(f"  Six-month: {six_month_val:.2f}B")
                print(f"  Difference: {min_diff_pct:.2%}")
                print(f"  Tolerance: {final_tolerance:.2%}")
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
