# Document Update System

A system for automatically updating financial metrics in documents using exact matching and LLM-based updates.

## Pipeline

1. **Data Loading**
   - Excel file: Extract metrics and their values
   - Word document: Extract sentences for processing

2. **Matching Process**
   - Find exact text matches between Excel headers and document sentences
   - No fuzzy matching or embeddings to ensure accuracy

3. **Value Updates**
   - Use LLM to update matched sentences with new values
   - Preserve original sentence structure and formatting
   - Only numerical values are changed

4. **Confidence Scoring**

The system uses a sophisticated confidence scoring mechanism:

```
confidence = 0.9 (base score for exact match)
           + length_score (0-0.15)
           + position_score (0-0.15)
           + update_score (0-0.1)
           - multiple_match_penalty
```

Factors considered:
1. **Length Ratio** (0-0.15)
   - Ratio of match length to sentence length
   - Longer matches relative to sentence get higher scores
   - Score = min(length_ratio * 0.5, 0.15)

2. **Position in Sentence** (0-0.15)
   - Earlier matches are weighted higher
   - Score = (1 - position_ratio) * 0.15
   - Position ratio = match_start / sentence_length

3. **Value Update Success** (0-0.1)
   - Ratio of successfully updated values
   - Score = (updated_values / total_values) * 0.1

4. **Multiple Match Penalty**
   - -0.1 for each additional match in the same sentence
   - Penalty = 0.1 * (num_matches - 1)

## Usage

```bash
python main.py <docx_file> <excel_file>
```

The system will:
1. Load both files
2. Find exact matches
3. Update values using LLM
4. Show changes with confidence scores
5. Save updated document

## Requirements

1. Excel files:
   - First column: Metric names
   - Other columns: Numerical values

2. Word documents:
   - Contains financial metrics matching Excel headers
   - Numerical values to be updated
