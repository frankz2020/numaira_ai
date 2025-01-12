# Financial Document Update Pipeline

## Overview
This system automatically updates financial documents by matching and replacing numerical values between Excel spreadsheets and Word documents. It uses semantic similarity matching and LLM-based formatting to ensure accurate updates while maintaining document structure.

## Architecture

### Pipeline Flow
```
Excel File → Excel Parser → Value Extraction
                                ↓
Word Doc  → Document Parser → Sentence Matching → Confidence Scoring → Document Update
                                ↓
                          Web Interface
```

### Components

1. **Document Processing**
   - `funnels/extract.py`: Extracts text from Word documents
   - `funnels/excel_utils.py`: Parses Excel files for financial metrics
   - `funnels/selection.py`: Filters and matches sentences with metrics

2. **RAG (Retrieval-Augmented Generation)**
   - `RAG/similarity.py`: Semantic similarity matching
   - `RAG/format_mapping.py`: LLM-based text formatting
   - `RAG/async_format.py`: Asynchronous format processing

3. **LLM Integration**
   - `utils/llm.py`: Centralized LLM configuration
   - `funnels/llm_provider.py`: Provider implementations (Qwen/Claude)

4. **Web Interface**
   - `app.py`: Flask web application
   - `templates/index.html`: Upload interface and results display

## API Endpoints

### Document Processing
```python
POST /upload
Content-Type: multipart/form-data
Files:
  - docx_file: Word document (.docx)
  - excel_file: Excel spreadsheet (.xlsx)
Response:
{
    "results": [
        {
            "original": str,  # Original sentence
            "modified": str,  # Modified sentence
            "confidence": float  # Update confidence (0.0-1.0)
        }
    ]
}
```

### Document Download
```python
GET /download
Response: Updated .docx file (application/vnd.openxmlformats-officedocument.wordprocessingml.document)
```

## Data Structures

### Document Changes
```python
{
    sentence_index: [
        (
            [target_words],  # List of words to match
            new_value,       # Updated value
            confidence       # Similarity score (0.0-1.0)
        )
    ]
}
```

### Confidence Scoring
- High: ≥80% (Green)
- Medium: 50-79% (Yellow)
- Low: <50% (Red)

## Configuration

### Environment Variables
```
LLM_PROVIDER=qwen|claude  # LLM provider selection
QWEN_API_KEY=sk-xxx      # Qwen API key
CLAUDE_API_KEY=sk-xxx    # Claude API key (optional)
```

### Critical Assumptions
1. Documents contain exact words matching Excel column/row names
2. Excel structure: leftmost column and topmost row contain cell meanings
3. English language documents using "billion" or "million" with two decimal places
4. Excel values are raw numbers without units

## Error Handling

### LLM Calls
- Timeout: 30 seconds default
- Retry logic for API failures
- Fallback to exact matching if LLM fails

### File Processing
- Size limit: 16MB per file
- Supported formats: .docx, .xlsx only
- Automatic cleanup of temporary files

### Web Interface
- Session-based file tracking
- Secure filename handling
- Error feedback to users

## Testing

### Test Data Generation
```python
python test_data_generator.py  # Generates test file pairs
python verify_test_files.py    # Verifies file structure
python evaluate_accuracy.py    # Evaluates accuracy metrics
```

### Accuracy Metrics
- Precision: Correct updates / Total updates
- Recall: Found changes / Actual changes
- F1 Score: Harmonic mean of precision and recall
- Confidence Correlation: Correlation between confidence scores and correct matches

## Development Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run development server:
```bash
python app.py  # Starts Flask server on port 8080
```

## Known Issues

1. Number Extraction
   - Current regex may capture dates instead of financial values
   - Need to improve pattern matching for complex number formats

2. Confidence Scoring
   - False positives with high confidence for similar but incorrect matches
   - Need to adjust similarity thresholds

3. Performance
   - LLM calls can be slow for large documents
   - Consider batch processing for multiple updates
