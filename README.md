# Financial Document Update Pipeline

## Overview
This system automatically updates financial documents by matching and replacing numerical values between Excel spreadsheets and Word documents. It uses exact text matching and LLM-based verification to ensure accurate updates while maintaining document structure.

## Pipeline
1. **Data Loading**
   - Excel Parser: Maps each cell to its row and column headers
   - Document Parser: Extracts sentences from Word documents

2. **Matching Process**
   - Finds exact text matches between Excel headers and document sentences
   - No assumptions made about text content or structure

3. **Value Updates**
   - Uses LLM to update matched sentences
   - Preserves original sentence structure and formatting
   - Only updates numerical values

4. **Confidence Scoring**
   Base confidence of 0.7 for exact matches, with additional factors:
   - Length ratio (0-0.15): Longer matches relative to sentence length
   - Position score (0-0.15): Earlier matches in sentence weighted higher
   - Update success (0-0.1): Ratio of successfully updated values
   - Multiple match penalty: -0.1 per additional match

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Set up environment variables:
```bash
LLM_PROVIDER=qwen  # LLM provider selection
QWEN_API_KEY=sk-xxx  # Your API key
```

### Running
```bash
python main.py <docx_file> <excel_file>
```

Or use the web interface:
```bash
python app.py  # Starts server on port 8080
```

## API Endpoints

### Document Processing
```
POST /upload
Files:
  - docx_file: Word document (.docx)
  - excel_file: Excel spreadsheet (.xlsx)
```

### Document Download
```
GET /download
Response: Updated .docx file
```

## Components

### Core Processing
- `funnels/excel_utils.py`: Excel file parsing
- `funnels/extract.py`: Word document text extraction
- `utils/document_processing/processor.py`: Main processing logic

### Web Interface
- `app.py`: Flask web application
- `templates/index.html`: Upload interface

### Configuration
- `config/`: Configuration management
- `utils/llm.py`: LLM integration
- `utils/logging.py`: Logging setup

## Error Handling
- Size limit: 16MB per file
- Supported formats: .docx, .xlsx only
- Automatic cleanup of temporary files
- LLM timeout: 30 seconds default

## Evaluation
Use `eval/evaluate_accuracy.py` to measure:
- Precision: Correct updates / Total updates
- Recall: Found changes / Actual changes
- F1 Score: Harmonic mean of precision and recall
