# Document Processing API Documentation

## Overview
This API provides functionality for processing Word documents using provided data to update numerical values while maintaining document structure and formatting.

## Base URL
```
http://localhost:8000
```

## Endpoints

### Process Documents
`POST /run-syncspace/`

Processes a Word document using provided data to update numerical values.

#### Request

**Content-Type:** `multipart/form-data`

**Parameters:**
- `docx_file`: (file, required) - Word document (.docx) to be processed
- `excel_data`: (JSON object, required) - Data structure containing values for updates

**Excel Data Structure (Example):**
```json
{
    "data": [
        {
            "row_header": "Revenue",
            "values": {
                "2022": 100,
                "2023": 120
            }
        },
        {
            "row_header": "Cost",
            "values": {
                "2022": 50,
                "2023": 60
            }
        }
    ]
}
```

#### Response (Example)

**Success Response (200 OK)**
```json
{
    "status": "success",
    "data": {
        "number_of_changes": 2,
        "results": [
            {
                "original_text": "The revenue was $100 million in 2022",
                "modified_text": "The revenue was $120 million in 2023",
                "confidence": 0.95
            }
        ],
        "output_file_path": "path/to/updated/document.docx"
    }
}
```

**Error Response (500 Internal Server Error)**
```json
{
    "status": "error",
    "message": "Error description"
}
```

## Error Handling
- The API returns appropriate HTTP status codes and error messages
- Temporary files are automatically cleaned up even if an error occurs
- All errors are logged for debugging purposes

## CORS
The API allows cross-origin requests from `http://numaira.app` with full access to methods and headers.
