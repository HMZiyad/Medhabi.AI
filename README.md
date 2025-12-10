# Bengali Physics Tutor (RAG Pipeline)

This project implements a Retrieval-Augmented Generation (RAG) system to answer physics questions based on the NCTB Physics 1st Paper (Bengali) book.

## Prerequisites

1.  **Python 3.9+**
2.  **Tesseract OCR**:
    *   Download and install Tesseract.
    *   **Important**: Install the **Bengali (ben)** language data during installation.
    *   Add Tesseract to your system PATH.
3.  **Poppler**:
    *   Required for `pdf2image`. Download and add `bin` folder to PATH.
4.  **Google Gemini API Key**:
    *   Get an API key from Google AI Studio.

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    *   Copy `.env.example` to `.env`.
    *   Add your API Key: `GOOGLE_API_KEY=your_key_here`.

## Usage

### 1. Ingest Data (Prepare the Brain)
You need the NCTB Physics PDF file. Run the ingestion script to OCR, chunk, and index the book.

```bash
python ingest.py "path/to/nctb_physics_1st_paper.pdf"
```
*   *Note: This process may take a while depending on the PDF size and your CPU (for OCR).*
*   *It creates a `chroma_db` folder storing the knowledge.*

### 2. Run the Tutor
**Option A: Web Interface (Recommended)**
```bash
streamlit run app.py
```

**Option B: CLI**
```bash
python main.py
```

### 3. Verification
*   Ask specific questions like "নিউটনের দ্বিতীয় সূত্রটি কী?".
*   The system will verify if the answer exists in the ingested book and respond in Bengali.

## Files
*   `ingest.py`: OCR and Vector Store creation.
*   `rag_chain.py`: Logic for retrieving headers and generating answers with Gemini.
*   `main.py`: User interface.
