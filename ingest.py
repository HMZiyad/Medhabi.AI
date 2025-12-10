import os
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import argparse
from tqdm import tqdm

# --- Configuration ---
# NOTE: You must have Tesseract-OCR installed on your system and added to PATH.
# You also need 'poppler' installed for pdf2image.

# If Tesseract is not in PATH, set the path:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Set Poppler Path (Winget location)
POPPLER_PATH = r'C:\Users\ziyad\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin'

# Set TESSDATA_PREFIX to our local directory containing ben.traineddata
os.environ['TESSDATA_PREFIX'] = os.path.join(os.getcwd(), 'tessdata')

EMBEDDING_MODEL_NAME = "l3cube-pune/bengali-sentence-similarity-sbert"
VECTOR_DB_DIR = "chroma_db"
COLLECTION_NAME = "physics_nctb"

def preprocess_image(image):
    """
    Applies preprocessing to an image to improve OCR accuracy.
    """
    # Convert PIL image to numpy array (OpenCV format)
    img = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # proper thresholding (Otsu's binarization)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def extract_text_from_pdf(pdf_path):
    """
    Converts PDF pages to images and extracts text using Tesseract OCR (Bengali).
    """
    print(f"Loading PDF: {pdf_path}")
    try:
        # Convert PDF to images (300 DPI is a good balance for OCR)
        # Explicitly provide poppler_path
        images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        print("Ensure 'poppler' is installed and valid in your system PATH.")
        return []

    documents = []
    print(f"Processing {len(images)} pages...")

    for i, image in enumerate(tqdm(images, desc="OCR Processing", unit="page")):
        # print(f"  Processing page {i+1}...") # tqdm replaces this
        
        # Preprocess for better OCR
        processed_img = preprocess_image(image)
        
        # Run OCR (Bengali language)
        # config='--oem 1 --psm 6' is often good for blocks of text
        text = pytesseract.image_to_string(processed_img, lang='ben', config='--oem 1 --psm 6')
        
        if text.strip():
            # Create a Document object for each page (or we can chunk later)
            # We treat the whole page as a context source initially, 
            # but chunking will happen next.
            # Storing page number in metadata is helpful.
            doc = Document(page_content=text, metadata={"source": pdf_path, "page": i+1})
            documents.append(doc)
            
    return documents

def chunk_documents(documents):
    """
    Splits documents into smaller chunks for embedding.
    """
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "।", " ", ""] # Bengali variations
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks):
    """
    Creates (or updates) a Chroma vector store with the chunks.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print(f"Creating Vector Store in '{VECTOR_DB_DIR}'...")
    # Persist the DB to disk
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_DIR,
        collection_name=COLLECTION_NAME
    )
    # vectorstore.persist() is deprecated/auto-called in newer versions, 
    # but good to note we rely on persistence.
    print("Vector Store created successfully.")
    return vectorstore

def main():
    parser = argparse.ArgumentParser(description="Ingest Bengali Physics PDF for RAG.")
    parser.add_argument("pdf_path", help="C:/Users/ziyad/RAG Physics/physics.pdf")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: File not found at {args.pdf_path}")
        return

    # 1. OCR Extraction
    raw_docs = extract_text_from_pdf(args.pdf_path)
    
    if not raw_docs:
        print("No text extracted. Exiting.")
        return

    # 2. Chunking
    chunks = chunk_documents(raw_docs)

    # 3. Vector Store Creation
    create_vector_store(chunks)
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
