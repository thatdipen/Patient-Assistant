# pdf_vectorizer.py
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
import PyPDF2
from typing import List
load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ChromaDB setup (same as vector_store.py)
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "medical-vectors")

# PDF folder path
PDF_FOLDER = os.getenv("PDF_FOLDER", "./pdfs")

# Chunking settings
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

def create_embedding(text: str):
    """Generate an embedding vector for a given text."""
    if not text:
        text = " "  # avoid empty input to embeddings API
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def clean_metadata(metadata: dict) -> dict:
    """Remove None values from metadata dict. ChromaDB doesn't accept None values."""
    return {k: v for k, v in metadata.items() if v is not None and v != ""}

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
    except Exception as e:
        print(f"⚠️  Error extracting text from {pdf_path}: {e}")
        return ""
    return text.strip()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for better context preservation.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If this is not the last chunk, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 200 characters
            sentence_endings = ['. ', '.\n', '! ', '!\n', '? ', '?\n']
            for i in range(end, max(start + chunk_size - 200, start), -1):
                if any(text[i:i+2] == ending for ending in sentence_endings):
                    end = i + 2
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap
    
    return chunks

def process_pdf_file(pdf_path: str, collection) -> int:
    """
    Process a single PDF file and add its chunks to ChromaDB.
    Returns the number of chunks added.
    """
    pdf_name = os.path.basename(pdf_path)
    print(f"\n📄 Processing PDF: {pdf_name}")
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print(f"   ⚠️  No text extracted from {pdf_name}, skipping...")
        return 0
    
    print(f"   Extracted {len(text)} characters from {pdf_name}")
    
    # Chunk the text
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"   Created {len(chunks)} chunks")
    
    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    metadatas = []
    documents = []
    
    for i, chunk in enumerate(chunks):
        # Generate unique ID for each chunk
        chunk_id = f"pdf_{uuid.uuid4().hex[:12]}_{i}"
        
        # Create embedding
        embedding = create_embedding(chunk)
        
        # Prepare metadata
        metadata = {
            "source_type": "pdf",
            "pdf_filename": pdf_name,
            "pdf_path": pdf_path,
            "chunk_index": str(i),
            "total_chunks": str(len(chunks)),
        }
        
        # Clean metadata
        metadata = clean_metadata(metadata)
        
        ids.append(chunk_id)
        embeddings.append(embedding)
        metadatas.append(metadata)
        documents.append(chunk)
    
    # Add to ChromaDB in batches
    batch_size = 100
    total_batches = (len(ids) + batch_size - 1) // batch_size
    
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_documents = documents[i:i + batch_size]
        
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents
        )
    
    print(f"   ✅ Added {len(chunks)} chunks from {pdf_name} to ChromaDB")
    return len(chunks)

def vectorize_pdfs():
    """Vectorize all PDFs in the specified folder and add to ChromaDB."""
    # Get embedding dimension from OpenAI model
    print(f"Getting embedding dimension for model: {EMBEDDING_MODEL}")
    sample_embedding = create_embedding("sample")
    embedding_dimension = len(sample_embedding)
    print(f"✅ Embedding dimension: {embedding_dimension}")
    
    # Initialize ChromaDB client (persistent mode)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get or create collection (same as Excel data)
    try:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"✅ Collection '{CHROMA_COLLECTION_NAME}' already exists")
        current_count = collection.count()
        print(f"   Current document count: {current_count}")
    except Exception:
        # Collection doesn't exist, create it
        print(f"Creating ChromaDB collection: {CHROMA_COLLECTION_NAME}")
        collection = chroma_client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Collection {CHROMA_COLLECTION_NAME} created successfully!")
    
    # Check if PDF folder exists
    if not os.path.exists(PDF_FOLDER):
        raise FileNotFoundError(f"PDF folder not found: {PDF_FOLDER}")
    
    # Find all PDF files
    pdf_files = list(Path(PDF_FOLDER).glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {PDF_FOLDER}")
        return
    
    print(f"\n📁 Found {len(pdf_files)} PDF file(s) in {PDF_FOLDER}")
    
    total_chunks = 0
    processed_files = 0
    
    # Process each PDF
    for pdf_path in pdf_files:
        try:
            chunks_added = process_pdf_file(str(pdf_path), collection)
            total_chunks += chunks_added
            processed_files += 1
        except Exception as e:
            print(f"   ❌ Error processing {pdf_path.name}: {e}")
            continue
    
    # Final summary
    final_count = collection.count()
    print(f"\n✅ PDF vectorization complete!")
    print(f"   Processed {processed_files}/{len(pdf_files)} PDF files")
    print(f"   Total chunks added: {total_chunks}")
    print(f"   Total documents in collection: {final_count}")
    print(f"   Collection path: {os.path.abspath(CHROMA_DB_PATH)}")

if __name__ == "__main__":
    vectorize_pdfs()

