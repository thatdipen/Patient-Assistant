# vector_store.py
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings

load_dotenv()

# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ChromaDB setup
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./patient_chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "patient-vectors")

INPUT_EXCEL = os.getenv("INPUT_EXCEL", "faqs.xlsx")

def build_text_for_embedding(question: str, answer: str) -> str:
    """
    Construct a single text blob from FAQ question and answer.
    Combines question and answer for better semantic search.
    """
    # Combine question and answer for embedding
    text_blob = f"Question: {question}\nAnswer: {answer}".strip()
    # Keep it within a reasonable size for embeddings
    return text_blob[:6000]

def create_embedding(text: str):
    """Generate an embedding vector for a given text."""
    if not text:
        text = " "  # avoid empty input to embeddings API
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding

def clean_metadata(metadata: dict) -> dict:
    """Remove None values from metadata dict. ChromaDB doesn't accept None values."""
    return {k: v for k, v in metadata.items() if v is not None and v != ""}

def store_embeddings():
    """Store embeddings in ChromaDB collection."""
    # Get embedding dimension from OpenAI model
    print(f"Getting embedding dimension for model: {EMBEDDING_MODEL}")
    sample_embedding = create_embedding("sample")
    embedding_dimension = len(sample_embedding)
    print(f"✅ Embedding dimension: {embedding_dimension}")
    
    # Initialize ChromaDB client (persistent mode)
    # Create directory if it doesn't exist
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Check if collection exists, get or create
    try:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"✅ Collection '{CHROMA_COLLECTION_NAME}' already exists")
        
        # Check if collection is empty or needs update
        count = collection.count()
        print(f"   Current document count: {count}")
        
        # Option: Clear existing collection if you want to rebuild
        # Uncomment the next 2 lines if you want to rebuild from scratch
        # chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
        # collection = chroma_client.create_collection(name=CHROMA_COLLECTION_NAME)
        
    except Exception:
        # Collection doesn't exist, create it
        print(f"Creating ChromaDB collection: {CHROMA_COLLECTION_NAME}")
        collection = chroma_client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        print(f"✅ Collection {CHROMA_COLLECTION_NAME} created successfully!")
    
    # Load Excel file
    if not os.path.exists(INPUT_EXCEL):
        raise FileNotFoundError(f"Input Excel file not found: {INPUT_EXCEL}")
    
    print(f"Loading Excel file: {INPUT_EXCEL}")
    df = pd.read_excel(INPUT_EXCEL)
    
    # Check if required columns exist
    if df.shape[1] < 2:
        raise ValueError(f"Excel file must have at least 2 columns (Question and Answer). Found {df.shape[1]} columns.")
    
    # Get first two columns (Question and Answer)
    question_col = df.columns[0]
    answer_col = df.columns[1]
    
    print(f"Using columns: '{question_col}' (Question) and '{answer_col}' (Answer)")
    
    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    metadatas = []
    documents = []
    
    print(f"\nProcessing {len(df)} FAQs...")
    
    for i, row in df.iterrows():
        # Get question and answer from first two columns
        question = str(row[question_col]).strip() if pd.notna(row[question_col]) else ""
        answer = str(row[answer_col]).strip() if pd.notna(row[answer_col]) else ""
        
        # Skip rows with empty question or answer
        if not question or not answer:
            print(f"   Skipping row {i+1}: empty question or answer")
            continue
        
        uid = f"faq_{i+1}"
        text = build_text_for_embedding(question, answer)
        embedding = create_embedding(text)
        
        # ChromaDB metadata - store question and answer separately for reference
        metadata = {
            "question": question,
            "answer": answer,
            "faq_id": uid,
            "chunk_index": "0",
        }
        
        # Clean metadata to remove any None or empty values
        metadata = clean_metadata(metadata)
        
        ids.append(uid)
        embeddings.append(embedding)
        metadatas.append(metadata)
        documents.append(text)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(df)} FAQs...")
    
    # Batch add to ChromaDB (ChromaDB handles batching internally, but we can do it manually for large datasets)
    batch_size = 100
    total_batches = (len(ids) + batch_size - 1) // batch_size
    
    print(f"\nAdding documents to ChromaDB in {total_batches} batches...")
    
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
        
        batch_num = i // batch_size + 1
        print(f"   Added batch {batch_num}/{total_batches} ({len(batch_ids)} documents)")
    
    # Verify final count
    final_count = collection.count()
    print(f"\n✅ All embeddings stored successfully in ChromaDB!")
    print(f"   Total documents in collection: {final_count}")
    print(f"   Collection path: {os.path.abspath(CHROMA_DB_PATH)}")

if __name__ == "__main__":
    store_embeddings()
