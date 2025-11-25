# vectorize_all.py
"""
Master script to vectorize both Excel FAQs and PDF documents.
This script processes both data sources and stores them in the same ChromaDB collection.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def vectorize_excel():
    """Vectorize Excel FAQs."""
    print("=" * 60)
    print("STEP 1: Vectorizing Excel FAQs")
    print("=" * 60)
    from vector_store import store_embeddings
    store_embeddings()
    print()

def vectorize_pdfs():
    """Vectorize PDF documents."""
    print("=" * 60)
    print("STEP 2: Vectorizing PDF Documents")
    print("=" * 60)
    from pdf_vectorizer import vectorize_pdfs
    vectorize_pdfs()
    print()

def main():
    """Main function to run both vectorization processes."""
    print("\n" + "=" * 60)
    print("DOCTOR ASSISTANT - COMPLETE VECTORIZATION")
    print("=" * 60)
    print()
    
    # Check what to process
    process_excel = os.getenv("PROCESS_EXCEL", "true").lower() == "true"
    process_pdfs = os.getenv("PROCESS_PDFS", "true").lower() == "true"
    
    # Allow command line arguments to override
    if len(sys.argv) > 1:
        if "excel" in sys.argv[1].lower():
            process_excel = True
            process_pdfs = False
        elif "pdf" in sys.argv[1].lower():
            process_excel = False
            process_pdfs = True
    
    try:
        # Process Excel
        if process_excel:
            vectorize_excel()
        else:
            print("⏭️  Skipping Excel vectorization (PROCESS_EXCEL=false or --pdf flag used)")
            print()
        
        # Process PDFs
        if process_pdfs:
            vectorize_pdfs()
        else:
            print("⏭️  Skipping PDF vectorization (PROCESS_PDFS=false or --excel flag used)")
            print()
        
        print("=" * 60)
        print("✅ VECTORIZATION COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during vectorization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

