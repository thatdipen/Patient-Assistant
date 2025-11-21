# vector_store.py
import json
import os
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
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "patient-support-vectors")

INPUT_JSON = os.getenv("INPUT_JSON", "patient_data.json")

def build_text_for_embedding(item: dict, item_type: str, item_id: str) -> str:
    """
    Construct a searchable text blob from patient support data.
    Different formats for different data types.
    """
    parts = []
    
    if item_type == "patient_education":
        parts.append(f"Disease: {item.get('disease_name', '')}")
        parts.append(f"Overview: {item.get('overview', '')}")
        
        if 'key_facts' in item:
            parts.append("Key Facts:")
            for fact in item['key_facts']:
                parts.append(f"  - {fact}")
        
        if 'medications' in item:
            parts.append("Medications:")
            for med in item['medications']:
                parts.append(f"  {med.get('name', '')}: {med.get('purpose', '')}")
                if 'common_brands' in med:
                    parts.append(f"    Brands: {', '.join(med['common_brands'])}")
                if 'important_notes' in med:
                    parts.append(f"    Important: {med['important_notes']}")
        
        if 'lifestyle_tips' in item:
            parts.append("Lifestyle Tips:")
            for tip in item['lifestyle_tips']:
                parts.append(f"  - {tip}")
        
        if 'when_to_seek_help' in item:
            parts.append("When to Seek Help:")
            for warning in item['when_to_seek_help']:
                parts.append(f"  - {warning}")
                
    elif item_type == "adherence_tools":
        parts.append(f"Adherence Topic: {item_id}")
        if 'tips' in item:
            parts.append("Tips:")
            for tip in item['tips']:
                parts.append(f"  - {tip}")
        if 'common_challenges' in item:
            parts.append("Common Challenges:")
            for challenge in item['common_challenges']:
                parts.append(f"  - {challenge}")
        if 'solutions' in item:
            parts.append("Solutions:")
            for solution in item['solutions']:
                parts.append(f"  - {solution}")
        if 'what_to_track' in item:
            parts.append("What to Track:")
            for item_track in item['what_to_track']:
                parts.append(f"  - {item_track}")
        if 'tracking_methods' in item:
            parts.append("Tracking Methods:")
            for method in item['tracking_methods']:
                parts.append(f"  - {method}")
        if 'strategies' in item:
            parts.append("Strategies:")
            for strategy in item['strategies']:
                parts.append(f"  - {strategy}")
                
    elif item_type == "symptom_tracking":
        parts.append(f"Symptom Tracking for: {item_id}")
        if 'symptoms_to_track' in item:
            parts.append("Symptoms to Track:")
            for symptom in item['symptoms_to_track']:
                parts.append(f"  - {symptom}")
        if 'tracking_frequency' in item:
            parts.append(f"Tracking Frequency: {item['tracking_frequency']}")
        if 'red_flags' in item:
            parts.append("Red Flags (Seek Immediate Help):")
            for flag in item['red_flags']:
                parts.append(f"  - {flag}")
                
    elif item_type == "patient_journey":
        parts.append(f"Patient Journey Stage: {item.get('stage', '')}")
        if 'typical_duration' in item:
            parts.append(f"Typical Duration: {item['typical_duration']}")
        if 'key_milestones' in item:
            parts.append("Key Milestones:")
            for milestone in item['key_milestones']:
                parts.append(f"  - {milestone}")
        if 'support_needed' in item:
            parts.append("Support Needed:")
            for support in item['support_needed']:
                parts.append(f"  - {support}")
                
    elif item_type == "support_programs":
        if 'programs' in item:
            parts.append("Support Programs:")
            for program in item['programs']:
                parts.append(f"  {program.get('name', '')}: {program.get('description', '')}")
                if 'benefits' in program:
                    parts.append("    Benefits:")
                    for benefit in program['benefits']:
                        parts.append(f"      - {benefit}")
                if 'duration' in program:
                    parts.append(f"    Duration: {program['duration']}")
                if 'eligibility' in program:
                    parts.append(f"    Eligibility: {program['eligibility']}")
        if 'online_resources' in item:
            parts.append("Online Resources:")
            for resource in item['online_resources']:
                parts.append(f"  - {resource}")
        if 'crisis_resources' in item:
            parts.append("Crisis Resources:")
            for resource in item['crisis_resources']:
                parts.append(f"  - {resource}")
    
    text_blob = "\n".join(parts).strip()
    return text_blob[:6000]  # Keep within reasonable size

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
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Check if collection exists, get or create
    try:
        collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)
        print(f"✅ Collection '{CHROMA_COLLECTION_NAME}' already exists")
        count = collection.count()
        print(f"   Current document count: {count}")
    except Exception:
        # Collection doesn't exist, create it
        print(f"Creating ChromaDB collection: {CHROMA_COLLECTION_NAME}")
        collection = chroma_client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"✅ Collection {CHROMA_COLLECTION_NAME} created successfully!")
    
    # Load JSON
    if not os.path.exists(INPUT_JSON):
        raise FileNotFoundError(f"Input JSON not found: {INPUT_JSON}")
    
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    metadatas = []
    documents = []
    
    doc_count = 0
    
    # Process Patient Education data
    if "patient_education" in data:
        print("\nProcessing Patient Education data...")
        for disease, info in data["patient_education"].items():
            text = build_text_for_embedding(info, "patient_education", disease)
            embedding = create_embedding(text)
            
            metadata = {
                "type": "patient_education",
                "disease": disease,
                "disease_name": info.get("disease_name", ""),
            }
            metadata = clean_metadata(metadata)
            
            ids.append(f"edu_{disease}")
            embeddings.append(embedding)
            metadatas.append(metadata)
            documents.append(text)
            doc_count += 1
    
    # Process Adherence Tools data
    if "adherence_tools" in data:
        print("\nProcessing Adherence Tools data...")
        for tool_type, tool_info in data["adherence_tools"].items():
            text = build_text_for_embedding(tool_info, "adherence_tools", tool_type)
            embedding = create_embedding(text)
            
            metadata = {
                "type": "adherence_tools",
                "tool_type": tool_type,
            }
            metadata = clean_metadata(metadata)
            
            ids.append(f"adherence_{tool_type}")
            embeddings.append(embedding)
            metadatas.append(metadata)
            documents.append(text)
            doc_count += 1
    
    # Process Symptom Tracking data
    if "symptom_tracking" in data:
        print("\nProcessing Symptom Tracking data...")
        if "common_symptoms" in data["symptom_tracking"]:
            for condition, symptom_info in data["symptom_tracking"]["common_symptoms"].items():
                text = build_text_for_embedding(symptom_info, "symptom_tracking", condition)
                embedding = create_embedding(text)
                
                metadata = {
                    "type": "symptom_tracking",
                    "condition": condition,
                }
                metadata = clean_metadata(metadata)
                
                ids.append(f"symptom_{condition}")
                embeddings.append(embedding)
                metadatas.append(metadata)
                documents.append(text)
                doc_count += 1
    
    # Process Patient Journey data
    if "patient_journey" in data:
        print("\nProcessing Patient Journey data...")
        for journey_type, stages in data["patient_journey"].items():
            for stage_name, stage_info in stages.items():
                text = build_text_for_embedding(stage_info, "patient_journey", stage_name)
                embedding = create_embedding(text)
                
                metadata = {
                    "type": "patient_journey",
                    "journey_type": journey_type,
                    "stage": stage_info.get("stage", stage_name),
                }
                metadata = clean_metadata(metadata)
                
                ids.append(f"journey_{journey_type}_{stage_name}")
                embeddings.append(embedding)
                metadatas.append(metadata)
                documents.append(text)
                doc_count += 1
    
    # Process Support Programs data
    if "support_programs" in data:
        print("\nProcessing Support Programs data...")
        for program_type, program_info in data["support_programs"].items():
            text = build_text_for_embedding(program_info, "support_programs", program_type)
            embedding = create_embedding(text)
            
            metadata = {
                "type": "support_programs",
                "program_type": program_type,
            }
            metadata = clean_metadata(metadata)
            
            ids.append(f"support_{program_type}")
            embeddings.append(embedding)
            metadatas.append(metadata)
            documents.append(text)
            doc_count += 1
    
    print(f"\nTotal documents prepared: {doc_count}")
    
    # Batch add to ChromaDB
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

