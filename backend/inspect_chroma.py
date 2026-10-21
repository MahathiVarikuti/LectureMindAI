#!/usr/bin/env python3
"""Inspect what chunks exist in the Chroma database."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from app.services.vector_store import get_chroma_client

client = get_chroma_client()

# Get the text chunks collection
try:
    collection = client.get_collection("text_chunks")
    print(f"Collection has {collection.count()} items\n")
    
    # Get all items
    results = collection.get(include=["documents", "metadatas", "distances"])
    
    for i, doc_id in enumerate(results["ids"]):
        meta = results["metadatas"][i]
        doc_text = results["documents"][i]
        
        section = meta.get("section", "unknown")
        filename = meta.get("filename", "unknown")
        
        print(f"[{i+1}] Section: {section}")
        print(f"    File: {filename}")
        print(f"    Text (first 100 chars): {repr(doc_text[:100])}")
        print()
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
