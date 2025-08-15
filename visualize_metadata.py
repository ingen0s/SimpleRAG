import chromadb
import sys

# Interactive script to visualize metadata from a selected ChromaDB collection

def select_collection(client):
    collections = client.list_collections()
    if not collections:
        print("No collections found.")
        sys.exit(0)
    print("Available collections:")
    for idx, col in enumerate(collections):
        print(f"  [{idx}] {col.name}")
    sel = input("Select collection by number: ")
    try:
        sel_idx = int(sel)
        collection = collections[sel_idx]
        return collection
    except Exception:
        print("Invalid selection.")
        sys.exit(1)

def main():
    client = chromadb.PersistentClient(path="photo_db")
    collection = select_collection(client)
    print(f"\nShowing metadata for collection: {collection.name}\n")
    # Get only video frame metadata
    results = collection.get(where={"type": "video_frame"}, include=["metadatas", "documents"])
    metadatas = results.get("metadatas", [])
    documents = results.get("documents", [])
    for idx, meta in enumerate(metadatas):
        print(f"ID: {documents[idx]}")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        # Explicitly show caption
        if "caption" in meta:
            print(f"  Caption: {meta['caption']}")
        print("-" * 40)

if __name__ == "__main__":
    main()
