import chromadb

# Connect to the ChromaDB persistent client
client_db = chromadb.PersistentClient(path="./photo_db")

collections = client_db.list_collections()
print("Available collections:")
for idx, col in enumerate(collections):
    print(f"  [{idx}] {col.name}")
choice = input("Select a collection by number: ").strip()
try:
    idx = int(choice)
    collection = client_db.get_collection(name=collections[idx].name)
except Exception:
    print("Invalid selection. Exiting.")
    exit(1)


# Query for all items in the collection
all_results = collection.get()
print(f"\nTotal items in collection: {len(all_results['ids'])}")
for idx, (item_id, meta) in enumerate(zip(all_results['ids'], all_results['metadatas'])):
    print(f"[{idx}] ID: {item_id}")
    print(f"     Metadata: {meta}")
