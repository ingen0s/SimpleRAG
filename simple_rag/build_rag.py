import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import chromadb
from pathlib import Path

# Load CLIP model (downloads ~500MB on first run)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Path to your photo folder
photo_folder = "/Users/igorkomolov/Pictures/RAG_Photos"  # Your specified path

# Recursively find all .jpg, .jpeg, and .png files
photo_folder = os.path.expanduser(photo_folder)
photos = [
    str(p) for p in Path(photo_folder).rglob("*")
    if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
]

# Initialize Chroma vector database
client = chromadb.PersistentClient(path="./photo_db")  # Saves to local folder
collection = client.get_or_create_collection(name="photos")

# Process and embed photos
device = "mps" if torch.backends.mps.is_available() else "cpu"  # Use MPS on Mac for speed
model.to(device)

for idx, photo_path in enumerate(photos):
    try:
        image = Image.open(photo_path).convert("RGB")
    except Exception as e:
        print(f"Skipping corrupted image: {photo_path}: {e}")
        continue
    
    # Prepare image for CLIP
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Generate embedding
    with torch.no_grad():
        embedding = model.get_image_features(**inputs).cpu().numpy().flatten()
    
    # Store in Chroma with metadata (e.g., file path)
    collection.add(
        embeddings=[embedding.tolist()],
        metadatas=[{"path": str(photo_path)}],
        ids=[f"photo_{idx}"]
    )
    print(f"Processed: {photo_path}")

print("Embeddings stored successfully!")