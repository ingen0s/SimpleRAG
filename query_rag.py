
import chromadb
from openai import OpenAI
client = chromadb.PersistentClient(path="./photo_db")
collection = client.get_collection(name="photos")

def generate_response(query_text, relevant_items):
    # Format item information for the prompt
    item_descriptions = "\n".join([str(item) for item in relevant_items])
    prompt = f"User query: {query_text}\nRelevant items found:\n{item_descriptions}\nDescribe what these items might show based on the query."
    # Call LM Studio's API
    client_lm = OpenAI(base_url="http://host.docker.internal:1234/v1", api_key="not-needed")
    response = client_lm.chat.completions.create(
        model="doesnt-matter",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that describes items based on provided paths and user queries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content

def main():
    print("RAG Query CLI")
    print("Type 'exit' at any prompt to quit.")
    while True:
        print("\nType your query (e.g., 'a photo of a blue car', 'video frame of a person', 'audio about trading'):")
        query = input("Query: ").strip()
        if query.lower() == 'exit':
            print("Exiting query CLI.")
            break
        print("Select query type:")
        print("  [1] Image/Video Frame")
        print("  [2] Audio Transcript")
        qtype = input("Type 1 or 2: ").strip()
        if qtype.lower() == 'exit':
            print("Exiting query CLI.")
            break
        top_k = input("How many results? (default 3): ").strip()
        if top_k.lower() == 'exit':
            print("Exiting query CLI.")
            break
        top_k = int(top_k) if top_k.isdigit() else 3

        if qtype == '1':
            import torch
            from transformers import CLIPProcessor, CLIPModel
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model.to(device)
            inputs = processor(text=query, return_tensors="pt").to(device)
            with torch.no_grad():
                query_embedding = model.get_text_features(**inputs).cpu().numpy().flatten()
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where={"type": {"$in": ["image", "video_frame"]}}
            )
            frames = results['metadatas'][0]
            print("Top matches (video frames):")
            for frame in frames:
                print(f"  Frame: {frame.get('path')} @ {frame.get('timestamp')}")
                # Find nearest transcript segment for this frame
                transcript_results = collection.get(where={"type": "audio_transcript", "path": frame.get('path')})
                nearest = None
                min_diff = float('inf')
                for seg in transcript_results['metadatas']:
                    diff = abs(seg.get('timestamp', 0) - frame.get('timestamp', 0))
                    if diff < min_diff:
                        min_diff = diff
                        nearest = seg
                if nearest:
                    print(f"    Nearest transcript: [{nearest.get('timestamp')}] {nearest.get('text')}")
            if frames:
                response = generate_response(query, [f"Frame: {f.get('path')} @ {f.get('timestamp')}" for f in frames])
                print("LLM response:", response)
        elif qtype == '2':
            results = collection.get(
                where={"type": "audio_transcript"},
                limit=top_k
            )
            transcripts = results['metadatas']
            print("Top transcript segments:")
            for seg in transcripts:
                print(f"  Transcript: [{seg.get('timestamp')}] {seg.get('text')}")
                # Find nearest video frame for this transcript
                frame_results = collection.get(where={"type": "video_frame", "path": seg.get('path')})
                nearest = None
                min_diff = float('inf')
                for frame in frame_results['metadatas']:
                    diff = abs(frame.get('timestamp', 0) - seg.get('timestamp', 0))
                    if diff < min_diff:
                        min_diff = diff
                        nearest = frame
                if nearest:
                    print(f"    Nearest frame: {nearest.get('path')} @ {nearest.get('timestamp')}")
            if transcripts:
                response = generate_response(query, [f"Transcript: [{s.get('timestamp')}] {s.get('text')}" for s in transcripts])
                print("LLM response:", response)
        else:
            print("Invalid query type.")

if __name__ == "__main__":
    main()