import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import chromadb
from pathlib import Path
import sys
import cv2
import tempfile
import numpy as np
import whisper
import ffmpeg
import pytesseract
import json
import re
import requests
import base64

# Allow override of photo and video folders via environment variables for Docker compatibility
photo_folder = os.environ.get("RAG_PHOTO_FOLDER", os.path.expanduser("~/Pictures/RAG_Photos"))
video_folder = os.environ.get("RAG_VIDEO_FOLDER", os.path.expanduser("~/Movies/RAG_Videos"))

# LM Studio API configuration
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

def find_photos():
    return [
        str(p) for p in Path(photo_folder).rglob("*")
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ]

def find_videos():
    return [
        str(p) for p in Path(video_folder).rglob("*")
        if p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')
    ]

def select_or_create_collection(client_db):
    collections = client_db.list_collections()
    print("\nAvailable RAG models (collections):")
    for idx, col in enumerate(collections):
        print(f" [{idx}] {col.name}")
    print(" [n] Create new model")
    print(" [d] Delete a model")
    choice = input("Select a model by number, 'n' to create new, or 'd' to delete: ").strip()
    if choice.lower() == 'n':
        new_name = input("Enter new model name: ").strip()
        return client_db.get_or_create_collection(name=new_name)
    elif choice.lower() == 'd':
        del_idx = input("Enter the number of the model to delete: ").strip()
        try:
            del_idx = int(del_idx)
            del_name = collections[del_idx].name
            confirm = input(f"Are you sure you want to delete model '{del_name}'? (y/N): ").strip().lower()
            if confirm == 'y':
                client_db.delete_collection(name=del_name)
                print(f"Model '{del_name}' deleted.")
            else:
                print("Delete cancelled.")
        except Exception:
            print("Invalid selection.")
        sys.exit(0)
    try:
        idx = int(choice)
        return client_db.get_collection(name=collections[idx].name)
    except Exception:
        print("Invalid selection. Exiting.")
        sys.exit(1)

def encode_image_to_base64(image):
    """Convert PIL image to base64 string for LM Studio API."""
    buffered = tempfile.NamedTemporaryFile(suffix=".png")
    image.save(buffered.name, format="PNG")
    with open(buffered.name, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def query_gemma3(image, prompt):
    """Query Gemma 3 via LM Studio API for image description."""
    base64_image = encode_image_to_base64(image)
    payload = {
        "model": "gemma3:12b-it-qat",  # Adjust based on your loaded model
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 500,  # Increased for more detailed response
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64
    }
    try:
        response = requests.post(LM_STUDIO_API_URL, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error querying Gemma 3 API: {e}")
        return ""

def extract_time_from_image(image):
    """Extract time with UTC+1 offset from the bottom right corner using OCR."""
    # Define the bottom right region (e.g., last 10% of width and height)
    width, height = image.size
    region_width = int(width * 0.1)
    region_height = int(height * 0.1)
    bottom_right = image.crop((width - region_width, height - region_height, width, height))
    
    # Use OCR to extract text
    ocr_text = pytesseract.image_to_string(bottom_right).strip()
    
    # Look for time pattern (e.g., "14:30 UTC+1")
    time_match = re.search(r'(\d{1,2}:\d{2}\s*UTC\+1)', ocr_text, re.IGNORECASE)
    if time_match:
        return time_match.group(1)
    return "Unknown"

def extract_ticker_from_image(image):
    """Extract ticker symbol from the image using OCR as a fallback."""
    # Define a region near the top left (adjust based on chart layout)
    width, height = image.size
    region_width = int(width * 0.3)
    region_height = int(height * 0.1)
    top_left = image.crop((0, 0, region_width, region_height))
    
    # Use OCR to extract text
    ocr_text = pytesseract.image_to_string(top_left).strip()
    
    # Look for ticker pattern (e.g., "BTC/USDT")
    ticker_match = re.search(r'([A-Z]{3,5}/[A-Z]{3,5})', ocr_text, re.IGNORECASE)
    if ticker_match:
        return ticker_match.group(1)
    return "Unknown"

def parse_gemma3_response(chart_description):
    """Parse structured fields from Gemma 3 response with improved robustness."""
    print(f"Raw Gemma 3 Response: {chart_description}")  # Debugging output
    parsed = {
        "ticker": "Unknown",
        "chart_timeframe": "Unknown",
        "current_price": "Unknown",
        "time": "Unknown",
        "candlestick_patterns": "Unknown",
        "technical_indicators": "Unknown",
        "price_action": "Unknown",
        "support_resistance_levels": "Unknown",
        "trend_lines": "Unknown",
        "vix_levels": "Unknown",
        "smart_money_analysis": "Unknown"
    }
    
    # Flexible regex patterns to handle variations
    patterns = [
        (r'(?:Ticker Symbol|Symbol):?\s*([^\n]+)', "ticker"),
        (r'(?:Timeframe):?\s*([^\n]+)', "chart_timeframe"),
        (r'(?:Current Price|Price):?\s*([^\n]+)', "current_price"),
        (r'(?:Time \(Bottom Right\)|Time):?\s*([^\n]+)', "time"),
        (r'(?:Candlestick Pattern|Patterns):?\s*([^\n]+)(?=\n(?:Technical Indicators|$))', "candlestick_patterns"),
        (r'(?:Technical Indicators|Indicators):?\s*([^\n]+)(?=\n(?:Price Action Chart|$))', "technical_indicators"),
        (r'(?:Price Action Chart|Price Action):?\s*([^\n]+)(?=\n(?:Support and Resistance Levels|$))', "price_action"),
        (r'(?:Support and Resistance Levels|Support/Resistance):?\s*([^\n]+)(?=\n(?:Trend Lines|$))', "support_resistance_levels"),
        (r'(?:Trend Lines|Trends):?\s*([^\n]+)(?=\n(?:VIX Levels|$))', "trend_lines"),
        (r'(?:VIX Levels|VIX):?\s*([^\n]+)(?=\n(?:Smart Money Analysis|$))', "vix_levels"),
        (r'(?:Smart Money Analysis|Smart Money):?\s*([^\n]+)', "smart_money_analysis")
    ]
    
    for pattern, key in patterns:
        match = re.search(pattern, chart_description, re.IGNORECASE | re.DOTALL)
        if match:
            parsed[key] = match.group(1).strip()
    
    return parsed

def main():
    print("What would you like to do?")
    print(" [1] Add data to a model (existing or new)")
    print(" [2] Query a model")
    action = input("Select 1 or 2: ").strip()
    client_db = chromadb.PersistentClient(path="./photo_db")
    if action == '2':
        # Query mode: select model, then run query script
        collection = select_or_create_collection(client_db)
        import subprocess
        print("Launching query interface...")
        subprocess.run([sys.executable, "query_rag.py"])
        sys.exit(0)
    elif action == '1':
        # Add data mode: select model, then continue as before
        collection = select_or_create_collection(client_db)
        # Defer heavy model loading until here
        print("Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        photos = find_photos()
        videos = find_videos()
        print("\nDiscovered video files:")
        for v in videos:
            print(f" {v}")
        if not videos:
            print("No video files found. Check your path and file extensions.")
        # --- Images ---
        for idx, photo_path in enumerate(photos):
            try:
                image = Image.open(photo_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    embedding = model.get_image_features(**inputs).cpu().numpy().flatten()
                # Extract time and ticker from image as fallback
                extracted_time = extract_time_from_image(image)
                extracted_ticker = extract_ticker_from_image(image)
                # Gemma 3 description via LM Studio API
                prompt = (
                    "Analyze this image. If it's a trading chart, extract and describe the following in this exact structured format: "
                    "**Ticker Symbol:** [symbol], "
                    "**Timeframe:** [timeframe], "
                    "**Current Price:** [price], "
                    "**Time (Bottom Right):** [HH:MM UTC+1], "
                    "**Candlestick Pattern:** [description of patterns], "
                    "**Technical Indicators:** [list and description], "
                    "**Price Action Chart:** [description], "
                    "**Support and Resistance Levels:** [levels and description], "
                    "**Trend Lines:** [description], "
                    "**VIX Levels:** [levels and description], "
                    "**Smart Money Analysis:** [analysis]. "
                    "Otherwise, provide a general description."
                )
                chart_description = query_gemma3(image, prompt)
                # Parse structured response
                parsed = parse_gemma3_response(chart_description)
                # Override with OCR fallbacks if needed
                if parsed["time"] == "Unknown":
                    parsed["time"] = extracted_time
                if parsed["ticker"] == "Unknown":
                    parsed["ticker"] = extracted_ticker
                collection.add(
                    embeddings=[embedding.tolist()],
                    metadatas=[{
                        "path": photo_path,
                        "type": "image",
                        "chart_description": chart_description,
                        "ticker": parsed["ticker"],
                        "current_price": parsed["current_price"],
                        "chart_timeframe": parsed["chart_timeframe"],
                        "time": parsed["time"],
                        "candlestick_patterns": parsed["candlestick_patterns"],
                        "technical_indicators": parsed["technical_indicators"],
                        "price_action": parsed["price_action"],
                        "support_resistance_levels": parsed["support_resistance_levels"],
                        "trend_lines": parsed["trend_lines"],
                        "vix_levels": parsed["vix_levels"],
                        "smart_money_analysis": parsed["smart_money_analysis"]
                    }],
                    ids=[f"photo_{idx}"]
                )
            except Exception as ie:
                print(f" Skipping image {photo_path}: {ie}")
        # --- Videos ---
        print("Loading Whisper model...")
        try:
            whisper_model = whisper.load_model("base")
            print("Whisper model loaded successfully.")
        except Exception as werr:
            print(f"Failed to load Whisper model: {werr}")
            sys.exit(1)
        # Check for unprocessed videos
        unprocessed_videos = []
        for video_path in videos:
            existing = collection.get(where={"path": video_path}, include=["metadatas"])
            already_processed = False
            for meta in existing.get("metadatas", []):
                if meta.get("type") == "video_frame":
                    already_processed = True
                    break
            if not already_processed:
                unprocessed_videos.append(video_path)
        if not unprocessed_videos:
            print("No video files found. Exiting.")
            return
        for vidx, video_path in enumerate(unprocessed_videos):
            try:
                print(f"\nProcessing video: {video_path}")
                # Check if video frames already exist in collection
                existing = collection.get(where={"path": video_path}, include=["metadatas"])
                already_processed = False
                for meta in existing.get("metadatas", []):
                    if meta.get("type") == "video_frame":
                        already_processed = True
                        break
                if already_processed:
                    print(f" Skipping video (already processed): {video_path}")
                    continue
                ext = os.path.splitext(video_path)[1].lower()
                if ext not in ['.mp4', '.mov', '.avi', '.mkv']:
                    print(f" Skipping unsupported video extension: {ext}")
                    continue
                if not os.path.exists(video_path):
                    print(f" Video file does not exist: {video_path}")
                    continue
                # Extract frames
                vidcap = cv2.VideoCapture(video_path)
                if not vidcap.isOpened():
                    print(f" Failed to open video file: {video_path}")
                    continue
                fps = vidcap.get(cv2.CAP_PROP_FPS)
                frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps else 0
                print(f" FPS: {fps}, Frame count: {frame_count}, Duration: {duration}s")
                frame_embeddings = []
                frame_metadatas = []
                frame_ids = []
                frame_times = []
                # Sample 1 frame every 10 seconds for speed
                frame_interval = max(int(fps * 10), 1)  # 1 frame every 10 seconds
                import time
                frame_total = 0
                start_time = time.time()
                last_time = start_time
                sampled_frames = list(range(0, frame_count, frame_interval))
                for frame_idx in sampled_frames:
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = vidcap.read()
                    if not ret:
                        print(f" Could not read frame {frame_idx} from video: {video_path}")
                        continue
                    try:
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        # CLIP embedding
                        inputs = processor(images=pil_img, return_tensors="pt").to(device)
                        with torch.no_grad():
                            emb = model.get_image_features(**inputs).cpu().numpy().flatten()
                        # Extract time and ticker from image as fallback
                        extracted_time = extract_time_from_image(pil_img)
                        extracted_ticker = extract_ticker_from_image(pil_img)
                        # Gemma 3 description via LM Studio API
                        prompt = (
                            "Analyze this trading chart image. Extract and describe the following in this exact structured format: "
                            "**Ticker Symbol:** [symbol], "
                            "**Timeframe:** [timeframe], "
                            "**Current Price:** [price], "
                            "**Time (Bottom Right):** [HH:MM UTC+1], "
                            "**Candlestick Pattern:** [description of patterns], "
                            "**Technical Indicators:** [list and description], "
                            "**Price Action Chart:** [description], "
                            "**Support and Resistance Levels:** [levels and description], "
                            "**Trend Lines:** [description], "
                            "**VIX Levels:** [levels and description], "
                            "**Smart Money Analysis:** [analysis]."
                        )
                        chart_description = query_gemma3(pil_img, prompt)
                        # Parse structured response
                        parsed = parse_gemma3_response(chart_description)
                        # Override with OCR fallbacks if needed
                        if parsed["time"] == "Unknown":
                            parsed["time"] = extracted_time
                        if parsed["ticker"] == "Unknown":
                            parsed["ticker"] = extracted_ticker
                        # Initialize candlestick_info for this frame
                        candlestick_info = []
                        # Candlestick info (fallback to OpenCV)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            x, y, w, h = cv2.boundingRect(cnt)
                            if h > w * 2 and w > 5 and h > 20:
                                cs_region = frame[y:y+h, x:x+w]
                                avg_color = np.mean(cs_region, axis=(0,1)).astype(int)
                                cs_type = 'bullish' if avg_color[1] > avg_color[2] + 50 else 'bearish' if avg_color[2] > avg_color[1] + 50 else 'neutral'
                                candlestick_info.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'type': cs_type})
                        ts = frame_idx / fps if fps else 0
                        frame_embeddings.append(emb.tolist())
                        frame_metadatas.append({
                            "path": video_path,
                            "type": "video_frame",
                            "timestamp": ts,
                            "chart_description": chart_description,
                            "ticker": parsed["ticker"],
                            "current_price": parsed["current_price"],
                            "chart_timeframe": parsed["chart_timeframe"],
                            "time": parsed["time"],
                            "candlestick_patterns": parsed["candlestick_patterns"],
                            "technical_indicators": parsed["technical_indicators"],
                            "price_action": parsed["price_action"],
                            "support_resistance_levels": parsed["support_resistance_levels"],
                            "trend_lines": parsed["trend_lines"],
                            "vix_levels": parsed["vix_levels"],
                            "smart_money_analysis": parsed["smart_money_analysis"],
                            "candlesticks": json.dumps(candlestick_info)
                        })
                        frame_ids.append(f"video_{vidx}_frame_{frame_idx}")
                        frame_times.append(ts)
                        frame_total += 1
                        if frame_total % 100 == 0:
                            now = time.time()
                            batch_time = now - last_time
                            elapsed = now - start_time
                            frames_left = len(sampled_frames) - frame_total
                            if frame_total > 0:
                                avg_time_per_frame = elapsed / frame_total
                                eta = avg_time_per_frame * frames_left
                            print(f" Processed {frame_total}/{len(sampled_frames)} frames. Last 100: {batch_time:.1f}s, Elapsed: {elapsed:.1f}s, ETA: {eta/60:.1f} min")
                            last_time = now
                    except Exception as fe:
                        print(f" Skipping frame {frame_idx}: {fe}")
                vidcap.release()
                # Store frame embeddings only if there is data
                if frame_embeddings and frame_metadatas and frame_ids:
                    collection.add(
                        embeddings=frame_embeddings,
                        metadatas=frame_metadatas,
                        ids=frame_ids
                    )
                    print(f" Stored {frame_total} frames for video: {video_path}")
                else:
                    print(f" No valid data to store for video: {video_path}")
                # Extract and transcribe audio using ffmpeg-python
                audio_segments_total = 0
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                    try:
                        (
                            ffmpeg
                            .input(video_path)
                            .output(tmp_audio.name, format='wav', acodec='pcm_s16le', ac=1, ar='16000')
                            .overwrite_output()
                            .run(quiet=True)
                        )
                        result = whisper_model.transcribe(tmp_audio.name)
                        transcript = result["text"]
                        segments = result.get("segments", [])
                        # Store transcript segments with timestamps
                        for sidx, seg in enumerate(segments):
                            collection.add(
                                embeddings=[np.zeros_like(frame_embeddings[0]).tolist()] if frame_embeddings else [np.zeros(512).tolist()],  # dummy embedding
                                metadatas=[{"path": video_path, "type": "audio_transcript", "timestamp": seg["start"], "text": seg["text"]}],
                                ids=[f"video_{vidx}_audio_{sidx}"]
                            )
                            audio_segments_total += 1
                        print(f" Stored {audio_segments_total} audio transcript segments for video: {video_path}")
                    except Exception as ae:
                        print(f" Skipping audio extraction/transcription: {ae}")
                print(f"Finished processing video: {video_path}")
            except Exception as e:
                print(f"Skipping video {video_path}: {e}")
                continue
        print("All embeddings and transcripts stored successfully!")
    else:
        print("Invalid selection. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()