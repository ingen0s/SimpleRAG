# SimpleRAG

SimpleRAG is an advanced Retrieval-Augmented Generation (RAG) pipeline for images, videos, and audio, with special support for financial and trading chart analysis. It embeds, transcribes, captions, and analyzes multimedia data—including extracting chart text, timeframes, and candlestick patterns from trading videos—then stores everything in a vector database (ChromaDB) for fast querying and interactive visualization. The project is designed for rapid experimentation, extensibility, and deep inspection of your data and model outputs, making it ideal for research, trading analytics, and multimedia search.

## Key Capabilities
- **Image & Video Embedding:** Uses CLIP to generate embeddings for images and video frames.
- **Automatic Captioning:** Uses BLIP to generate descriptive captions for images and video frames.
- **Audio Transcription:** Uses Whisper to transcribe audio from videos.
- **Chart OCR & Timeframe Extraction:** Uses Tesseract OCR to extract visible chart text (e.g., pair names, timeframe labels) from each frame. Timeframe (e.g., 1m, 5m, 1h, 1d) is automatically detected and stored in metadata.
- **Candlestick Detection:** Uses OpenCV to detect candlestick patterns and chart elements in video frames. Candlestick bounding boxes are stored in metadata for further analysis.
- **Vector Database:** Stores all embeddings and rich metadata in ChromaDB for fast retrieval and inspection.
- **Interactive CLI:** Add, query, and manage models (collections) with a simple command-line interface. Delete models as needed.
- **Visualization:** Inspect metadata in the terminal or as a 3D interactive graph (GUI).
- **Docker & Local Support:** Works in Docker and on local machines, with environment variable support for data paths.

## Docker Setup

You can run the entire pipeline in a Docker container for maximum compatibility:

1. Build the Docker image:
    ```bash
    docker build -t simplerag .
    ```

2. Run the container (mounts your data folders and sets environment variables for compatibility):
    ```bash
    docker run -it --rm \
        --network=host \
        -v /Users/igorkomolov/Movies/RAG_Videos:/app/Movies/RAG_Videos \
        -v /Users/igorkomolov/Pictures/RAG_Photos:/app/Pictures/RAG_Photos \
        -v "$PWD/photo_db:/app/photo_db" \
        -e RAG_VIDEO_FOLDER=/app/Movies/RAG_Videos \
        -e RAG_PHOTO_FOLDER=/app/Pictures/RAG_Photos \
        simplerag
    ```

This will run `build_rag.py` in a clean Python 3.12 environment with all dependencies pre-installed, and ensure your host data folders are accessible inside the container.

You can modify the Dockerfile or entrypoint as needed for other scripts or workflows.

**Note:**
- The script supports `RAG_VIDEO_FOLDER` and `RAG_PHOTO_FOLDER` environment variables for folder paths. This makes it easy to use custom or mounted directories in Docker.

---

See start.sh / start.md for manual setup instructions (if not using Docker).

## Features & Usage Updates (August 2025)

### Model Management
- Delete models (collections) from the CLI. Select `[d]` and choose the model to delete.
- The pipeline skips already-processed videos and images, preventing duplicate data.
- If all videos are processed, the script exits early with a message.

### Chart & Video Frame Analysis
- Each video frame is automatically captioned using BLIP (Salesforce/blip-image-captioning-base).
- Tesseract OCR extracts visible chart text (pair names, timeframe labels, etc.) from each frame.
- Chart timeframe (e.g., 1m, 5m, 1h, 1d) is detected and stored in metadata.
- OpenCV detects candlestick patterns and chart elements; bounding boxes are stored in metadata.
- All extracted info is stored in metadata and can be inspected with visualization scripts.

### Visualization Tools
- `visualize_metadata.py`: Inspect metadata for any model, including captions, OCR text, timeframe, and candlestick data for video frames.
- `visualize_metadata_3d.py`: Explore your model metadata as a 3D interactive graph (requires PyQt5, PyQtWebEngine, and Plotly).

### Requirements & Installation Notes
- For GUI/3D visualization, install:
  ```sh
  pip install PyQt5 PyQtWebEngine plotly networkx
  ```
- For chart OCR and analysis, install:
  ```sh
  pip install pytesseract opencv-python
  ```
- On ARM/Docker, you may need to pin PyQt5 to a wheel version (e.g. `PyQt5==5.15.9`) and install system Qt libraries.
- BLIP, CLIP, Whisper, ffmpeg-python, and chromadb are required for the main pipeline.

### Example Workflow
1. Run `python build_rag.py` and select a model or create a new one.
2. Add data: images and videos are embedded, transcribed, captioned, and chart info is extracted.
3. Use `[d]` to delete a model if needed.
4. Inspect metadata with `python visualize_metadata.py`.
5. Explore in 3D with `python visualize_metadata_3d.py` (GUI required).

---

## Vector DB
- /vectordb
- researching how Vector Database works




