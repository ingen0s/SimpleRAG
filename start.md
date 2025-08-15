# Following Grok RAG guide for Mac.

NOTES:
- make a Docket container and auto start sh foir the core libs process.

## Create and activate a virtual environment
python -m venv rag_env
source rag_env/bin/activate

## Install core libraries
pip install torch torchvision  # For PyTorch (Apple Silicon auto-detects MPS acceleration)
pip install transformers  # For Hugging Face models like CLIP
pip install chromadb  # Vector database
pip install pillow  # For image handling
pip install ollama  # For running local LLM (optional for generation step)

