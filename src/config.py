import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Data
    dataset_path: Path = Path(os.getenv("DATASET_PATH", "dataset/patients_data.json"))

    # FAISS
    faiss_dir: Path = Path(os.getenv("FAISS_DIR", "faiss_index"))
    faiss_index_name: str = os.getenv("FAISS_INDEX_NAME", "medical_index")

    # Retrieval
    k: int = int(os.getenv("RETRIEVER_K", "6"))

    # Embeddings
    hf_embedding_model: str = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # LLM (Gemini)
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    google_api_key: str | None = os.getenv("GOOGLE_API_KEY")


settings = Settings()