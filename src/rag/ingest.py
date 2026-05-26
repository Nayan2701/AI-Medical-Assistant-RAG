from __future__ import annotations

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.json_loader import JSONLoader


def load_reports(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {json_path}. "
            "Place patients_data.json there (or set DATASET_PATH)."
        )

    loader = JSONLoader(
        str(json_path),
        jq_schema=".",
        text_content=False,
    )
    return loader.load()


def split_reports(reports):
    splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\nmedical_history:",
            "\nchronic_conditions:",
            "\nallergies:",
            "\nclinical_notes:",
            "\n\n",
            "\n",
            " ",
            "",
        ],
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_documents(reports)