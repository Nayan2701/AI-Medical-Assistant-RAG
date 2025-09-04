import os
import streamlit as st
import numpy as np
import requests
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing import Any, List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI


# Step 1: Download JSON from Google Drive if not already present

gdrive_url = "https://drive.google.com/file/d/1op1s7vX9vf9u--mQ_DidFPNqYInZQPZt/view?usp=sharing"
local_file = Path("dataset/patients_data.json")

# Make sure dataset directory exists
local_file.parent.mkdir(parents=True, exist_ok=True)

if not local_file.exists():
    st.info("Downloading dataset from Google Drive...")
    r = requests.get(gdrive_url)
    if r.status_code == 200:
        with open(local_file, "wb") as f:
            f.write(r.content)
        st.success("Dataset downloaded successfully ✅")
    else:
        st.error("Failed to download dataset. Please check the Google Drive link.")


# Load JSON file
loader = JSONLoader(
    str(local_file),
    jq_schema='.',
    text_content=False,
)
reports = loader.load()

# Defining the text_splitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\nmedical_history:", "\nchronic_conditions:", "\nallergies:", "\nclinical_notes:", "\n\n", "\n", " ", ""],
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = text_splitter.split_documents(reports)

# Step 2: Vectorization and FAISS Indexing 
output_dir = "faiss_index"
faiss_index_name = "medical_index"

if not os.path.exists(os.path.join(output_dir, f"{faiss_index_name}.faiss")):
    emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = FAISS.from_documents(chunks, emb)
    vs.save_local(output_dir, index_name=faiss_index_name)
    print("FAISS index created and saved.")
else:
    print("FAISS index already exists. Skipping creation.")

# Load saved FAISS index
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vs = FAISS.load_local(output_dir, emb, index_name=faiss_index_name, allow_dangerous_deserialization=True)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Step 3: RAG Setup
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # or "gemini-1.5-pro"
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful medical assistant.\n"
     "Use ONLY the information in <CONTEXT>. If the context is insufficient, say so.\n"
     "Never fabricate data. Return concise, accurate answers with citations like [S1], [S2].\n"
     "If asked about a patient, summarize facts from context and avoid speculative clinical advice.\n"
     "This is not medical diagnosis; include a brief caution if the user asks for treatment decisions."
    ),
    ("human",
     "User question:\n{question}\n\n"
     "<CONTEXT>\n{context}\n</CONTEXT>\n\n"
     "Citations:\n{citations}\n\n"
     "Instructions:\n"
     "1) If the answer is fully supported by context, answer and cite.\n"
     "2) If partially supported, clearly state limits and cite only supported parts.\n"
     "3) If unsupported, say: 'The provided context does not contain that information.'")
])

def retrieve(query: str, k: int = 6) -> List[Dict[str, Any]]:
    docs = retriever.get_relevant_documents(query)
    out = []
    for i, d in enumerate(docs[:k], 1):
        out.append({
            "rank": i,
            "content": d.page_content.strip(),
            "source": d.metadata.get("source") or d.metadata.get("path") or "unknown",
            "page": d.metadata.get("page") or d.metadata.get("page_number")
        })
    return out

def build_context(chunks: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    pieces = []
    used = 0
    for i, ch in enumerate(chunks, 1):
        tag = f"[S{i}]"
        text = ch["content"].replace("\n", " ").strip()
        block = f"{tag} {text}"
        if used + len(block) > max_chars:
            break
        pieces.append(block)
        used += len(block)
    return "\n\n".join(pieces)

def build_citation_block(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, ch in enumerate(chunks, 1):
        page = f", p.{ch['page']}" if ch.get("page") is not None else ""
        lines.append(f"[S{i}] {ch['source']}{page}")
    return "\n".join(lines) if lines else "No sources."

def generate_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return "The provided context does not contain that information."
    context = build_context(chunks)
    citations = build_citation_block(chunks)
    msg = RAG_PROMPT.format_messages(question=question, context=context, citations=citations)
    resp = llm(msg)
    return resp.content.strip()

# Streamlit App
st.title("🔎 Medical Report Search")
user_query = st.text_input("Enter the patient's name or the information you want.")

if st.button("Start Search"):
    if user_query:
        with st.spinner("Fetching relevant context and generating an answer…"):
            chunks = retrieve(user_query)
            answer = generate_answer(user_query, chunks)
        st.success("Done ✅")
        st.markdown(answer)

        with st.expander("📖 Show retrieved sources"):
            st.text(build_citation_block(chunks))
    else:
        st.warning("Please enter a research question.")