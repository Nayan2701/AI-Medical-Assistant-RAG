import streamlit as st

from src.config import settings
from src.rag.ingest import load_reports, split_reports
from src.rag.index import build_or_load_vectorstore
from src.rag.qa import make_llm, retrieve, generate_answer, build_citation_block


st.set_page_config(page_title="Medical Report Search", layout="centered")
st.title("🔎 Medical Report Search")

st.caption(
    "RAG-based search over a local JSON dataset. "
    "Make sure your dataset exists locally; this app will not download data automatically."
)


@st.cache_resource
def get_retriever():
    reports = load_reports(settings.dataset_path)
    chunks = split_reports(reports)

    vs = build_or_load_vectorstore(
        chunks=chunks,
        output_dir=settings.faiss_dir,
        index_name=settings.faiss_index_name,
        embedding_model_name=settings.hf_embedding_model,
    )
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": settings.k})


@st.cache_resource
def get_llm():
    return make_llm(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )


user_query = st.text_input("Enter the patient's name or the information you want.")

if st.button("Start Search"):
    if not user_query:
        st.warning("Please enter a research question.")
    else:
        try:
            retriever = get_retriever()
            llm = get_llm()
        except Exception as e:
            st.error(str(e))
            st.stop()

        with st.spinner("Fetching relevant context and generating an answer…"):
            chunks = retrieve(retriever, user_query, k=settings.k)
            answer = generate_answer(llm, user_query, chunks)

        st.success("Done ✅")
        st.markdown(answer)

        with st.expander("📖 Show retrieved sources"):
            st.text(build_citation_block(chunks))