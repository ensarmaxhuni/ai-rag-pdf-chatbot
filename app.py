import re
from typing import List, Tuple

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_NAME = "AI RAG PDF Chatbot"


def extract_text_from_pdf(uploaded_file) -> str:
    """Extract readable text from a PDF file."""
    reader = PdfReader(uploaded_file)
    full_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            full_text.append(f"[Page {page_number}] {text}")

    return "\n\n".join(full_text)


def split_text_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """Split extracted document text into overlapping chunks."""
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks


def build_vector_index(chunks: List[str]):
    """Build a local TF-IDF vector index."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(chunks)
    return vectorizer, vectors


def retrieve_chunks(question: str, chunks: List[str], vectorizer, vectors, top_k: int = 4) -> List[Tuple[str, float]]:
    """Retrieve the most relevant chunks for a user question."""
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, vectors).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for index in top_indices:
        score = float(similarities[index])
        if score > 0:
            results.append((chunks[index], score))

    return results


def create_answer(question: str, retrieved_chunks: List[Tuple[str, float]]) -> str:
    """Create a grounded answer using the retrieved document sections."""
    if not retrieved_chunks:
        return (
            "I could not find enough relevant information in the uploaded PDF to answer that question. "
            "Try asking something more specific or upload a clearer text-based PDF."
        )

    answer = "### Answer based on the uploaded PDF\n\n"
    answer += (
        "The most relevant document sections related to your question are shown below. "
        "This version uses local retrieval, so the answer is grounded directly in the PDF text.\n\n"
    )

    for i, (chunk, score) in enumerate(retrieved_chunks, start=1):
        answer += f"#### Source {i} — Relevance Score: {score:.2f}\n"
        answer += f"{chunk[:1300]}...\n\n"

    answer += (
        "---\n"
        "**Note:** This project demonstrates a RAG-style workflow: PDF ingestion, chunking, retrieval, "
        "and grounded response generation. A future advanced version can add OpenAI or another LLM "
        "for richer synthesized answers."
    )
    return answer


def reset_app_state():
    """Reset uploaded document and chat state."""
    for key in ["document_text", "chunks", "vectorizer", "vectors", "file_name", "messages"]:
        if key in st.session_state:
            del st.session_state[key]


st.set_page_config(page_title=APP_NAME, page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
    .hero {
        padding: 2rem;
        border-radius: 1.5rem;
        background: linear-gradient(135deg, #020617, #1e3a8a);
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
    }
    .hero p {
        font-size: 1.05rem;
        opacity: 0.92;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>AI RAG PDF Chatbot</h1>
        <p>Upload a PDF, ask questions, and retrieve answers grounded in the document.</p>
        <p>Built with Python, Streamlit, PDF parsing, local vector search, and a RAG-style architecture.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    st.divider()
    top_k = st.slider("Sources to retrieve", min_value=2, max_value=8, value=4)
    chunk_size = st.slider("Chunk size", min_value=400, max_value=1500, value=900, step=100)

    st.divider()
    if st.button("Reset App"):
        reset_app_state()
        st.rerun()

    st.caption("Best results come from digital text-based PDFs, not scanned image PDFs.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is not None:
    if st.session_state.get("file_name") != uploaded_file.name:
        reset_app_state()
        with st.spinner("Extracting text and building retrieval index..."):
            document_text = extract_text_from_pdf(uploaded_file)
            chunks = split_text_into_chunks(document_text, chunk_size=chunk_size)

            if not chunks:
                st.error("No readable text was found in this PDF. Please try another PDF.")
            else:
                vectorizer, vectors = build_vector_index(chunks)
                st.session_state.document_text = document_text
                st.session_state.chunks = chunks
                st.session_state.vectorizer = vectorizer
                st.session_state.vectors = vectors
                st.session_state.file_name = uploaded_file.name
                st.session_state.messages = []

if "chunks" in st.session_state:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Uploaded File", st.session_state.file_name)
    with col2:
        st.metric("Document Chunks", len(st.session_state.chunks))
    with col3:
        st.metric("Approx. Words", f"{len(st.session_state.document_text.split()):,}")

    st.subheader("Document Preview")
    with st.expander("Preview extracted text"):
        st.write(st.session_state.document_text[:4000])

    st.subheader("Ask Questions")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about the uploaded PDF...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.spinner("Retrieving relevant document sections..."):
            retrieved = retrieve_chunks(
                question=question,
                chunks=st.session_state.chunks,
                vectorizer=st.session_state.vectorizer,
                vectors=st.session_state.vectors,
                top_k=top_k,
            )
            answer = create_answer(question, retrieved)

        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.divider()
    st.subheader("Manual Source Search")
    manual_query = st.text_input(
        "Search inside the uploaded PDF",
        placeholder="Example: revenue, risk factors, conclusion, termination clause",
    )

    if manual_query:
        results = retrieve_chunks(
            question=manual_query,
            chunks=st.session_state.chunks,
            vectorizer=st.session_state.vectorizer,
            vectors=st.session_state.vectors,
            top_k=top_k,
        )
        for i, (chunk, score) in enumerate(results, start=1):
            with st.expander(f"Source {i} — Relevance Score: {score:.2f}"):
                st.write(chunk[:2000])
else:
    st.info("Upload a PDF from the sidebar to begin.")
    st.subheader("What this project demonstrates")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### RAG Workflow")
        st.write("Document upload, text extraction, chunking, retrieval, and grounded responses.")
    with c2:
        st.markdown("### Document AI")
        st.write("Useful for reports, policies, research papers, contracts, and business documents.")
    with c3:
        st.markdown("### AI Portfolio Value")
        st.write("Shows practical understanding of retrieval systems and enterprise AI applications.")
