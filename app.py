import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import numpy as np
from typing import List

# -----------------------------
# Helpers
# -----------------------------


def init_session_state():
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None  # numpy array [num_chunks, dim]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    all_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            all_text.append(text)
    return "\n".join(all_text)


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunking with overlap.
    """
    text = text.replace("\r", " ")
    text = " ".join(text.split())  # normalize spaces

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_chars - overlap
    return chunks


def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """
    Create embeddings for a list of texts and return a 2D numpy array.
    """
    if not texts:
        return np.zeros((0,))

    resp = client.embeddings.create(
        model=model,
        input=texts
    )
    vectors = [np.array(d.embedding, dtype="float32") for d in resp.data]

    # Stack into 2D matrix [num_texts, embedding_dim]
    mat = np.vstack(vectors)

    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
    mat = mat / norms
    return mat


def embed_query(client: OpenAI, query: str, model: str = "text-embedding-3-small") -> np.ndarray:
    resp = client.embeddings.create(
        model=model,
        input=[query]
    )
    v = np.array(resp.data[0].embedding, dtype="float32")
    v = v / (np.linalg.norm(v) + 1e-10)  # normalize
    return v


def retrieve_relevant_chunks(query_vector: np.ndarray, k: int = 4) -> List[str]:
    """
    Given a query vector and stored embeddings in session_state,
    return the top-k most similar chunks.
    """
    if st.session_state.embeddings is None or len(st.session_state.chunks) == 0:
        return []

    doc_embeddings = st.session_state.embeddings  # [num_chunks, dim]

    # Compute cosine similarity with matrix multiplication
    sims = doc_embeddings @ query_vector  # [num_chunks]
    top_k_indices = sims.argsort()[-k:][::-1]  # highest first

    return [st.session_state.chunks[i] for i in top_k_indices]


def build_rag_prompt(question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are a helpful assistant. You answer user questions using ONLY the context from the document.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User question: {question}"
    )
    return prompt


def call_llm(client: OpenAI, question: str, context_chunks: List[str], model: str = "gpt-4.1-mini") -> str:
    prompt = build_rag_prompt(question, context_chunks)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a document Q&A assistant using RAG."},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


# -----------------------------
# Streamlit UI
# -----------------------------


def main():
    st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")
    st.title("📄 Chat with your PDF (RAG)")

    init_session_state()

    # Sidebar for settings / API key
    st.sidebar.header("Settings")

    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Your OpenAI API key (it is only used in this session).",
    )

    default_model = "gpt-4.1-mini"
    llm_model = st.sidebar.text_input(
        "Chat model",
        value=default_model,
        help="Any chat-capable OpenAI model, e.g. gpt-4.1, gpt-4.1-mini, gpt-4o, etc.",
    )

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    client = OpenAI(api_key=api_key)

    st.subheader("1️⃣ Upload your PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner("Extracting and indexing your PDF..."):
            # Extract text
            full_text = extract_text_from_pdf(uploaded_file)
            if not full_text.strip():
                st.error("No text could be extracted from this PDF.")
                st.stop()

            # Chunk text
            chunks = chunk_text(full_text)

            # Embed chunks
            embeddings = embed_texts(client, chunks)

            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings

        st.success(f"Indexed {len(st.session_state.chunks)} chunks from the PDF ✅")

    st.subheader("2️⃣ Chat with your document")

    if st.session_state.embeddings is None:
        st.info("Upload a PDF first to start chatting.")
        return

    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask something about your PDF...")
    if user_input:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Retrieve context and answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                query_vec = embed_query(client, user_input)
                context_chunks = retrieve_relevant_chunks(query_vec, k=4)
                if not context_chunks:
                    answer = "I couldn't find any relevant information in the document."
                else:
                    answer = call_llm(client, user_input, context_chunks, model=llm_model)
                st.markdown(answer)

        # Save assistant message
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
