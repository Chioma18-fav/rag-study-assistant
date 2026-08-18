import pdfplumber
import math
import hashlib
import os
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

st.title("My RAG App")

# history: stores the chat log shown on screen for THIS browser tab
if "history" not in st.session_state:
    st.session_state.history = []

# session_id: a random ID unique to this browser session, used later if you
# want to scope retrieval strictly per-user (not required right now since we
# already scope by doc_id, but kept here for future use)
if "session_id" not in st.session_state:
    st.session_state.session_id = os.urandom(8).hex()


# Load models once, not on every rerun
@st.cache_resource
def load_model():
    # This is the SentenceTransformer model — already trained by someone else.
    # We are NOT training anything here, just using it to turn text into vectors.
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_chroma():
    # PersistentClient writes vectors to disk in ./chroma_db so they survive
    # an app restart — no need to re-upload/re-embed the same PDF again.
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection("my_documents")

model = load_model()
collection = load_chroma()


# ---- PDF loading ----
def load_pdf_file(uploaded_file):
    # uploaded_file comes from st.file_uploader — pdfplumber can read it
    # directly without needing a saved file path on disk.
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


# ---- Chunking (fixed-size with overlap — Concept 11) ----
def chunking(text):
    list_chunk = []
    start = 0
    while start < len(text):
        chunk = text[start:start + 500]   # grab 500 characters
        start += 400                      # move forward 400 -> 100 char overlap with next chunk
        list_chunk.append(chunk)
    return list_chunk


# ---- Fingerprint the uploaded file so we don't re-embed the same PDF twice ----
def get_file_hash(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    return hashlib.md5(file_bytes).hexdigest()


# ---- Indexing: chunk -> embed -> store in Chroma ----
def index_document(doc_id, chunks):
    # Check if this exact file (by hash) was already indexed before.
    # If yes, skip re-embedding entirely — saves time on repeat uploads.
    existing = collection.get(where={"doc_id": doc_id})
    if len(existing["ids"]) > 0:
        return

    chunk_vectors = model.encode(chunks)  # turn each text chunk into a vector
    collection.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        embeddings=chunk_vectors.tolist(),
        documents=chunks,
        metadatas=[{"doc_id": doc_id} for _ in chunks],  # tags each chunk with its source doc
    )


# ---- YOUR original hand-rolled cosine similarity math (unchanged logic) ----
def magnitude_vector(vector):
    sum_squared_values = 0
    for value in vector:
        sum_squared_values += pow(value, 2)
    return math.sqrt(sum_squared_values)

def cosine_similarity(chunk_vector, quest_vector):
    dot_product = 0
    for doc, ques in zip(chunk_vector, quest_vector):
        dot_product += doc * ques
    return dot_product / (magnitude_vector(chunk_vector) * magnitude_vector(quest_vector))


# ---- Retrieval: pull candidate chunks for this doc out of Chroma,
#      then rank them ourselves using our own cosine similarity ----
def retrieve(question, doc_id, top_k=5):
    question_vector = model.encode(question)

    # Ask Chroma just for the embeddings + text belonging to THIS document
    # (no similarity ranking from Chroma here — we're only using it as storage)
    stored = collection.get(where={"doc_id": doc_id}, include=["embeddings", "documents"])
    chunk_vectors = stored["embeddings"]
    chunks = stored["documents"]

    # Now do the ranking ourselves, exactly like your original each_chunk_vector function
    similarities = []
    for index, each_vector in enumerate(chunk_vectors):
        similarity = cosine_similarity(each_vector, question_vector)
        similarities.append((index, similarity))

    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Take the top_k highest-scoring chunks and join them into one context block
    retrieved_chunks = [chunks[i] for i, score in sorted_similarities[:top_k]]
    context = "\n\n".join(retrieved_chunks)
    return context


# ---- Generation step (the only part that touches your API key) ----
def get_answer(context, question):
    API_KEY = os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
    prompt = f"""
    Use the following notes to answer the question, answer concisely, in plain text.
    {context}
    Question: {question}
    """
    message = [{'role': 'user', 'content': prompt}]
    response = client.chat.completions.create(messages=message, model="openai/gpt-oss-20b:free")
    return response.choices[0].message.content


# ---- UI ----
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")  # <-- this IS your upload button

if uploaded_file is not None:
    doc_id = get_file_hash(uploaded_file)
    text = load_pdf_file(uploaded_file)
    chunks = chunking(text)
    index_document(doc_id, chunks)

    question = st.chat_input("Ask a question")
    if question:
        context = retrieve(question, doc_id)
        answer = get_answer(context, question)
        st.session_state.history.append({"question": question, "answer": answer})

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])