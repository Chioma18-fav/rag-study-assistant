import pdfplumber
import math
from sentence_transformers import SentenceTransformer
import os
import chromadb
import hashlib
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

st.title("My Study Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = os.urandom(8).hex()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection("my_documents")

collection = load_chroma()

@st.cache_data
def load_pdf_file(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    return text

def get_file_hash(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    return hashlib.md5(file_bytes).hexdigest()


@st.cache_data
def chunking(text):
    list_chunk = []
    start = 0
    while start < len(text):
        chunks = text[start:start + 1200]
        start+=1000
        list_chunk.append(chunks)
    return list_chunk

@st.cache_data
def chunk_embedding(chunks):
    chunk_embeddings = model.encode(chunks)
    return chunk_embeddings

def question_embedding(question):
    question_embeddings = model.encode(question)
    return question_embeddings

def is_already_indexed(doc_id):
    existing = collection.get(where={"doc_id": doc_id})
    return len(existing["ids"]) > 0

def store_chunks(doc_id, chunks, chunk_vectors, full_text):
    metadatas = []
    for i in range(len(chunks)):
        meta = {"doc_id": doc_id}
        if i == 0:
            meta["full_text"] = full_text
        metadatas.append(meta)

    collection.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        embeddings=chunk_vectors.tolist(),
        documents=chunks,
        metadatas=metadatas,
    )

def load_stored_chunks(doc_id):
    stored = collection.get(where={"doc_id": doc_id}, include=["embeddings", "documents"])
    return stored["embeddings"], stored["documents"]

def load_full_text(doc_id):
    stored = collection.get(
        where={"doc_id": doc_id},
        include=["metadatas"],
        ids=[f"{doc_id}_0"],
    )
    return stored["metadatas"][0]["full_text"]



def each_chunk_vector(chunk_vect,question_vector):
    similarities = []

    for index, each_vector in enumerate(chunk_vect):
        similarity = cosine_similarity(each_vector, question_vector)
        similarities.append((index, similarity))

    sorted_similarities = sorted(
        similarities,
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_similarities

def magnitude_chunk_vector(each_chunk):
    sum_squared_values = 0
    for each_value in each_chunk:
        squared_value = pow(each_value, 2)
        sum_squared_values  += squared_value
    magnitude = math.sqrt(sum_squared_values)
    return magnitude

def magnitude_ques_vector(que_vect):
    sum_squared_values = 0
    for each_value in que_vect:
        squared_value = pow(each_value, 2)
        sum_squared_values  += squared_value
    magnitude = math.sqrt(sum_squared_values)
    return magnitude
 
def cosine_similarity(chunk_vector, quest_vector):
    dot_product = 0
    for doc, ques in zip(chunk_vector, quest_vector):
        product = doc * ques
        dot_product += product
    cos_similarity = dot_product / (magnitude_chunk_vector(chunk_vector) * magnitude_ques_vector(quest_vector))
    return cos_similarity

def Retrieval(each_chunk_retrieval, chunk):
    retrieved_chunks = []

    for index, score in each_chunk_retrieval[:5]:
        retrieved_chunks.append(chunk[index])
    context = "\n\n".join(retrieved_chunks)
    return context

def is_summary_type_question(question):
    keywords = [
        "summarize", "summary", "overview", "everything",
        "all of", "what's missing", "whats missing",
        "list all", "entire document", "whole document", "whole note"
    ]
    question_lower = question.lower()
    return any(word in question_lower for word in keywords)

def get_answer(context, question):
    API_KEY = os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(api_key= API_KEY, base_url="https://openrouter.ai/api/v1")
    prompt = f""" 
    Use the following notes to answer the question. Do not use any outside knowledge.
    If the notes do not contain enough information to answer, say so clearly instead of guessing
   {context}
   Question: {question}
   """
    message = [
    {'role': 'user', 'content': prompt}]

    response = client.chat.completions.create(
    messages=message, 
    model="openai/gpt-oss-20b:free", 
    )

    return response.choices[0].message.content


uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    doc_id = get_file_hash(uploaded_file)

    if not is_already_indexed(doc_id):
        text = load_pdf_file(uploaded_file)
        chunks = chunking(text)
        chunk_vectors = chunk_embedding(chunks)
        store_chunks(doc_id, chunks, chunk_vectors, text)

    question = st.chat_input("Ask a question")

    if question:
        if is_summary_type_question(question):
            context = load_full_text(doc_id)
        else:
            question_vector = question_embedding(question)
            stored_vectors, stored_chunks = load_stored_chunks(doc_id)
            ranked_chunks = each_chunk_vector(stored_vectors, question_vector)
            context = Retrieval(ranked_chunks, stored_chunks)

        answer = get_answer(context, question)
        st.session_state.history.append({"question": question, "answer": answer})

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])