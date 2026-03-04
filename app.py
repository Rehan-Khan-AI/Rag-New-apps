
import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PDF AI Explorer", page_icon="📄", layout="wide")

st.title("📄 Smart PDF Assistant")
st.markdown("Upload a PDF and chat with its content using local AI.")

# ---------------- LOAD MODELS (CACHE) ----------------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    model_id = "google/flan-t5-small"
    pipe = pipeline(
        "text2text-generation",
        model=model_id,
        max_new_tokens=256
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    return embeddings, llm

embeddings_model, llm_model = load_models()

# ---------------- SESSION STATE ----------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📂 Upload Document")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file and st.button("Process & Index PDF"):
        with st.spinner("Processing document..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)

            st.session_state.vectorstore = FAISS.from_documents(
                splits,
                embeddings_model
            )

            os.remove(tmp_path)

            st.success("✅ PDF Indexed Successfully!")

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# ---------------- CHAT INTERFACE ----------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about the PDF..."):

    if not st.session_state.vectorstore:
        st.error("⚠️ Please upload and process a PDF first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n".join([doc.page_content for doc in docs])

                final_prompt = f"""
Answer the question strictly based on the context below.

Context:
{context}

Question:
{prompt}

Answer:
"""

                response = llm_model.invoke(final_prompt)

                st.markdown(response)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": response}
                )
