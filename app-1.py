
import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --- PAGE SETUP ---
st.set_page_config(page_title="PDF AI", page_icon="🤖", layout="wide")
st.title("📄 AI PDF Assistant")

# --- 1. DOWNLOAD SYSTEM (For your convenience) ---
def get_script_content():
    with open(__file__, "r") as f:
        return f.read()

# --- 2. ASSET LOADING (Cached) ---
@st.cache_resource
def load_rag_assets():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    
    pipe = pipeline(
        task="text2text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=256
    )
    return embeddings, HuggingFacePipeline(pipeline=pipe)

embeddings_model, llm_model = load_rag_assets()

# --- 3. SESSION STATE ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Admin & Upload")
    
    # Download button so you can easily save this file
    st.download_button(
        label="📥 Download this app.py file",
        data=get_script_content(),
        file_name="app.py",
        mime="text/x-python"
    )
    
    st.divider()
    
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Index Document"):
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splits = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100).split_documents(docs)
            st.session_state.vectorstore = FAISS.from_documents(splits, embeddings_model)
            os.remove(tmp_path)
            st.success("Ready!")

# --- 5. CHAT ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your PDF..."):
    if not st.session_state.vectorstore:
        st.error("Upload a PDF first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
            context = "\n".join([d.page_content for d in docs])
            response = llm_model.invoke(f"Context: {context}\n\nQuestion: {prompt}\nAnswer:")
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
