import streamlit as st
import os
import shutil
import requests
from streamlit_lottie import st_lottie
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint  # CHANGED FROM OLLAMA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="RetriPDF", page_icon="📄", layout="wide")

# --- LOAD SECRET ---
# This grabs the key you saved in Step 3
hf_token = os.getenv("HF_TOKEN")

# --- CUSTOM CSS (Same as before) ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    section[data-testid="stSidebar"] { background-color: #262730; }
    .stChatMessage { background-color: #1e1e1e; border-radius: 10px; border: 1px solid #333; }
    h1 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# --- SETUP DIRECTORIES ---
# Hugging Face Spaces are Linux containers. We use /tmp for temporary storage.
WORKING_DIR = "/tmp" 
UPLOAD_DIR = os.path.join(WORKING_DIR, "temp_pdf")
DB_PATH = os.path.join(WORKING_DIR, "vectorstore")

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- HELPER FUNCTIONS ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        return r.json() if r.status_code == 200 else None
    except: return None

def save_uploaded_file(uploaded_file):
    # Clear directory
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def process_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
    return vectorstore.as_retriever()

def get_rag_chain(retriever):
    # CHANGED: Using Hugging Face API instead of Ollama
    # Mistral is a great free model comparable to Llama 3
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    
    llm = HuggingFaceEndpoint(
        repo_id=repo_id, 
        temperature=0.3, 
        huggingfacehub_api_token=hf_token
    )
    
    template = """Answer based on context. If unknown, say so.
    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])), 
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- UI LOGIC (Simplified for brevity) ---
if "messages" not in st.session_state: st.session_state.messages = []
if "rag_chain" not in st.session_state: st.session_state.rag_chain = None

with st.sidebar:
    st.title("RetriPDF Cloud ☁️")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Process"):
        with st.spinner("Processing..."):
            path = save_uploaded_file(uploaded_file)
            retriever = process_document(path)
            st.session_state.rag_chain = get_rag_chain(retriever)
            st.success("Ready!")

st.title("RetriPDF")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    if st.session_state.rag_chain:
        with st.chat_message("assistant"):
            response = st.session_state.rag_chain.invoke(prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})