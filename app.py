import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# App UI
st.set_page_config(page_title="Kazakhstan Constitution AI (Ollama)", layout="wide")
st.title("🇰🇿 Constitution AI Assistant (Powered by Ollama)")

# Initialize Ollama LLM and vector store
llm = ChatOllama(model="llama3")
embedding = OllamaEmbeddings()
persist_dir = "chroma_db"

@st.cache_resource
def init_vectorstore():
    return Chroma(embedding_function=embedding, persist_directory=persist_dir)

vectorstore = init_vectorstore()

# Upload and process PDFs
uploaded_files = st.file_uploader("📄 Upload Constitution or Related PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(file.name)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        vectorstore.add_documents(docs)
        os.remove(file.name)
    st.success("✅ Documents uploaded and indexed.")

# Chat interface
query = st.text_input("💬 Ask a question about the Constitution or uploaded documents:")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if query:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    result = qa.run(query)
    st.session_state.chat_history.append((query, result))

# Show chat
if st.session_state.chat_history:
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")
