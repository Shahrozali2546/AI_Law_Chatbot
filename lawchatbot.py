import sys
sys.modules['torch.classes'] = None  # 🧯 Fix torch + streamlit issue

import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # ✅ NEW IMPORT - no warning!


load_dotenv()

st.title("⚖️ Law Chatbot")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

@st.cache_resource
def get_vectorstore():
    try:
        pdf_path = "universal_declaration_of_human_rights.pdf"
        loader = PyPDFLoader(pdf_path)
        docs = loader.load_and_split(
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        )

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)

        return vectorstore

    except Exception as e:
        st.error(f"Vectorstore loading error: {e}")
        return None

prompt = st.chat_input("Ask me anything about the document...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    try:
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        if not GROQ_API_KEY:
            st.error("🚫 GROQ_API_KEY is missing in your .env file.")
            st.stop()

        groq_chat = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model="llama3-8b-8192"
        )

        vectorstore = get_vectorstore()
        if not vectorstore:
            st.error("❌ Failed to load vectorstore.")
            st.stop()

        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )

        result = chain.invoke({'query': prompt})
        response = result['result']

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

    except Exception as e:
        st.error(f"💥 Error occurred: {e}")
