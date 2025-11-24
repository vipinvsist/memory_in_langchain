from regex import E
import streamlit as st
from langchain.chains import create_history_aware_retriver, create_retrival_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_perplexity import ChatPerplexity
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv
load_dotenv()
try:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    os.environ["PERPLEXITY_API_KEY"] = os.getenv("PERPLEXITY_API_KEY")
except Exception as e:
    print("Error loading environment variables:", e)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


st.title("Conversational RAG with chat history")
st.write("Upload the PDF document and chat with the content")



if os.environ["PERPLEXITY_API_KEY"]:
    model=ChatPerplexity(name='sonar')
    session_id = st.text_input("Session ID (to maintain chat history)", value="default_session")
    # Statefully manage the chat history

    if "chat_history" not in st.session_state:
        st.session.state.store={}
    
