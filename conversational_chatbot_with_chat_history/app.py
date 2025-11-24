import streamlit as st
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain_perplexity import ChatPerplexity
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder

import os
from dotenv import load_dotenv
load_dotenv()
try:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    os.environ["PPLX_API_KEY"] = os.getenv("PPLX_API_KEY")
except Exception as e:
    print("Error loading environment variables:", e)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


st.title("Conversational RAG with chat history")
st.write("Upload the PDF document and chat with the content")



if os.environ["PPLX_API_KEY"]:
    model=ChatPerplexity(name='sonar')
    session_id = st.text_input("Session ID (to maintain chat history)", value="default_session")
    # Statefully manage the chat history

    if "store" not in st.session_state:
        st.session_state["store"] = {}# stores the session state


    uploaded_files = st.file_uploader("choose a PDF file", type="pdf",accept_multiple_files=True)
    if uploaded_files:
        documents=[]
        for file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as f:
                f.write(file.getvalue())
                file_name = file.name
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        split = text_splitter.split_documents(documents)
        vectorestore=Chroma.from_documents(split, embeddings)
        retriver = vectorestore.as_retriever(search_kwargs={"k": 3})


        contextualize_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [   
                ("system",contextualize_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human","{input}"),
            ]
        )

        history_aware_retriver = create_history_aware_retriever(
            llm=model,
            retriever=retriver,
            prompt=contextualize_prompt)
        

        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human","{input}"),  
            ]
        )

        qa_chain = create_stuff_documents_chain(
            llm=model,prompt=qa_prompt)
        

        rag_chain = create_retrieval_chain(history_aware_retriver,qa_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input=st.text_input("Your Question:")
        if user_input:
            session_history =get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )

            st.write(st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write("Chat History:",session_history.messages)
    else:
        st.write("Please upload a PDF file to start chatting.")
