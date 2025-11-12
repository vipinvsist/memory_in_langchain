from xxlimited import Str
import streamlit as st
from dotenv import load_dotenv
import openai 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os

from sympy import use
load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAINN_API_KEY']=os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_PROJECT']='QA_Chatbot'

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')


prompt = ChatPromptTemplate.from_messages([

    ("system",'you are an helpful AI assistant. Please answer the users questions as best as you can!'),
    ("user","Question: {question}")])


def generate_response(question,llm,temperature, max_tokens):
    llm = ChatOpenAI(model_name=llm, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain=prompt|llm|output_parser
    response = chain.invoke({"question": question})
    return response

#Title of the App
st.title("QA Chatbot with Langchain and OpenAI")


# Sidebar for the settings
st.sidebar.title("Settings")
# api_key = st.sidebar.text_input("OpenAI API KEY", type="password")

#dropdown to select OpenAI model
llm = st.sidebar.selectbox("Select the OpenAI Model",["gpt-4o-mini","gpt-4o", "gpt-4.1-nano"])
temperature = st.sidebar.slider("Temperature",min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

##---- Main Interface ----##
st.write("Ask Anything to the AI Chatbot!")
user_input = st.text_input("Here:")

if user_input:
    response = generate_response(user_input,llm,temperature, max_tokens)

    st.write("Response:", response)

else: 
    st.write("Please enter a question to get a response.")