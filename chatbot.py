from langchain_perplexity import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os   
load_dotenv()
if "PPLX_API_KEY" not in os.environ:
    raise ValueError("PPLX_API_KEY not found in environment variables. Please set it before running the script.")

else:
    model=ChatPerplexity(name="sonar")
    repsonse = model.invoke([HumanMessage(content="Hi, I'm Vipin and I am a Data Scientist")])

    print(repsonse.content)
