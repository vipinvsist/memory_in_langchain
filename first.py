import os
import getpass
from dotenv import load_dotenv
load_dotenv()
if "PPLX_API_KEY" not in os.environ:
    os.environ['PPLX_API_KEY']=getpass.getpass("Enter your PPLX API:\n ")


from langchain_core.prompts import ChatPromptTemplate
from langchain_perplexity import ChatPerplexity

chat = ChatPerplexity(temperature=0, model="sonar-pro")
system = "You are a helpful assistant."
human = "{input}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
response = chain.invoke({"input": "Why is the Higgs Boson important?"})
print(response.content)
""