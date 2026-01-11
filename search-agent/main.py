from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over Internet
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOllama(temperature=0, model="functiongemma")
#tools = [TavilySearch()]
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    
    print("Hello from search-agent!")
    result = agent.invoke({"messages": HumanMessage(content="3 famoust texts in Sumerian")})
    print(result)

if __name__ == "__main__":
    main()
