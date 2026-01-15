from typing import List
from pydantic import BaseModel, Field
#import os
from dotenv import load_dotenv

load_dotenv("../.env")
#print(os.environ["TAVILY_API_KEY"])

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient


tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that searches over Internet
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

class Source(BaseModel):
    """Schema for a source used by agent"""
    url:str = Field(description="The URL of the source that TavitySearch brings")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources to generate the answer")


llm = ChatOpenAI(temperature=0, model="gpt-5")
#tools = [TavilySearch()]
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():

    print("Hello from search-agent!")
    result = agent.invoke(
        {"messages": HumanMessage(content="3 famoust texts in Assirian")}
    )
    print("Result:")
    print(result)
    


if __name__ == "__main__":
    main()