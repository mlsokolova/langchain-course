from typing import List

# import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv("../.env")
# print(os.environ["TAVILY_API_KEY"])

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
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

    url: str = Field(description="url")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources to generate the answer"
    )


llm = ChatOllama(temperature=0, model="gpt-oss")
# tools = [TavilySearch()]
tools = [search]
agent = create_agent(model=llm, tools=tools)  # , response_format=AgentResponse)


def main():

    print("Hello from search-agent!")
    result = agent.invoke(
        {"messages": HumanMessage(content="3 famoust texts in Sumerian")}
    )
    print("Result:")
    print(result)
    # print("Source 0 URL")
    # print(result.messages.keys())


if __name__ == "__main__":
    main()
