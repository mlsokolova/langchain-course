from dotenv import load_dotenv

load_dotenv(".env")

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4")

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for 3 web page links of 3 different private guides that make guided hiking trips in Judean desert",
                }
            ]
        }
    )
    print("Hello from react-search-agent!")
    print(result.get("structured_response"))


if __name__ == "__main__":
    main()
