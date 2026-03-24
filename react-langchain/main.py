from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.prompts import Prompt, PromptTemplate
from langchain_core.tools import render_text_description
from langchain_openai import ChatOpenAI
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from typing import Union
from langchain_classic.agents.agent import AgentAction, AgentFinish
from langchain_classic.tools import Tool
from typing import List

load_dotenv(".env")

@tool()
def get_text_length(text: str) -> int:
    """Get the length of the text by characters"""
    text = text.strip("'\n").strip('"') #strip the text of the return and double quotecharacters
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

if __name__ == "__main__":
    print("Hello from react-langchain!")
    tools = [get_text_length]
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought:
    """
prompt = PromptTemplate.from_template(template=template)\
                       .partial(tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]))

llm = ChatOpenAI(model="gpt-4", temperature=0, stop=["\nObservation:","Observation"])
agent = {"input": lambda x: x["input"]} | prompt | llm | ReActSingleInputOutputParser()
agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the length of the text 'Hello, world!'?"})
print(agent_step)

if isinstance(agent_step, AgentAction):
    tool_name = agent_step.tool
    tool_to_use = find_tool_by_name(tools, tool_name)
    tool_input = agent_step.tool_input

    observation = tool_to_use.func(str(tool_input))
    print(f"{observation=}")