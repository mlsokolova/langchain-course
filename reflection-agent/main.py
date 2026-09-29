import os
import dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages

dotenv.load_dotenv()

from chains import generate_chain, reflection_chain

os.environ["LANGCHAIN_PROJECT"] = "reflection_agent"

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = "reflect"    
GENERATE = "generate"

def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke(state["messages"])]}

def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke(state["messages"])
    return {"messages": [HumanMessage(content=res.content)]}

builder =StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

builder.add_conditional_edges(GENERATE, should_continue
      , path_map={END:END, REFLECT:REFLECT})
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

if __name__ == "__main__":
    print(f"Hello, {os.getenv('LANGCHAIN_PROJECT')}")
    inputs = HumanMessage(content="""Make this tweet better:
    'blockchain-based BIM models and InterPlanetary File System 
     are the future of ConstructionTech'""")
    

    response = graph.invoke({"messages":[inputs]})