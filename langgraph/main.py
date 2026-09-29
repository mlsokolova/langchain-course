from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST =-1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END: END,
    ACT: ACT})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="./langgraph/flow.png")

if __name__ == "__main__":
    print("Hello, ReAct Langraph Function Calling")
    start_point = "Haifa"
    end_point = "Deir Al Mukhraqa"
    content = f"""What is the distance in kilometers 
    and the elevation gain in meters between {start_point}  and {end_point}
    using walking route?
    List it and calculate the estimated time in hours to walk the distance."""
    res = app.invoke({"messages": [HumanMessage(content=content)]})    #print(os.getenv("OPENAI_API_KEY"))
    print(res["messages"][LAST].content)