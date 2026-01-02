from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
import operator
from typing import TypedDict, Annotated

# Load environment variables
load_dotenv()

# Define the state for the graph
class State(TypedDict):
    messages: Annotated[list, operator.add]

# Define a simple tool
@tool
def search(query: str):
    """Searches for information."""
    # This is a dummy tool, returning a fixed string
    print(f"---TOOL CALLED: search(query='{query}')---")
    return "The weather in SF is 50 degrees"

# Initialize the model and tools
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
tools = [search]
model_with_tools = llm.bind_tools(tools)

# Define the graph nodes
def chatbot(state: State):
    """Chatbot node to generate a response."""
    print("---CHATBOT---")
    return {"messages": [model_with_tools.invoke(state["messages"])]}

def tool_node(state: State):
    """Tool node to execute a tool call."""
    print("---TOOL NODE---")
    tool_calls = state["messages"][-1].tool_calls
    tool_outputs = []
    for tool_call in tool_calls:
        tool_outputs.append(
            ToolMessage(
                tool_call_id=tool_call["id"],
                content=tools[0].invoke(tool_call["args"]),
            )
        )
    return {"messages": tool_outputs}

# Define the graph
workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("chatbot")
workflow.add_conditional_edges(
    "chatbot",
    tools_condition,
    {"tools": "tools", END: END},
)
workflow.add_edge("tools", "chatbot")

# Set up memory and the graph
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, interrupt_before=["tools"])

# Main loop
thread_id = {"configurable": {"thread_id": "1"}}
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        break
    if not user_input.strip():
        continue

    events = graph.stream({"messages": [HumanMessage(content=user_input)]}, thread_id)
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    snapshot = graph.get_state(thread_id)
    if snapshot.next:
        print("\nThe model wants to use a tool. Do you approve? (yes/no)")
        approval = input("> ")
        if approval.lower() == "yes":
            events = graph.stream(None, thread_id)
            for event in events:
                if "messages" in event:
                    event["messages"][-1].pretty_print()
        else:
            print("Tool usage denied. Ending conversation.")
            # To properly end, we can just stop or send a message back
            # For simplicity, we'll just break the loop here
            break

