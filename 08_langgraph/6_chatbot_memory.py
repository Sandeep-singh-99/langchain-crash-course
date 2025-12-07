from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.store.memory import InMemoryStore
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize the store
store = InMemoryStore()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Updated to a standard available model version
    temperature=0.5
)

class ChatState(TypedDict):
    messages: List[str]
    thread_id: str

# -------------------------
# Nodes
# -------------------------

def load_memory(state: ChatState):
    thread_id = state["thread_id"]
    
    # 1. Get the item from the store
    saved_item = store.get(("chat_memory",), thread_id)
    
    # 2. Check if item exists and access the .value property
    if saved_item:
        state["messages"] = saved_item.value.get("messages", [])
    else:
        state["messages"] = []
    
    return state

def user_input(state: ChatState):
    # Note: Using input() inside a node is okay for testing, 
    # but in production, you usually pass input via .invoke()
    text = input("\nYou: ") 
    state["messages"].append(f"You: {text}")
    return state

def bot_response(state: ChatState):
    # Prepare history for the LLM
    history = "\n".join(state["messages"])
    
    response = llm.invoke(history)
    state["messages"].append(f"Bot: {response.content}")
    
    # ✅ FIX: Use store.put instead of store.set
    store.put(
        ("chat_memory",), 
        state["thread_id"], 
        { "messages": state["messages"] }
    )
    
    return state

# -------------------------
# Build Graph
# -------------------------

graph = StateGraph(ChatState)

graph.add_node("load", load_memory)
graph.add_node("user", user_input)
graph.add_node("bot", bot_response)

graph.set_entry_point("load")
graph.add_edge("load", "user")
graph.add_edge("user", "bot")
graph.add_edge("bot", END)

app = graph.compile()

# -------------------------
# Run
# -------------------------

thread_id = "akash-session"

# Initial state
state = {
    "messages": [],
    "thread_id": thread_id
}

print("\n✅ LangGraph Memory Chatbot started. Ctrl+C to stop\n")

while True:
    try:
        # We invoke the app. Because "load_memory" is the entry point,
        # it will fetch the DB memory before asking for user input.
        result = app.invoke(state)
        
        # Update our local state variable with the result for the next loop
        state = result
        
        # Print the last message (the bot's response)
        if state["messages"]:
            print("\n" + state["messages"][-1])
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
        break