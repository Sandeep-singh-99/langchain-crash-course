from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

import os


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("API key loaded:", bool(GROQ_API_KEY))


# --------------------------------------------------
# MongoDB
# --------------------------------------------------

DB_URI = "mongodb://localhost:27017"


# --------------------------------------------------
# Tool
# --------------------------------------------------

def get_user_info() -> str:
    """Look up information about the current user."""
    return "No user profile on file."


# --------------------------------------------------
# Custom Agent State
# --------------------------------------------------

class CustomAgentState(AgentState):
    user_id: str
    preferences: dict


# --------------------------------------------------
# Message trimming middleware
# --------------------------------------------------

@before_model
def trim_messages(
    state: AgentState,
    runtime: Runtime,
) -> dict[str, Any] | None:
    """
    Keep only the first message and the most recent messages
    to control the context window.
    """

    messages = state["messages"]

    # Nothing to trim
    if len(messages) <= 3:
        return None

    first_msg = messages[0]

    # Keep the latest messages.
    # Make sure we don't leave an incomplete tool-call sequence.
    recent_messages = (
        messages[-3:]
        if len(messages) % 2 == 0
        else messages[-4:]
    )

    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages,
        ]
    }


# --------------------------------------------------
# MongoDB Checkpointer
# --------------------------------------------------

with MongoDBSaver.from_conn_string(
    DB_URI,
    db_name="langchain-practice",
    collection_name="checkpoints",
) as checkpointer:

    # --------------------------------------------------
    # Create Agent
    # --------------------------------------------------

    agent = create_agent(
        model="groq:openai/gpt-oss-120b",
        tools=[get_user_info],
        middleware=[trim_messages],
        checkpointer=checkpointer,
        state_schema=CustomAgentState,
    )

    # --------------------------------------------------
    # Thread configuration
    # --------------------------------------------------

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "user-123-thread-1"
        }
    }

    # --------------------------------------------------
    # First message
    # --------------------------------------------------

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Hi, my name is Bob."
                }
            ],
            "user_id": "user_123",
            "preferences": {
                "theme": "dark"
            },
        },
        config,
    )

    print("\nAI:", response["messages"][-1].content)

    # --------------------------------------------------
    # Second message
    # --------------------------------------------------

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Write a short poem about cats."
                }
            ]
        },
        config,
    )

    print("\nAI:", response["messages"][-1].content)

    # --------------------------------------------------
    # Third message
    # --------------------------------------------------

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Now do the same but for dogs."
                }
            ]
        },
        config,
    )

    print("\nAI:", response["messages"][-1].content)

    # --------------------------------------------------
    # Fourth message
    # --------------------------------------------------

    final_response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "What's my name?"
                }
            ]
        },
        config,
    )

    final_response["messages"][-1].pretty_print()