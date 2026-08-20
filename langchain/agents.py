from langchain.agents import create_agent, AgentState
from langchain.tools import tool
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import AIMessage, HumanMessage
import asyncio
import os

load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("API key loaded:", bool(GROQ_API_KEY))


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Result for: {query}"


class MyState(AgentState):
    user_id: str
    call_count: int



config = {"configurable": {"thread_id": str(uuid7())}}


agents = create_agent(
    model="groq:openai/gpt-oss-120b",
    tools=[search],
    system_prompt="You are a helpful assistant. Be concise and accurate.",
    state_schema=MyState,
    checkpointer=InMemorySaver(),
)

result = agents.invoke(
    {
        "messages": [
            {"role": "user", "content": "What's the weather in san Francisco?"}
        ],
        "user_id": "user_123",
        "call_count": 1,
    },
    config=config,
)

# print("Response received.")
# print(result["structured_response"])
# print(result["messages"][-1].content)

async def main():
    stream = agents.astream_events(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Search for AI news and summarize the findings."
                }
            ],
            "user_id": "user_123",
            "call_count": 1,
        },
        config=config,
        version="v2",
    )

    async for event in stream:
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            chunk = event["data"]["chunk"]

            if chunk.content:
                print(chunk.content, end="", flush=True)

        elif event_type == "on_tool_start":
            print(f"\nCalling tool: {event['name']}")

        elif event_type == "on_tool_end":
            print(f"Tool finished: {event['name']}")

    print()


asyncio.run(main())
