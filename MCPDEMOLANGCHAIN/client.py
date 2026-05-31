from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

import asyncio


async def main():
    client = MultiServerMCPClient(
        {"math": {"command": "python", "args": ["mathserver.py"], "transport": "stdio"}}
    )

    import os

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    tools = await client.get_tools()
    model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    agent = create_agent(model=model, tools=tools)

    math_response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is 2 + 3 and 4 * 5?"}]
    })

    print("Math response:", math_response['messages'][-1].content)

asyncio.run(main())
