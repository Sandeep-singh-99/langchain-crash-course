from fastapi import FastAPI
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from mcp.tools import calculator, get_notes

app = FastAPI()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key="YOUR_API_KEY",
    temperature=0
)

tools = [calculator, get_notes]

prompt = PromptTemplate.from_template("""
You are a helpful assistant.

Question: {input}
{agent_scratchpad}
""")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(req: ChatRequest):
    result = agent_executor.invoke(
        {"input": req.message}
    )
    return {"response": result["output"]}
