from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

load_dotenv()

# ---------------- BASIC MODEL (FAST) ---------------- #
basic_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_output_tokens=1024,
)

# ---------------- ADVANCED MODEL (HF) ---------------- #
hf_llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    task="text-generation",
    max_new_tokens=1024,
    temperature=0.7,
)

advanced_model = ChatHuggingFace(llm=hf_llm)

# ---------------- TOOL ---------------- #
def get_weather(city: str) -> str:
    return f"It's always sunny in {city}!"

# ---------------- DYNAMIC MIDDLEWARE ---------------- #
@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    messages = request.state.get("messages", [])

    if len(messages) > 10:
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))


# ---------------- AGENT ---------------- #
agent = create_agent(
    model=basic_model,
    tools=[get_weather],
    middleware=[dynamic_model_selection],
    system_prompt="You are a helpful assistant.",
)

# ---------------- RUN ---------------- #
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(response)
