from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_output_tokens=1024)

agent = create_agent(model, agent_type="zero-shot-react-description")