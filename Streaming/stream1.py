from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

for chunk in model.stream("What is the capital of France?"):
    print(chunk.text, end="", flush=True)