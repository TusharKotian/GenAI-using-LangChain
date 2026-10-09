from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(model="gemini-3.8-flash")

result = chat_model.invoke("What is the capital of France?")
print(result.content)