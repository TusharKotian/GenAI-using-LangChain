from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model="gpt-5.5", temperature=0.7)

result=chat_model.invoke("What is the capital of France?")

print(result.content)