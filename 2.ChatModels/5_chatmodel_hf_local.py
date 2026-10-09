from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_length": 512, "temperature": 0.7}
)

model = ChatHuggingFace(llm=llm)

result=model.invoke("What is the top 5 programming languages in 2026?")
print(result.content)