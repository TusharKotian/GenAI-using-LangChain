from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about MS Dhoni"

doc_embeddings= embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarity_scores=cosine_similarity(
    np.array([query_embedding]), 
    np.array(doc_embeddings)
)
index, score =(sorted(list(enumerate(similarity_scores[0])), key = lambda x:x[1])[-1])

print(documents[index])
print("Similarity Score is: ",score)