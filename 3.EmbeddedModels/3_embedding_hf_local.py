from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model = "sentence-transformers/all-MiniLM-L6-v2")

documents =[
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "Bangalore is the IT hub of India.",
]

vectors = embeddings.embed_documents(documents)
for i, vector in enumerate(vectors):
    print(f"Document {i + 1}: {str(vector)}")