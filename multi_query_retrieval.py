#query provided by the user will have multiple variations (hence multi_query retrieval) for these queries hybrid retrieval(vector+keyword will be done)--> chunks will be generated using rrf and a reranking model top chunks will be generated then query+chunks will be given to llm

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel #for output by llm to match a specific structure
from typing import List

load_dotenv()

persistent_directory = "db/faiss_index"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0) #when temprature is kept 0 there is no randomness in response the answer which is provided will be accurate

db = FAISS.load_local(
    persistent_directory,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
#this is for structured ouutput by llm to match a specific structure
class QueryVariations(BaseModel):
    queries: List[str]

original_query = "What is the Transformer architecture?"
print(f"Original Query: {original_query}\n")

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:
Original query: {original_query}
Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

print("Generated Query Variations:")
for i, variation in enumerate(query_variations, 1):
    print(f"{i}. {variation}")

#retrieving 5 chunks for each query variation
retriever = db.as_retriever(search_kwargs={"k": 5})
all_retrieval_results = []

for i, query in enumerate(query_variations, 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)
    
    print(f"Retrieved {len(docs)} documents:\n")
    
    for j, doc in enumerate(docs, 1):
        print(f"Document {j}:")
        print(f"{doc.page_content[:150]}...\n")

print("Multi-Query Retrieval Complete!")