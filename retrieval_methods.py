from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Setup
persistent_directory = "db/faiss_index"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

db = FAISS.load_local(
    persistent_directory,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

query = "How much did Microsoft pay to acquire GitHub?"
print(f"Query: {query}\n")

#method 1 basic similarity search->returns top k most similar elements
#remember faiss db here also acts as a search tool
retriever=db.as_retriever(search_kwargs={"k":3})
docs=retriever.invoke(query)
print(f"Retrieved {len(docs)} documents:\n")
for i, doc in enumerate(docs,1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")

#method 2 Similarity w score threshold->returns all elements with similarity score above a certain threshold
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.3  # Only return docs with similarity >= 0.3
    }
)
docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents (threshold: 0.3):\n")
for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")


#method 3 mmr-> includes relevance + diversity by selecting documents that are not only similar to the query but also diverse from each other

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           # Final number of docs
        "fetch_k": 10,    # Initial pool to select from
        "lambda_mult": 0.5  # 0=max diversity, 1=max relevance therefore 0.5(balanced)
    }
)
docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents (λ=0.5):\n")
for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")
print("Done! Try different queries or parameters to see the differences.")