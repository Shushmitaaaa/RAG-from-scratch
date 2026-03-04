from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

faiss_path="db/faiss_index"

embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

db= FAISS.load_local(
    faiss_path, #where to load from
    embeddings=embedding_model, #which em to use
    allow_dangerous_deserialization=True
)

query="How much did Microsoft pay to acquire GitHub?"

retriever= db.as_retriever(search_kwargs={"k":5})

relevant_docs=retriever.invoke(query)

print(f"\nUser Query: {query}")
print("\nContext")

for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}")
    print(doc.page_content)
    print("-" * 50)