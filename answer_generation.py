from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

faiss_path ="db/faiss_index"

embedding_model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}
)

db= FAISS.load_local(
    faiss_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

#user query
query="How much did Microsoft pay to acquire GitHub?"
retriever= db.as_retriever(search_kwargs={"k":5})
relevant_docs=retriever.invoke(query)

print(f"\nUser Query: {query}")

print("\nContext")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}")
    print(doc.page_content)
    print("-" * 50)

#combine context+query
context="\n".join([doc.page_content for doc in relevant_docs])
prompt=f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

If the answer is not present in the context, say you don't know.
"""

#using openai
model = ChatGroq(model="llama-3.1-8b-instant")
response=model.invoke(prompt)

print("\nGenerated answer")
print(response.content)


