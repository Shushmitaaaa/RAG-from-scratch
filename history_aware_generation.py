from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


load_dotenv()

#loading embedddings
embedding_model=HuggingFaceEmbeddings(
    model_name= "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}
)

#loading FAISS database
db=FAISS.load_local(
    "db/faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

model= ChatGroq(model="llama-3.1-8b-instant")

#store conversation history
chat_history= []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    if chat_history:
        messages = [
            SystemMessage(
                content="Rewrite the user's question to be standalone using the conversation history. Return only the rewritten question."
            )
        ] + chat_history + [
            HumanMessage(content=user_question)
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"Searching for: {search_question}")

    else:
        search_question = user_question

    #retrieving relevant documents
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10}
    )

    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")

    for i, doc in enumerate(docs, 1):
        preview = doc.page_content[:120]
        print(f"Doc {i}: {preview}...")

    #build some contenxt
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {user_question}

    If the answer is not present in the context, say you don't know.
    """

    #generate answer
    messages = [
        SystemMessage(content="You answer questions using only the provided documents."),
        HumanMessage(content=prompt)
    ]

    result = model.invoke(messages)
    answer = result.content


    #save convoo
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"\nAnswer: {answer}")
    return answer

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()