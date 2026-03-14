from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
from collections import defaultdict

load_dotenv()

persistent_directory = "db/faiss_index"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

db = FAISS.load_local(
    persistent_directory,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

class QueryVariations(BaseModel):
    queries: List[str]


#multi query retrieval
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

#apply rrf
def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):

    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
    
    rrf_scores = defaultdict(float) #stores rrf score for each chunk
    all_unique_chunks = {}
    
    chunk_id_map = {}
    chunk_counter = 1
    
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        for position, chunk in enumerate(chunks, 1):
            chunk_content = chunk.page_content
            
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            all_unique_chunks[chunk_content] = chunk
            
            position_score = 1 / (k + position)
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    if verbose:
        print(f"RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks

fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60, verbose=True)

print("FINAL RRF RANKING")


print(f"\nTop {min(10, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:10], 1):
    print(f"RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    

print(f"\nRRF Complete! Fused {len(fused_results)} unique documents from {len(query_variations)} query variations.")
print("\nKey benefits:")
print("   Documents appearing in multiple queries get boosted scores")
print("   Higher positions contribute more to the final score")
print("   Balanced fusion using k=60 for gentle position penalties")
