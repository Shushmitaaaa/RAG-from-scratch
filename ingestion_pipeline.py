import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from dotenv import load_dotenv

# load_dotenv()
#loading docs
def load_documents(docs_path="docs"):
    """Load all text files from the docs directory"""
    print(f"Loading documents from {docs_path}...")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please craete it and add your company files.")
    #loading all the text files frm the directory
    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    document=loader.load()

    if len(document)==0:
        raise FileNotFoundError(f"No .txt file found in the {docs_path} directory.Please add your company documents")
    for i,doc in enumerate(document[:2]):
        print(f"\n Document {i+1}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")
    return document

#chunking
def split_documents(documents,chunk_size=800,chunk_overlap=0):
    """SPlit documnets into smaller chunks with overlap"""
    print("Splitting documents into chunks")

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
    chunks= text_splitter.split_documents(documents)

    if chunks:
        
        for i,chunk in enumerate(chunks[:5]):
            print(f"\n Chunk {i+1}")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length:{len(chunk.page_content)} characters")
            print(f"COntent")
            print(chunk.page_content)
            print("-"*50)

            if len(chunks)>5:
                print(f"\n...and {len(chunks)-5} more chunks")
    return chunks

#vector embeddings+storing in db
def create_vector_store(chunks,save_path="db/faiss_index"):
    """Create and save FAISS vector store"""
    print("Create embeddings and storing in FAISS")

    #creating embedding model
    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
    print("creating vector store")
    vectorstore=FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    os.makedirs(save_path, exist_ok=True)

    # Save FAISS index locally
    vectorstore.save_local(save_path)

    print(f"--- FAISS vector store saved to {save_path} ---")

    return vectorstore
         

def main():
    documents=load_documents(docs_path="docs")
    chunks=split_documents(documents)
    vectorstore = create_vector_store(chunks)

if __name__ == "__main__":
    main()

