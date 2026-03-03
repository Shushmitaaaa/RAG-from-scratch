import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

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
    

def main():
    documents=load_documents(docs_path="docs")

if __name__ == "__main__":
    main()

