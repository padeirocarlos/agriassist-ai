
import os
import glob

from dotenv import load_dotenv
from huggingface_hub import login
from langchain_chroma import Chroma
from mcp.server.fastmcp import FastMCP

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

load_dotenv(override=True)
# hf_token = os.getenv("HUGGING_FACE_API_TOKEN")
# login(hf_token, add_to_git_credential=True)

# Create FastMCP server
mcp = FastMCP(name="Retrievl_Server")

@mcp.resource("config://app-version")
def get_app_version() -> dict:
    """Static resource providing application version information"""
    return {
        "name": "Retrievel Server",
        "version": "1.0.0",
        "release_date": "2025-11-07",
        "environment": "PRODUCTION"
    }

@mcp.tool()
def retrieve_context(query: str, num_chunks:int=90,  k=2) -> str:
    """Retrieve information from vector documents to help answer a query.
        Args:
            query: The query.
            num_chunks: number of chunks.
            k: number of chunks.
        return:
            serialized: serialization
            retrieved_docs: retrieved documents
    """
    vectorstore = vector_store(num_chunks=num_chunks)
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

def documents_loader(path:str="data/crop_disease") -> list:
    """Read in documents using LangChain's loaders
       Take everything in all the sub-folders of our data"""

    folders = glob.glob(path)
    text_loader_kwargs = {'encoding': 'ISO-8859-1'} # for PDF
    documents = []
    
    for folder in folders:
        loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
        folder_docs = loader.load()
        for doc in folder_docs:
            source_name = str(str(doc.metadata["source"]).split("/")[-1]).split(".")[0]
            doc.metadata["doc_type"] = source_name
            documents.append(doc)
    return documents

def vector_store(path:str="data/crop_disease", num_chunks:int=183):
    # graph_db_name = "graph_db/graph_db.db"
    graph_db_name = "graph_db"
    documents = documents_loader(path)
       
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Check if a Chroma Datastore already exists - if so, delete the collection to start from scratch
    if os.path.exists(graph_db_name):
        Chroma(persist_directory=graph_db_name, embedding_function=embeddings).delete_collection()
        
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=graph_db_name)
    # the retriever is an abstraction over the VectorStore that will be used during RAG; k is how many chunks to use
    return vectorstore.as_retriever(search_kwargs={"k": num_chunks})

if __name__ == "__main__":
    # login(hf_token, add_to_git_credential=True)
    mcp.run(transport='stdio')