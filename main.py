from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def main():

    # read file
    pdf_path = f"{Path(__file__).parent}/assets/node-handbook.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()

    # Split the docs into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)

    chunks = text_splitter.split_documents(documents=docs)

    # Vector embeddings
    # embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # explicitly use CPU
        encode_kwargs={
            "normalize_embeddings": True
        },  # recommended for cosine similarity
    )

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,  # ← plug it in directly
        url="http://localhost:6333",
        collection_name="learning_rag",
    )

    print("Indexing of documents done!")


if __name__ == "__main__":
    main()
