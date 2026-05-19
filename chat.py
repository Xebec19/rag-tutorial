from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},  # explicitly use CPU
        encode_kwargs={
            "normalize_embeddings": True
        },  # recommended for cosine similarity
    )

    vectordb = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        embedding=embeddings,
        collection_name="learning_rag",
    )

    # Take user input
    user_query = input("Ask something")

    search_results = vectordb.similarity_search(query=user_query)

    context = "\n\n\n".join(
        [
            f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
            for result in search_results
        ]
    )

    SYSTEM_PROMPT = f"""
    You are a helpful AI Assistant who answers user query based on the available context
    retrieved from  a PDF file along with page_contents and page number.
    You should only answer the user based on the following context and navigate the user to
    open the right page number to know more.

    Context: {context}
    """


if __name__ == "__main__":
    main()
