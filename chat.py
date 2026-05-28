from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

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
    user_query = input("Ask something\t")

    search_results = vectordb.similarity_search(query=user_query, k=5)

    if not search_results:
        print("No releant documents found for your query")
        return

    context = "\n\n\n".join(
        [
            f"Page Content: {r.page_content}\n"
            f"Page Number: {r.metadata.get('page_label', 'N/A')}\n"
            f"File: {r.metadata.get('source', 'Unknown')}"
            for r in search_results
        ]
    )

    SYSTEM_PROMPT = """
    You are a helpful AI Assistant who answers user queries based on the context below.
    Only answer from the context and guide the user to the right page number.

    Context: {context}
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite", temperature=0.2, max_retries=3
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"input": user_query, "context": context})
    print(response.content)


if __name__ == "__main__":
    main()
