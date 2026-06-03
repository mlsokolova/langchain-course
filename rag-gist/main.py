import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

load_dotenv("../.env")
load_dotenv()

print("Initializing components...")
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()
vectorstore = PineconeVectorStore(index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: 
    {context}
    Question: {question}
    Provide a detailed answer:
    """
)

def format_docs(docs):
    """Format the documents into a string"""
    return "\n".join([doc.page_content for doc in docs])

def retrieval_chain_without_llm(query):
    """Simple retrieval chain without LCEL
       Manually retrieve the documents, format them, and return the result
    """
    docs = retriever.invoke(query)
    return format_docs(docs)

def retrieval_chain_with_llm(query):
    """Retrieval chain with LCEL
       Use the retriever to get the documents, format them, and return the result
       limitations:
       - Manual step-by-step execution
       - No built-in streaming support
       - No async support without additional code
       - Harder to compose with other chains
       - More verbose and error-prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)
    # Step 2: Format the documents
    context = format_docs(docs)
    # Step 3: Create a prompt template
    messages = prompt_template.format_messages(context=context, question=query)
    # Step 4: Invoke the LLM
    response = llm.invoke(messages)
    # Step 5: Return the response
    return response.content

def create_retrieval_chain_with_lcel() :
    """
    Create retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with a query.
    Advantages over non-LCEL approach:
    - Declarative and composable: easy to chain operations with pipe operator (|)
    - Built-in streamig: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.bath()  for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(context=itemgetter("question") | retriever | format_docs ) 
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain

if __name__ == "__main__":
    query = "What is a root in Accadian language? Give 1 examle of the word, word translation to english, the root of this word and root meaning"
    # =============================
    # Option 0: Raw invocation without RAG
    # =============================
    print("\n" + "=" * 70)
    print("Implementation 0: Raw LLM Invocation (no RAG)")
    print("=" * 70 + "\n")
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)
    # =============================
    # Option 1: Implementation without LCEL
    # =============================
    print("\n" + "=" * 70)
    print("Implementation 1: Without LCEL")
    print("=" * 70 + "\n")
    result_without_lcel = retrieval_chain_without_llm(query)
    print("\nAnswer:")
    print(result_without_lcel)
     # =============================
    # Option 2: Implementation with LCEL
    # =============================
    print("\n" + "=" * 70)
    print("Implementation 2: With LCEL")
    print("=" * 70 + "\n")
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)