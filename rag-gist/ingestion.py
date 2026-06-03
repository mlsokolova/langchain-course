import os
from dotenv import load_dotenv
#from langchain_community.document_loaders import TextLoader
#from langchain_community.document_loaders import PDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv("../.env")
load_dotenv()
print(os.getenv("PINECONE_API_KEY"))
print(os.getenv("OPENAI_API_KEY"))


if __name__ == '__main__':
    loader = PDFPlumberLoader("/home/masha/projects/langchain//akkadian/mad2.pdf")
    document = loader.load()
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunks = text_splitter.split_documents(document)

    print(f"created {len(chunks)} chunks")
    print("embedding...")
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    print("ingesting...")
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME"))
    print("done")

