import asyncio
import os
import ssl
from typing import Any, Dict, List
import certifi
from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_info, log_error, log_warning, log_success, log_header)
load_dotenv()

#Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_min_seconds=10)
#vectorstore = PineconeVectorStore(index_name="langchain-doc", embedding=embeddings)
tavily_extract = TavilyExtract(extract_depth="advanced")
tavily_map = TavilyMap(max_depth=5, 
                       limit=100, 
                       max_breadth=20, 
                       max_pages=1000, 
                       #paths=["/python/.*"], 
                       allow_external=False,
                       instructions="Only http lesson/content pages. Avoid ipynb content",
                       )
tavily_crawl = TavilyCrawl()

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Chunk a list of URLs into smaller lists of a given size"""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs"""
    try:
        log_info(
            f"TavilyExtract: Processing batch {batch_num} of {len(urls)} URLs"
        )
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        log_success(f"TavilyExtract: Completed batch {batch_num} - extracted {len(docs.get('results', []))} documents")
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Error processing batch {batch_num}: {e}")
        return []

async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENTATION EXTRACTION PHASE")
    log_info(
        f"TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN
    )
    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    #Filter out exceptions and fletten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Error processing batch {result}: {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:
                document = Document(
                    page_content=extracted_page["raw_content"], 
                    metadata={"source": extracted_page["url"]}
                )
                all_pages.append(document)
    log_success(f"TavilyExtract: Completed extraction of {len(all_pages)} documents")
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed")
    return all_pages
    #return results

async def async_map(url: str):
    log_header("DOCUMENTATION MAPPING PHASE")
    log_info(f"TavilyMap: Starting to map documentation from {url}", Colors.DARKCYAN)
    site_map = await tavily_map.ainvoke(input={"url": url})

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batche asynchronously"""
    log_header("VECTOR STORAGE INGESTION PHASE")
    log_info(f"VectorStorage: Preparing to add {len(documents)} documents to vector store", Colors.DARKCYAN)
    batches = [
        documents[i:i + batch_size] for i in range(0, len(documents), batch_size)
    ] 
    log_info(f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents", Colors.BLUE)
    
    # add batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            vectorstore = PineconeVectorStore(index_name="akkadian-index", embedding=embeddings)
            await vectorstore.aadd_documents(batch)
            log_success(f"VectorStore Indexing: Successfully indexed batch {batch_num} / {len(batch)} documents")
        except Exception as e:
            log_error(f"VectorStore Indexing: Error indexing batch {batch_num} - {e}")
            return False
        return True
    
    #Process batches concurrently
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # count successful batches
    successful_batches = sum(1 for result in results if result is True)
    if successful_batches == len(batches):
        log_success(f"VectorStore Indexing: All batches indexed successfully {successful_batches} / {len(batches)}")
    else:
        log_warning(f"VectorStore Indexing: Failed to index {len(batches) - successful_batches} / {len(batches)} batches")
    
def crawl_documentation(url: str):
    log_header("DOCUMENTATION CRAWLING PHASE")
    log_info(f"TavilyCrawl: Starting to Crawl documentation from {url}", Colors.PURPLE)
    res = tavily_crawl.invoke({
    "url": url,
    "max_depth": 5,
    "extract_depth": "advanced",
    "max_pages": 1000,
    "limit": 500,
    "allow_external": True,
    "include_images": False, 
    "max_breadth": 20,
    "instructions": "Crawl the eAkkadian site and extract all http lesson/content page content. Avoid ipynb content"
    #"allow_external": False,
    })
    return res

def convert_tavily_crawl_results_to_docs(crawl_res: List[Dict[str, Any]]) -> List[Document]:
    all_docs = []
    for tavily_crawl_result_item in crawl_res["results"]:
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )
    return all_docs

def split_documents(documents: List[Document]) -> List[Document]:
    log_info(
        f"✂️  Text Splitter: Processing {len(documents)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENTATION EXTRACTION PHASE")
    log_info(
        f"TavilyExtract: Starting concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN
    )
    tasks = [extract_batch(batch, i+1) for i, batch in enumerate(url_batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    #Filter out exceptions and fletten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Error processing batch {result}: {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]:
                document = Document(
                    page_content=extracted_page["raw_content"], 
                    metadata={"source": extracted_page["url"]}
                )
                all_pages.append(document)
    log_success(f"TavilyExtract: Completed extraction of {len(all_pages)} documents")
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed")
    return all_pages

async def tavily_map_extract_doc(url: str) -> List[Dict[str, Any]]:
    site_map = tavily_map.invoke(url)
    log_success(f"TavilyMap: Successfully generated site map for {url}")

    #Split URLs into chunks
    url_batches = chunk_urls(list(site_map["results"]), chunk_size=20)
    #print(f"{len(site_map["results"])}")
    log_info( 
    f"URL Processing: Split {type(site_map['results'])}  into {len(url_batches)} batches",
    Colors.BLUE)
    all_docs = await async_extract(url_batches)
    log_success(f"TavilyExtract: Completed extraction of {len(all_docs)} documents")
    return all_docs

async def main():
    #url = "https://reference.langchain.com"
    url = "https://digitalpasts.github.io/eAkkadian/home.html"
    """Main async function to orchestrate the entire process"""
    log_header("DOCUMENTATION INGESTION PIPELINE")
    #crawl_res = crawl_documentation(url)
    # Convert Tavily crawl results to LangChain documents
    #all_docs = convert_tavily_crawl_results_to_docs(crawl_res)
    # Extract documents from URLs
    # Split documents into chunks
    all_docs = await tavily_map_extract_doc(url)
    log_header("DOCUMENT CHUNKING PHASE")
    splitted_docs = split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")



if __name__ == "__main__":
    asyncio.run(main())
