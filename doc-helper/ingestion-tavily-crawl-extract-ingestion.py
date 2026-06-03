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
vectorstore = PineconeVectorStore(index_name="langchain-doc", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=5000, paths=["/python/.*"])
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

async def main():
    #url = "https://python.langchain.com"
    url = "https://reference.langchain.com"
    """Main async function to orchestrate the entire process"""
    log_header("DOCUMENTATION INGESTION PIPELINE")
   
    
    site_map = tavily_map.invoke(url)
    log_success(f"TavilyMap: Successfully generated site map for {url}")

    #Split URLs into chunks
    url_batches = chunk_urls(list(site_map["results"]), chunk_size=20)
    #print(f"{len(site_map["results"])}")
    log_info( 
    f"URL Processing: Split {type(site_map['results'])}  into {len(url_batches)} batches",
    Colors.BLUE)

    # Crawl the documentation site
    #log_info(f"TavilyCrawl: Starting to Crawl documentation from {url}",
    #Colors.PURPLE
    #)
    #res = tavily_crawl.invoke({
    #"url": url,
    #"max_depth": 5,
    #"extract_depth": "advanced",
    #"instructions": "content for AI agents"
    #})
    #all_docs = [Document(page_content=doc["raw_content"], metadata={"source": doc["url"]}) for doc in res["results"]]
    #log_success(f"TavilyCrawl: Successfully crawled {len(all_docs)} pages")
    
    # Extract documents from URLs
    all_docs = await async_extract(url_batches)



if __name__ == "__main__":
    asyncio.run(main())
