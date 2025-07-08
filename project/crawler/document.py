import time
from dataclasses import dataclass, field
import hashlib
from typing import List, Dict, Set, Optional, Any
from datetime import datetime
from urllib.parse import urlparse
import logging
import json
from bs4 import BeautifulSoup


@dataclass
class Document:
    url: str
    title: str = ""
    content_hash: str = field(default="", init=False)
    
    content: str = ""
    meta_description: str = ""
    meta_keywords: str = ""
    h1_tags: List[str] = field(default_factory=list)
    h2_tags: List[str] = field(default_factory=list)
    h3_tags: List[str] = field(default_factory=list)

    crawl_timestamp: float = field(default_factory=time.time)
    crawler_relevance_score: float = 0.0
    keywords_found: List[str] = field(default_factory=list)
    keyword_density: Dict[str, float] = field(default_factory=dict)

    word_count: int = 0
    content_to_html_ratio: float = 0.0
    sentence_count: int = 0
    paragraph_count: int = 0
    has_ssl: bool = False

    incoming_links: Set[str] = field(default_factory=set)
    outbound_links: Set[str] = field(default_factory=set)

    page_size_bytes: int = 0
    load_time_ms: float = 0.0
    status_code: int = 200
    canonical_url: str = ""


    domain: str = field(default="", init=False)
    subdomain: str = field(default="", init=False)
    path_depth: int = field(default=0, init=False)

    language: str = "en"
    country_code: str = ""

    last_modified: Optional[datetime] = None
    last_crawled: datetime = field(default_factory=datetime.now)
    crawl_frequency: int = 0  # number of times crawled



    def __post_init__(self):

        parsed_url = urlparse(self.url)
        url_parts = self.get_url_parts()

        self.domain = url_parts["domain"]
        self.subdomain = url_parts["subdomain"]
        self.path_depth = len([p for p in parsed_url.path.split('/') if p])
        self.canonical_url = self.canonical_url or self.url
        self.has_ssl = parsed_url.scheme == 'https'

        if self.content:
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()


    def update_metrics(self, html: str):
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        self.content = text
        self.word_count = len(self.content.split())
        self.sentence_count = len([s for s in text.split('.') if s.strip()])
        self.paragraph_count = html.count('<p>')


    def is_duplicate(self, other: 'Document') -> bool:
        if not self.content_hash or not other.content_hash:
            return False
        return self.content_hash == other.content_hash
    

    def get_url_parts(self) -> Dict[str, Any]:
        parsed_url = urlparse(self.url)
        return {
            "domain" : parsed_url.hostname or "",
            "subdomain": parsed_url.hostname.split('.')[0] if parsed_url.hostname else "",
        }

    
    def load_from_dict(self, data: Dict):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Recalculate derived fields
        self.__post_init__()
    
    def to_dict(self) -> Dict:
        return {
            "url": self.url,
            "title": self.title,
            "content_hash": self.content_hash,
            "content": self.content,
            "meta_description": self.meta_description,
            "meta_keywords": self.meta_keywords,
            "h1_tags": self.h1_tags,
            "h2_tags": self.h2_tags,
            "h3_tags": self.h3_tags,
            "crawl_timestamp": self.crawl_timestamp,
            "crawler_relevance_score": self.crawler_relevance_score,
            "keywords_found": self.keywords_found,
            "keyword_density": self.keyword_density,
            "word_count": self.word_count,
            "content_to_html_ratio": self.content_to_html_ratio,
            "sentence_count": self.sentence_count,
            "paragraph_count": self.paragraph_count,
            "has_ssl": self.has_ssl,
            "outbound_links": list(self.outbound_links),
            "page_size_bytes": self.page_size_bytes,
            "load_time_ms": self.load_time_ms,
            "status_code": self.status_code,
            "canonical_url": self.canonical_url,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "path_depth": self.path_depth,
            "language": self.language,
            "country_code": self.country_code,
            "last_modified": str(self.last_modified) if self.last_modified else None,
            "last_crawled": str(self.last_crawled),
            "crawl_frequency": self.crawl_frequency
        }


class DocumentCollection:
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.content_hashes: Dict[str, str] = {}
        self.domain_documents: Dict[str, List[str]] = {}


    def add_document(self, doc: Document) -> bool:
        # Check if document content already in collection
        if doc.content_hash in self.content_hashes:
            existing_url = self.content_hashes[doc.content_hash]
            if existing_url != doc.url:
                logging.warning(f"Duplicate content detected: {doc.url} has same content as {existing_url}")
                return False

        # Add to documents and content_hashes dicts
        self.documents[doc.url] = doc
        self.content_hashes[doc.content_hash] = doc.url

        # Update domain documents
        if doc.domain not in self.domain_documents:
            self.domain_documents[doc.domain] = []
        if doc.url not in self.domain_documents[doc.domain]:
            self.domain_documents[doc.domain].append(doc.url)

        return True
    

    def write_collection_to_file(self, file_path: str):
        """ Save the document collection to a file in JSONL format, one document per line."""
        for url, doc in self.documents.items():

            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(doc.to_dict()) + "\n")
            logging.info(f"Document {url} saved to {file_path}")


    def load_collection_from_file(self, file_path: str):
        """ Load the document collection from a file in JSONL format, one document per line."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
                for doc_data in docs_data:
                    doc = Document(**doc_data)
                    self.add_document(doc)
            logging.info(f"Loaded {len(self.documents)} documents from {file_path}")
        except Exception as e:
            logging.error(f"Failed to load document collection from {file_path}: {e}")

    def get_document(self, url: str) -> Optional[Document]:
        return self.documents.get(url)

    def remove_document(self, url: str):
        if url in self.documents:
            del self.documents[url]

    def get_all_documents_list(self) -> List[Document]:
        return list(self.documents.values())
    
    def get_documents_by_domain(self, domain: str) -> List[Document]:
        if domain not in self.domain_documents: 
            return []
        urls = self.domain_documents.get(domain, [])
        return [self.documents[url] for url in urls if url in self.documents]
