import time
from dataclasses import dataclass, field
import hashlib
from typing import List, Dict, Set, Optional, Any
from urllib.parse import urlparse
import logging
import json
from bs4 import BeautifulSoup
import numpy as np
import math


@dataclass
class Document:
    DEFAULT_KEYWORDS = (r't\S+bingen', 'eberhard karl', 'palmer', 'lustnau', r's\S+dstadt', 'neckarinsel', 'stocherkahn', 'bebenhausen')

    url: str
    title: str = ""
    content_hash: str = field(default="", init=False)

    html: str = ""
    meta_description: str = ""

    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    canonical_url: str = ""

    parent_url: Optional[str] = None
    domain: str = field(default="", init=False)
    subdomain: str = field(default="", init=False)
    path_depth: int = field(default=0, init=False)
    has_ssl: bool = field(default=False, init=False)

    crawl_frequency: int = 0  # number of times crawled

    embedding = None  # Placeholder for embedding, can be set later

    relevant_keywords: tuple = field(default=DEFAULT_KEYWORDS)
    relevance_score: int = 0

    last_crawl_timestamp: float = field(default_factory=time.time)



    def __post_init__(self):
        parsed_url = urlparse(self.url)
        url_parts = self.get_url_parts()

        self.domain = url_parts["domain"]
        self.subdomain = url_parts["subdomain"]
        self.path_depth = len([p for p in parsed_url.path.split('/') if p])
        self.canonical_url = self.canonical_url or self.url
        self.has_ssl = parsed_url.scheme == 'https'

        if self.html:
            self.update_metrics()
            self.content_hash = hashlib.md5(self.get_content().encode()).hexdigest()


    def get_content(self) -> str:
        """ Returns the content of the document, extracting text from HTML."""
        if not self.html:
            return ""
        soup = BeautifulSoup(self.html, 'html.parser')
        return soup.get_text(separator=' ', strip=True)


    def _update_html(self, html: str):
        """ Set new HTML content and update metrics."""
        self.html = html
        if html:
            self.update_metrics()
            self.content_hash = hashlib.md5(self.get_content().encode()).hexdigest()


    def update_metrics(self):
        soup = BeautifulSoup(self.html, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        self.title = soup.title.string if soup.title and soup.title.string else ""
        self.word_count = len(text.split())
        self.sentence_count = len([s for s in text.split('.') if s.strip()])
        self.paragraph_count = self.html.count('<p>')
        self.last_crawl_timestamp = time.time()
        self.crawl_frequency += 1

        # metas = soup.find_all("meta")
        # meta_desc_tags = [meta.get('content', '') for meta in metas 
        #           if meta.get('name', '').lower() == 'description']

        # self.meta_description = meta_desc_tags[0] if meta_desc_tags else ""

        keyword_count = 0
        for keyword in self.relevant_keywords:
            if keyword.lower() in self.get_content().lower():
                keyword_count += self.get_content().lower().count(keyword.lower())

        if keyword_count == 0:
            self.relevance_score -= 10 # reduce score if no keywords found
        else:
            self.relevance_score += math.ceil(np.log(keyword_count))


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

    
    def append_to_file(self, file_path: str):
        """ Append this document to a JSONL file."""
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(self.to_dict()) + "\n")
        logging.info(f"Document {self.url} appended to {file_path}")


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
            "meta_description": self.meta_description,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "paragraph_count": self.paragraph_count,
            "has_ssl": self.has_ssl,
            "canonical_url": self.canonical_url,
            "domain": self.domain,
            "subdomain": self.subdomain,
            "path_depth": self.path_depth,
            "crawl_frequency": self.crawl_frequency,
            "parent_url": self.parent_url,
            "relevance_score": self.relevance_score,
            "last_crawl_timestamp": self.last_crawl_timestamp,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "html": self.html
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
    

    def add_document_and_save(self, doc: Document, file_path: str) -> bool:
        """ Add document to collection with duplicate checking, then append to file if successful."""
        if self.add_document(doc):
            doc.append_to_file(file_path)
            return True
        return False


    def write_collection_to_file(self, file_path: str = "indexed_docs.jsonl"):
        """ Save the document collection to a file in JSONL format, one document per line."""
        with open(file_path, 'w', encoding='utf-8') as f:
            for url, doc in self.documents.items():
                f.write(json.dumps(doc.to_dict()) + "\n")
        logging.info(f"Saved {len(self.documents)} documents to {file_path}")


    def load_collection_from_file(self, file_path: str):
        """ Load the document collection from a file in JSONL format, one document per line."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        doc_data = json.loads(line)
                        doc = Document(url="")
                        doc.load_from_dict(doc_data)
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
