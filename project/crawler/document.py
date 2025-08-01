from readability import Document as ReadabilityDocument
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from urllib.parse import urlparse
from bs4 import BeautifulSoup, element
from pathlib import Path
from collections import defaultdict
import logging
import json
import re
import os
import hashlib
import time

HTML_FILE = "indexed_html.jsonl"
DOCS_FILE = "indexed_docs.jsonl"

@dataclass
class Document:
    url: str
    title: str = ""
    author: str = ""
    description: str = ""
    site_type: str = ""

    content_hash: str = field(default="", init=False)
    html: str = ""

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

    relevant_keywords: tuple = field(default=())
    relevance_score: int = 0

    last_crawl_timestamp: float = field(default_factory=time.time)
    soup: BeautifulSoup = None

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
        if self.content_hash == "":
            self.content_hash = hashlib.md5(self.get_content().encode()).hexdigest()

    def get_soup(self) -> BeautifulSoup:
        if not self.soup and self.html:
            self.soup = BeautifulSoup(self.html, 'lxml')
        return self.soup

    def get_content(self) -> str:
        if not self.html:
            return ""
        soup = self.get_soup()
        return soup.get_text(separator=' ', strip=True)

    def _update_html(self, html: str):
        """ Set new HTML content and update metrics."""
        self.html = html
        if html:
            self.update_metrics()
            self.content_hash = hashlib.md5(self.get_content().encode()).hexdigest()

    def update_metrics(self, crawl: bool = True):
        soup = self.get_soup()
        readable_doc = ReadabilityDocument(self.html)

        text = readable_doc.content() if readable_doc and readable_doc.content() else soup.get_text(separator=' ', strip=True)
        self.title = readable_doc.short_title() if readable_doc and readable_doc.short_title() else ""
        self.word_count = len(text.split())
        self.sentence_count = len([s for s in text.split('.') if s.strip()])
        self.paragraph_count = self.html.count('<p>')
        if crawl:
            self.last_crawl_timestamp = time.time()
            self.crawl_frequency += 1

        description = self.extract_description(soup, readable_doc)

        site_type = soup.find("meta", property="og:type")
        if isinstance(site_type, element.Tag):
            site_type = site_type.get("content")
        elif isinstance(site_type, element.NavigableString):
            site_type = str(site_type).strip()
        
        author = None
        for attr in ["author", "article:author"]:
            meta_author = soup.find("meta", attrs={"name": attr}) or soup.find("meta", attrs={"property": attr})
            if isinstance(meta_author, element.Tag):
                author = meta_author.get("content")
                break
            elif isinstance(meta_author, element.NavigableString):
                author = str(meta_author).strip()
                break
        if not author:
            author = "No author available"

        self.description = description if description else "No description available"
        self.site_type = site_type if site_type else "No type available"
        if isinstance(author, str):
            self.author = author
        else:
            self.author = "No author available"

        keyword_count = sum(1 for kw in self.relevant_keywords if re.search(kw, self.url, re.IGNORECASE))

        if keyword_count == 0:
            self.relevance_score -= 10
        else:
            self.relevance_score += keyword_count

    def extract_description(self, soup: BeautifulSoup, readable_doc: ReadabilityDocument, max_length=100) -> str:
        """Extract a summary description from the HTML soup."""
        summary_html = readable_doc.summary(html_partial=True) if readable_doc else None
        if summary_html:
            summary_soup = BeautifulSoup(summary_html, 'lxml')
            text = summary_soup.get_text(separator=' ', strip=True)
        if text: return text[:max_length]

        description = soup.find("meta", attrs={"name": "description"})  # if no summary, try to get description from meta tags
        if not description: description = soup.find("meta", attrs={"property": "og:description"})
        
        if description and description.get("content"):
            if isinstance(description, str):
                text = description
            else:
                text = description["content"]
        else:  # if no meta description, try to extract from paragraphs
            for tag in soup(["script", "style", "head", "footer", "nav"]):
                tag.decompose()
            p = soup.find("p")
            if p:
                words = p.get_text(separator=" ", strip=True).split()
                text = " ".join(words[:max_length])
        if not text:  # if still no text, extract from whole text
            words = soup.get_text(separator=' ', strip=True).split()
            text = " ".join(words[:max_length])
        if not text:    # if still no text, return a default message
            return "No description available"
        return text

    def is_duplicate(self, other: 'Document') -> bool:
        if not self.content_hash or not other.content_hash:
            return False
        return self.content_hash == other.content_hash

    def get_url_parts(self) -> Dict[str, Any]:
        parsed_url = urlparse(self.url)
        return {
            "domain": parsed_url.hostname or "",
            "subdomain": parsed_url.hostname.split('.')[0] if parsed_url.hostname else "",
        }

    def append_to_file(self, path: str = "data"):
        """ Append this document to a JSONL file."""
        info_path = os.path.join(path, DOCS_FILE)
        with open(info_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(self.info_to_dict()) + "\n")
        html_path = os.path.join(path, HTML_FILE)
        with open(html_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({self.url: self.html}) + "\n")

    def load_from_dict(self, data: Dict):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # # Recalculate derived fields
        # self.__post_init__()

    def info_to_dict(self) -> Dict:
        return {
            "url": self.url,
            "title": self.title,
            "content_hash": self.content_hash,
            "description": self.description,
            "author": self.author,
            "site_type": self.site_type,
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
            "embedding": self.embedding.tolist() if self.embedding is not None else None
        }


class DocumentCollection:
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.content_hashes: Dict[str, str] = {}
        self.domain_documents: Dict[str, List[str]] = defaultdict(list)

    def add_document(self, doc: Document) -> bool:
        # check if document content already in collection
        if doc.content_hash in self.content_hashes:
            existing_url = self.content_hashes[doc.content_hash]
            if existing_url != doc.url:
                logging.warning(f"Duplicate content detected: {doc.url} has same content as {existing_url}")
                return False

        # add to documents and content_hashes dicts
        self.documents[doc.url] = doc
        self.content_hashes[doc.content_hash] = doc.url

        # update domain documents
        if doc.url not in self.domain_documents[doc.domain]:
            self.domain_documents[doc.domain].append(doc.url)

        return True

    def add_document_and_save(self, doc: Document, path: str = "data") -> bool:
        """ Add document to collection with duplicate checking, then append to file if successful."""
        if self.add_document(doc):
            doc.append_to_file(path)
            return True
        return False

    def write_collection_to_file(self, path: str = "data"):
        """ Save the document collection to two files in JSONL format, one document per line."""
        info_path = os.path.join(path, DOCS_FILE)
        html_path = os.path.join(path, HTML_FILE)
        with open(info_path, 'w', encoding='utf-8') as docs:
            with open(html_path, 'w', encoding='utf-8') as html:
                for _, doc in self.documents.items():
                    html.write(json.dumps({doc.url: doc.html}) + "\n")
                    docs.write(json.dumps(doc.info_to_dict()) + "\n")
            logging.info(f"Saved {len(self.documents)} documents to {info_path} and {html_path}")

    def load_from_file(self, dir_path: str, load_html: bool = False):
        base = Path(dir_path)
        for pair in [
            (DOCS_FILE, self._add_doc),
            (HTML_FILE, self._add_html) if load_html else None
        ]:
            if pair is None: continue
            fn, handler = pair
            try:
                for line in (base / fn).read_text(encoding="utf-8").splitlines():
                    if not (line := line.strip()): continue
                    data = json.loads(line)
                    handler(data)
                logging.info(f"Processed {fn}, Loaded {len(self.documents)} documents.")
            except FileNotFoundError:
                logging.error(f"Missing file: {fn}")
            except Exception as e:
                logging.error(f"Error in {fn}: {e}")

    def _add_doc(self, d: dict):
        doc = Document(url="")
        doc.load_from_dict(d)
        if doc.url:
            new_url = doc.url.split("#")[0]
            self.documents[new_url] = doc
            self.documents[new_url].url = new_url
        else:
            logging.warning("Skipped doc without URL: %r", d)

    def _add_html(self, h: dict):
        url = list(h.keys())[0].split("#")[0]
        html = h[url]
        self.documents[url].html = html

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
