import logging
import random
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
# from collections import deque
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing
from langdetect import detect, LangDetectException
from .proxy_manager import ProxyManager
from .document import Document, DocumentCollection
import heapq


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)

default_user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
 " (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
 "(KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
 "(KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
]

class Crawler:
    def __init__(self, urls=None, max_workers=multiprocessing.cpu_count()//2, keywords=None, user_agents=None):
        self.visited_pages = {}
        # initialize as a list of (priority, url) tuples for heap operations
        initial_urls = [(0, url) for url in (urls or [])]  # default priority 0
        self.urls_to_visit = initial_urls
        heapq.heapify(self.urls_to_visit)
        self.proxy_manager = ProxyManager(max_workers=max_workers)
        self.lock = threading.Lock()
        self.max_workers = max_workers
        self.keywords = [kw.lower() for kw in (keywords or ['tubingen', 'tübingen', 'tuebingen'])]
        # allowed hostname prefixes (e.g., language subdomains)
        self.allowed_prefixes = ['www.', 'en.']
        self.filtered_substrings = ['.php', 'File:', 'Special:', 'Talk:', 'Template']
        self.domain_last_access = {}
        self.domain_lock = threading.Lock()
        self.user_agents = user_agents or default_user_agents
        # initialize document collection
        self.doc_collection = DocumentCollection()
        self.write_frequency = 10  # write to file every 10 documents
        self.output_file = None
        self.pending_docs = []  # cache for documents waiting to be written
        self.write_lock = threading.Lock()  # lock for thread-safe document write operations



    def add_document_to_cache(self, doc: Document):
        """Thread-safe method to add document to cache and handle batch writing."""
        with self.write_lock:
            # try to add to collection
            if self.doc_collection.add_document(doc):
                self.pending_docs.append(doc)
                logging.info(f"Added document {doc.url} to cache. Cache size: {len(self.pending_docs)}")
                
                # if write frequency reached, write to file
                if len(self.pending_docs) >= self.write_frequency:
                    self._write_pending_docs()



    def _write_pending_docs(self):
        """Write pending documents to file. Must be called with write_lock held."""
        if not self.pending_docs or not self.output_file:
            return
            
        try:
            # append all pending docs to file
            for doc in self.pending_docs:
                self.doc_collection.add_document_and_save(doc, self.output_file)

            logging.info(f"Appended {len(self.pending_docs)} documents to {self.output_file}")
            self.pending_docs.clear()
        except Exception as e:
            logging.error(f"Failed to write documents to file: {e}")



    def finalize_crawl(self, backup: bool = True):
        """Write any remaining documents and save final collection."""
        with self.write_lock:
            # append any remaining pending documents
            if self.pending_docs:
                self._write_pending_docs()
            
            # save entire collection as backup
            if self.output_file and self.doc_collection.documents and backup:
                backup_file = self.output_file.replace('.jsonl', '_complete.jsonl')
                self.doc_collection.write_collection_to_file(backup_file)
                logging.info(f"Saved backup collection to {backup_file}")



    def get_random_headers(self):
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }



    def is_captcha_page(self, html):
        markers = ['captcha', 'g-recaptcha', 'hcaptcha', 'confirm this search was made by a human']
        text = BeautifulSoup(html, 'html.parser').get_text(separator=' ').lower()
        return any(m in text for m in markers)



    def download_sitehtml(self, url):
        last_exc = None
        attempt = 0
        domain = urlparse(url).hostname

        with self.domain_lock:
            last_time = self.domain_last_access.get(domain)
            now = time.time()
            if last_time:
                delay = random.uniform(1, 3)
                wait_time = last_time + delay - now
                if wait_time > 0:
                    time.sleep(wait_time)
            self.domain_last_access[domain] = time.time()

        while self.proxy_manager.proxies:
            attempt += 1
            entry = self.proxy_manager.get_random_proxy()
            proxy_url = entry['proxy']
            proxies = {'http': proxy_url, 'https': proxy_url}
            headers = self.get_random_headers()
            start = time.time()
            try:
                resp = requests.get(url, proxies=proxies, headers=headers, timeout=10)
                html = resp.text
                elapsed = time.time() - start
                if 400 <= resp.status_code < 600:
                    logging.info(f"Received {resp.status_code} for {url} via {proxy_url}, stopping retries.")
                    return resp.text
                if self.is_captcha_page(html):
                    logging.warning(f"Captcha detected for {url}, retrying after backoff.")
                    time.sleep(random.uniform(5, 10))
                    continue
                resp.raise_for_status()
                self.proxy_manager.update_proxy_quality(proxy_url, success=True, elapsed=elapsed)
                return html
            except Exception as e:
                last_exc = e
                self.proxy_manager.update_proxy_quality(proxy_url, success=False)
                logging.warning(f"Proxy {proxy_url} failed on attempt {attempt}: {e}")
        logging.error(f"All proxy attempts failed for {url}")
        if last_exc:
            raise last_exc
        else:
            raise Exception(f"No proxies available to crawl {url}")



    def get_linked_urls(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a', href=True):
            if any(map(lambda x: x in link, self.filtered_substrings)): continue
            href = link['href']
            if href.startswith('/'):
                href = urljoin(url, href)
            elif not href.startswith('http'):
                continue
            parsed = urlparse(href)
            host = parsed.hostname or ''
            if any(host.startswith(p) for p in self.allowed_prefixes):
                yield href



    def add_url_to_visit(self, url, priority=1):
        """Add URL to visit with given priority. Lower values = higher priority."""
        with self.lock:
            # Check if URL is already visited
            if url not in self.visited_pages:
                # Check if URL is not already in the heap
                existing_urls = {item[1] for item in self.urls_to_visit}
                if url not in existing_urls:
                    heapq.heappush(self.urls_to_visit, (priority, url))



    def is_english(self, html):
        text = BeautifulSoup(html, 'html.parser').get_text(separator=' ')
        try:
            lang = detect(text)
            return lang == 'en'
        except LangDetectException:
            logging.info("Language detection failed, assuming English")
            return True



    def crawl(self, url):
        keywords_found = False
        english = False
        # start_time = time.time()
        
        try:
            html = self.download_sitehtml(url)
            # load_time_ms = (time.time() - start_time) * 1000
            
            english = self.is_english(html)
            if not english:
                logging.info(f"Page {url} skipped: not detected as English.")
                return
            
            text_lower = html.lower()
            keywords_found = any(kw in text_lower for kw in self.keywords)
            
            # initialize Document object
            doc = Document(url=url)
            doc.html = html  # store raw HTML
                        
            # update metrics using the stored HTML
            doc.update_metrics()
            
            # add document to cache
            self.add_document_to_cache(doc)
            
            # If keywords found, add linked URLs for further crawling
            if keywords_found:
                for linked_url in set(self.get_linked_urls(url, html)):
                    self.add_url_to_visit(linked_url, priority=1)
            else:
                logging.info(f"Page {url} skipped for link extraction: no keywords found.")
                
        except Exception as e:
            logging.error(f"Error crawling {url}: {e}")
        finally:
            with self.lock:
                self.visited_pages[url] = {
                    'keywords_found': keywords_found,
                    'is_english': english
                }
                logging.info(f"Visited {len(self.visited_pages)} pages (Visited {url}, english={english}, keywords_found={keywords_found})")

    def run(self, amount=None, output_file="crawled_documents.jsonl"):
        """Run the crawler, processing URLs in priority order."""
        self.output_file = output_file
        # start_url_amt = len(self.visited_pages)
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = set()
                while True:
                    with self.lock:
                        while len(futures) < self.max_workers and self.urls_to_visit:
                            priority, next_url = heapq.heappop(self.urls_to_visit)
                            if next_url not in self.visited_pages:
                                futures.add(executor.submit(self.crawl, next_url))
                    if not futures: break
                    if amount is not None and len(self.visited_pages) >= amount: break
                    done, futures = set(as_completed(futures)), set()
                    for future in done:
                        try:
                            future.result()
                        except Exception:
                            pass
        finally:
            # finalize the crawl to save any remaining documents in the cache
            self.finalize_crawl()
        logging.info("Crawling complete.")

    def get_crawling_stats(self):
        """Get statistics about the crawling session."""
        with self.write_lock:
            return {
                'total_documents': len(self.doc_collection.documents),
                'pending_documents': len(self.pending_docs),
                'visited_pages': len(self.visited_pages),
                'urls_to_visit': len(self.urls_to_visit)
            }

def test_url(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/'):
            href = urljoin(url, href)
        print(href)
    return resp
