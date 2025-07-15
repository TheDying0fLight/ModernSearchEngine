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
import math
from collections import defaultdict
import re
import urllib.robotparser as urobot
import json
import os


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
    def __init__(self,
                 seed: list,
                 max_workers=multiprocessing.cpu_count()//2,
                 keywords: list = [r't\S+bingen', 'eberhard karl', 'palmer', 'lustnau', r's\S+dstadt', 'neckarinsel', 'stocherkahn', 'bebenhausen'],
                 user_agents:list=default_user_agents,
                 use_proxies: bool = False,
                 verbose: bool = True,
                 auto_resume: bool = False,
                 state_file: str = "crawler_state.json",
                 doc_collection_file: str = "indexed_docs.jsonl"):
        self.verbose = verbose
        self.use_proxies = use_proxies
        self.keywords = keywords
        self.user_agents = user_agents
        self.max_workers = max_workers
        self.auto_resume = auto_resume # set if crawler will write state to file and resume from it
        self.state_file = state_file
        self.doc_collection_file = doc_collection_file

        self.seed = seed if isinstance(seed, list) else [seed]
        self.visited_pages = {}
        # Initialize as (priority, url, parent_url) tuples - parent_url is None for seed URLs
        initial_urls = [(0, url, None) for url in (seed or [])]  # same start priority for all seed URLs
        self.frontier = initial_urls
        heapq.heapify(self.frontier)
        self.proxy_manager = ProxyManager(max_workers=max_workers, verbose=self.verbose) if use_proxies else None

        self.lock = threading.Lock()
        # allowed hostname prefixes (e.g., language subdomains)
        self.allowed_lang_prefixes = ['en.']
        self.filtered_substrings = ['.php', 'File:', 'Special:', 'Talk:', 'Template']
        self.domain_last_access = {}
        self.domain_lock = threading.Lock()
        self.visit_lock = threading.Lock()
        self.domain_dict = defaultdict(lambda: defaultdict())

        # initialize document collection
        self.doc_collection = DocumentCollection()
        self.write_frequency = 10  # write to file every 10 documents
        self.output_file = None
        self.pending_docs = []  # cache for documents waiting to be written
        self.write_lock = threading.Lock()  # lock for thread-safe document write operations

        recrawl_interval = 60 * 60 * 24 * 7  # recrawl if site is older than 7 days
        self.recrawl_interval = recrawl_interval  # seconds



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
        """Get a random user agent header."""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }



    def is_captcha_page(self, html):
        """Check if the HTML content indicates a captcha page."""
        markers = ['captcha', 'g-recaptcha', 'hcaptcha', 'confirm this search was made by a human']
        text = BeautifulSoup(html, 'html.parser').get_text(separator=' ').lower()
        return any(m in text for m in markers)



    def download_sitehtml(self, url, headers_only=False):
        """Download HTML content from a URL, using proxies if configured."""
        last_exc = None
        attempt = 0
        domain = urlparse(url).hostname
        # request_fn = requests.head if headers_only else requests.get
        request_fn = requests.get

        with self.domain_lock:
            last_time = self.domain_dict[domain]['last_access']
            if last_time:
                now = time.time()
                delay = random.uniform(1, 3)
                wait_time = last_time + delay - now
                if wait_time > 0: time.sleep(wait_time)
        
        with self.domain_lock:
            self.domain_last_access[domain] = time.time()

        # use proxies if configured
        if self.proxy_manager and self.use_proxies:
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
        else:
            # direct request without proxies
            try:
                resp = request_fn(url, headers=self.get_random_headers(), timeout=10)
                resp.raise_for_status()
                return resp.text # if not headers_only else resp
            except Exception as e:
                logging.error(f"Direct request failed for {url}: {e}")
                raise



    def get_linked_urls(self, url, html):
        """Extract linked URLs from the HTML content."""
        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('/'): yield urljoin(url, href)
            elif href.startswith('http'): yield href



    def relevance_score(self, url, parent_url=None):
        """Calculate relevance score based on URL and keywords."""
        if not self.keywords:
            return 0
        
        # scoring based on keyword presence in URL
        score = sum(1 for kw in self.keywords if re.search(kw, url, re.IGNORECASE))
        
        # additional scoring based on URL structure
        parsed = urlparse(url)
        if parsed.hostname:
            score += 1
        
        # score shorter URLs higher (prefer top-level pages)
        path_parts = [p for p in parsed.path.split('/') if p]
        path_depth = len(path_parts)
        
        if path_depth == 0:  # homepage
            score += 5
        elif path_depth == 1:  # Top-level sections
            score += 3
        elif path_depth == 2:  # Second-level pages
            score += 2
        elif path_depth == 3:  # Third-level pages
            score += 1
        
        # add half the score of the parent
        if parent_url:
            parent_doc = self.doc_collection.get_document(parent_url)
            if parent_doc:
                score += parent_doc.relevance_score * .5
        
        # bump if URL is https
        if parsed.scheme == 'https':
            score += 1

        return math.ceil(score)



    def add_url_to_frontier(self, url, priority=1, parent_url=None):
        """Add URL to visit with given priority and parent URL. Lower values = higher priority."""
        with self.lock:
            # check if URL is already visited
            if url not in self.visited_pages:
                # check if URL is not already in the heap
                existing_urls = {item[1] for item in self.frontier}
                if url not in existing_urls:
                    # priority is negative of relevance score
                    priority = -self.relevance_score(url, parent_url)
                    heapq.heappush(self.frontier, (priority, url, parent_url))



    def is_useragent_allowed(self, url: str):
        """Check if the user agent is allowed to access the URL according to robots.txt."""
        parse = urlparse(url)
        domain = parse.hostname
        path = parse.path
        url = f"{parse.scheme}://{parse.netloc}"
        with self.domain_lock:
            try: return self.domain_dict[domain]['robot']('*', path)
            except: pass
            robot = urobot.RobotFileParser()
            robot.set_url(url + "/robots.txt")
            p = multiprocessing.Process(target=robot.read)
            p.start()
            p.join(1)
            if p.is_alive:self.domain_dict[domain]['robot'] = (lambda x, y: True)
            else: self.domain_dict[domain]['robot'] = robot.can_fetch
            return self.domain_dict[domain]['robot']('*', path)



    def is_english(self, html):
        """Check if the HTML content is in English."""
        text = BeautifulSoup(html, 'html.parser').get_text(separator=' ')
        try:
            lang = detect(text)
            return lang == 'en'
        except LangDetectException:
            logging.info("Language detection failed, assuming English")
            return True



    def crawl(self, url, parent_url=None):
        """Crawl a single URL, extract links, and add them to the frontier."""
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
            if not parent_url:
                parent_url = "Seed"
            
            doc = Document(url=url, html=html, parent_url=parent_url)

            # update metrics using the stored HTML
            doc.update_metrics()
            
            # add document to cache
            self.add_document_to_cache(doc)


            for linked_url in set(self.get_linked_urls(url, html)):
                self.add_url_to_frontier(linked_url, parent_url=url)

            # If keywords found, add linked URLs for further crawling
            # if keywords_found:
            #     for linked_url in set(self.get_linked_urls(url, html)):
            #         self.add_url_to_frontier(linked_url, priority=1, parent_url=url)
            # else:
            #     logging.info(f"Page {url} skipped for link extraction: no keywords found.")
                
        except Exception as e:
            logging.error(f"Error crawling {url}: {e}")
        finally:
            with self.lock:
                if parent_url:
                    self.visited_pages[url] = {
                        "keywords_found": keywords_found,
                        "parent_url": parent_url,
                    }
                    logging.info(f"Visited {len(self.visited_pages)} pages (Visited {url}, english={english}, keywords_found={keywords_found})")



    def run(self, amount=None, output_file="crawled_documents.jsonl"):
        """Run the crawler, processing URLs in priority order."""
        self.output_file = output_file
        
        # check for auto-resume possibility
        if self.auto_resume and self._can_resume():
            logging.info(f"Auto-resuming from previous state: {self.state_file}")
            try:
                self.load_state(self.state_file)
            except Exception as e:
                logging.warning(f"Failed to load previous state: {e}. Starting fresh crawl.")
            
            try:
                self.doc_collection.load_collection_from_file(self.doc_collection_file)
                # add stale documents back to frontier for re-crawling
                self.add_stale_docs_to_frontier()
            except Exception as e:
                logging.warning(f"Failed to load document collection: {e}. Starting fresh crawl.")
                self.doc_collection = DocumentCollection()
            
            print(self.get_crawling_stats())
        
        # start_url_amt = len(self.visited_pages)
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = set()
                while True:
                    with self.lock:
                        while len(futures) < self.max_workers and self.frontier:
                            _, next_url, parent_url = heapq.heappop(self.frontier)
                            if next_url not in self.visited_pages:
                                futures.add(executor.submit(self.crawl, next_url, parent_url))
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
            # save state for potential resumption
            if self.auto_resume:
                self.save_current_state(self.state_file)
            logging.info(self.get_crawling_stats())
        logging.info("Crawling complete.")



    def _can_resume(self) -> bool:
        """
        Check if previous crawler state and document collection exists and 
        is recent enough to resume.
        Returns True if can resume, False otherwise.
        """
        if not os.path.exists(self.state_file):
            logging.warning(f"State file {self.state_file} not found. Cannot resume.")
            return False
        if not os.path.exists(self.doc_collection_file):
            logging.warning(f"Document collection file {self.doc_collection_file} not found. Cannot resume.")
            return False

        # check if state file is recent (e.g., less than 30 days old)
        stat = os.stat(self.state_file)
        age_hours = (time.time() - stat.st_mtime) / 3600
        if age_hours > 30 * 24:  # state too old, start fresh
            logging.info(f"Previous state is {age_hours:.1f} hours old, starting fresh crawl")
            return False
        
        logging.info(f"Found recent crawler state ({age_hours:.1f} hours old)")
        return True
    


    def get_crawling_stats(self):
        """Get statistics about the crawling session."""
        with self.write_lock:
            return {
                'total_documents': len(self.doc_collection.documents),
                'visited_pages': len(self.visited_pages),
                'frontier': len(self.frontier)
            }
    


    def save_current_state(self, file_path: str = "crawler_state.json"):
        """Save the current frontier and visited pages to a file."""
        with self.write_lock:
            state = {
                'frontier': list(self.frontier),
                'visited_pages': self.visited_pages
            }
            with open(file_path, 'w') as f:
                json.dump(state, f)
            logging.info(f"Saved current state to {file_path}")
    


    def load_state(self, state_file: str):
        """Load the frontier and visited pages from a file."""
        with self.write_lock:
            with open(state_file, 'r') as f:
                state = json.load(f)
                self.frontier = state.get('frontier', [])
                heapq.heapify(self.frontier)
                self.visited_pages = state.get('visited_pages', {})
            logging.info(f"Loaded crawler state from {state_file}")



    def add_stale_docs_to_frontier(self):
        """Add stale documents to the frontier for recrawling."""
        recrawl_interval = self.recrawl_interval
        with self.lock:
            now = time.time()
            stale_docs = []
            for url, doc in self.doc_collection.documents.items():
                if now - doc.last_crawl_timestamp > recrawl_interval:
                    stale_docs.append((doc.relevance_score, url, doc.parent_url))
            
            # add stale documents to the frontier with their relevance score as priority
            for score, url, parent_url in stale_docs:
                self.add_url_to_frontier(url, priority=-score, parent_url=parent_url)
            logging.info(f"Added {len(stale_docs)} stale documents to the frontier for recrawling.")
    




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
