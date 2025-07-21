from urllib.parse import urljoin, urlparse, ParseResult
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from collections import defaultdict
from langdetect import detect, LangDetectException
from langcodes import Language
from termcolor import colored
from typing import Optional
import validators
import re
import json
import os
import warnings
import mimetypes
import logging
import random
import requests
import time
import threading
import multiprocessing
import sys
import traceback

from .utils import predict_language_from_url
from .utils import TrackingThreadPoolExecutor, TimeoutRobotFileParser
from .proxy_manager import ProxyManager
from .document import Document, DocumentCollection, HTML_FILE, DOCS_FILE


logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

default_user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    " (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
]

PARSER = "lxml"
DEFAULT_DELAY = 1
TIMEOUT = 5
STATE_FILE = "crawler_state.json"

class Crawler:
    def __init__(self,
                 seed: list[str],
                 max_workers: int = multiprocessing.cpu_count() // 2,
                 keywords: list = [r't\S{1,6}bingen', 'neckar island'],
                 user_agents: list = default_user_agents,
                 use_proxies: bool = False,
                 auto_resume: bool = False,
                 path: str = "data"):
        self.use_proxies = use_proxies
        self.keywords = keywords
        self.user_agents = user_agents
        self.max_workers = max_workers
        self.auto_resume = auto_resume

        self.path = path
        self.backup_path = f"{path}/backup"
        self.state_path = os.path.join(path, STATE_FILE)
        self.doc_collection_path = os.path.join(path, DOCS_FILE)
        self.html_path = os.path.join(path, HTML_FILE)

        self.write_lock = threading.Lock()

        self.proxy_manager = ProxyManager(max_workers=max_workers) if use_proxies else None

        self.frontier_lock = threading.Lock()
        self.seed = seed if isinstance(seed, list) else [seed]
        # initialize as (priority, url, parent_url, recrawl) tuples - parent_url is None for seed URLs
        initial_urls = [(0.0, url, None, False) for url in (seed or [])]
        self.frontier = initial_urls

        self.visited_lock = threading.Lock()
        self.visited_pages = {}

        self.domain_lock = threading.Lock()
        self.domain_dict = defaultdict(lambda: defaultdict(int))
        self.domain_locks = defaultdict(threading.Lock)

        self.delay_lock = threading.Lock()
        self.domain_delay = {}

        # initialize document collection
        self.doc_lock = threading.Lock()
        self.doc_collection = DocumentCollection()
        self.write_frequency = 10  # write to file every 10 documents
        self.pending_docs = []  # cache for documents waiting to be written

        recrawl_interval = 60 * 60 * 24 * 7  # recrawl if site is older than 7 days
        self.recrawl_interval = recrawl_interval  # seconds

        for file in [self.state_path, self.doc_collection_path, self.html_path]:
            backup_file = file.replace(self.path, self.backup_path)
            if os.path.exists(file):
                if not auto_resume:
                    os.remove(file)
                    logging.info(f"Removed existing file: {file}")
            if os.path.exists(backup_file):
                if not auto_resume:
                    os.remove(backup_file)
                    logging.info(f"Removed existing backup file: {backup_file}")
            else:
                os.makedirs(os.path.dirname(file), exist_ok=True)
                # also create backup folder structure
                os.makedirs(os.path.dirname(backup_file), exist_ok=True)

    def get_domain(self, parse: ParseResult):
        return ".".join(parse.hostname.split(".")[-2:]) if parse.hostname else ""

    def add_document_to_cache(self, doc: Document):
        """Thread-safe method to add document to cache and handle batch writing."""
        should_save_state = False
        with self.write_lock:
            # try to add to collection
            if self.doc_collection.add_document(doc):
                self.pending_docs.append(doc)
                thread_id = threading.current_thread().ident
                logging.debug(
                    f"[Thread {thread_id}]: Added new document {doc.url} to cache. Cache size: {len(self.pending_docs)}")

                # if write frequency reached, write to file
                if len(self.pending_docs) >= self.write_frequency:
                    self._write_pending_data()
                    should_save_state = True

        if should_save_state: self.save_current_state()

    def _write_pending_data(self):
        """Write pending data to file. Must be called with write_lock held."""
        if not self.pending_docs or not self.doc_collection_path:
            return

        try:
            # append all pending docs to file
            for doc in self.pending_docs:
                self.doc_collection.add_document_and_save(doc, self.path)
                full_doc = self.doc_collection.get_document(doc.url)
                if full_doc:
                    full_doc.html = ""  # clear HTML to save memory
                    self.doc_collection.documents[doc.url] = full_doc

            logging.info(f"Appended {len(self.pending_docs)} documents to {self.doc_collection_path}")
            self.pending_docs.clear()

            # overwrite crawler state file with current state
            # self.save_current_state(self.state_file)
        except Exception as e:
            logging.error(f"Failed to write data to file: {e}")

    def finalize_crawl(self, backup: bool = True):
        """Write any remaining documents and save final collection."""
        with self.write_lock:
            # append any remaining pending documents
            if self.pending_docs:
                self._write_pending_data()

            # save entire collection as backup
            if self.doc_collection_path and self.doc_collection.documents and backup:
                self.doc_collection.write_collection_to_file(self.backup_path)
                logging.info(f"Saved backup collection to {self.backup_path}")

    def get_random_headers(self):
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

    def is_captcha_page(self, html):
        markers = ['captcha', 'g-recaptcha', 'hcaptcha', 'confirm this search was made by a human']
        text = BeautifulSoup(html, PARSER).get_text(separator=' ').lower()
        return any(m in text for m in markers)

    def download_site(self, entry, headers: bool = False):
        _, url, _, _ = entry
        last_exc = None
        attempt = 0
        domain = self.get_domain(urlparse(url))
        request_fn = requests.head if headers else requests.get

        if not self.use_proxies:
            resp = request_fn(url, headers=self.get_random_headers(), timeout=TIMEOUT)
            if resp.status_code == 429:
                with self.delay_lock:
                    self.domain_delay[domain] = self.domain_delay.get(domain, DEFAULT_DELAY) + 1
                    delay = self.domain_delay[domain]
                with self.frontier_lock: self.frontier.append(entry)
                logging.warning(colored(f'429 Received: Delay increased to {delay}s for: {domain}', 'red'))
            resp.raise_for_status()
            return resp

        while self.proxy_manager.proxies:
            attempt += 1
            entry = self.proxy_manager.get_random_proxy()
            proxy_url = entry['proxy']
            proxies = {'http': proxy_url, 'https': proxy_url}
            headers = self.get_random_headers()
            start = time.time()
            try:
                resp = request_fn(url, proxies=proxies, headers=headers, timeout=TIMEOUT)
                text = resp.text
                elapsed = time.time() - start
                if 400 <= resp.status_code < 600:
                    logging.info(f"Received {resp.status_code} for {url} via {proxy_url}, stopping retries.")
                    return text
                if self.is_captcha_page(text):
                    logging.warning(f"Captcha detected for {url}, stopping retries.")
                    return text
                resp.raise_for_status()
                self.proxy_manager.update_proxy_quality(proxy_url, success=True, elapsed=elapsed)
                return resp
            except Exception as e:
                last_exc = e
                self.proxy_manager.update_proxy_quality(proxy_url, success=False)
                logging.warning(f"Proxy {proxy_url} failed on attempt {attempt}: {e}")
        logging.error(f"All attempts failed for {url}")
        raise last_exc

    def get_linked_urls(self, url: str, soup: BeautifulSoup):
        """Extract linked URLs from the HTML content."""
        for link in soup.find_all('a', href=True):
            href = link['href']

            if href.startswith('/'): yield urljoin(url, href)
            elif href.startswith('http'): yield href

    def should_crawl_url(self, url):
        """Check if the URL should be crawled based on robots.txt and other criteria."""
        useless = ["file:", "category:", "template:", "user:", "help",
                   "user_talk:", "talk:", "template_talk:", "&diff=",
                   "&oldid=", "&restore=", "&printable=", "action="]
        if any(pattern in url.lower() for pattern in useless): return False

        with self.visited_lock:
            if url in self.visited_pages: return False
        if self.doc_collection.get_document(url):
            return False

        # skip mobile versions
        if 'm.wikipedia.org' in url.lower() or '.m.' in url.lower():
            return False

        with self.frontier_lock: existing_urls = {item[1] for item in self.frontier}
        if url in existing_urls: return False

        if not validators.url(url): return False
        if predict_language_from_url(url) not in ["und", "en", "eu", "root"]:
            return False

        t = mimetypes.guess_type(url)[0]
        if str(t).split("/")[0] not in ["None", "text"]: return False

        if not self.is_useragent_allowed(url):
            logging.debug(f"URL {url} is disallowed by robots.txt")
            return False
        return True

    def is_useragent_allowed(self, url: str):
        parse = urlparse(url)
        domain = self.get_domain(parse)
        path = parse.path
        base_url = f"{parse.scheme}://{parse.netloc}"

        with self.domain_lock:
            entry = self.domain_dict[domain]
            if 'robot' in entry: return entry['robot']('*', path)

        robot = TimeoutRobotFileParser()
        robot.set_url(base_url + "/robots.txt")
        robot.read(3)

        with self.domain_lock:
            self.domain_dict[domain]['robot'] = robot.can_fetch
            return self.domain_dict[domain]['robot']('*', path)

    def add_urls_to_frontier(self, urls: set, priority=1, parent_url=None, recrawl=False):
        """Add URL to visit with given priority and parent URL. Lower values = higher priority."""
        added = 0
        for url in urls:
            if not self.should_crawl_url(url): continue

            priority = -self.relevance_score(url, parent_url)
            with self.frontier_lock:
                self.frontier.append((priority, url, parent_url, recrawl))
            added += 1
        return added

    def relevance_score(self, url, parent_url=None):
        """Calculate relevance score based on URL and keywords."""
        if not self.keywords: return 0
        score = sum(10 for kw in self.keywords if re.search(kw, url.lower()))
        score += max(0, 40 - len(url))  # deeper paths get lower score

        if parent_url:
            parent_doc = self.doc_collection.get_document(parent_url)
            if parent_doc:
                score += parent_doc.relevance_score * .5  # add half the score of the parents content

        thread_id = threading.current_thread().ident
        logging.debug(f"[Thread {thread_id}]: Relevance score for {url}: {score}")
        return score

    def is_english(self, soup: BeautifulSoup):
        text = soup.get_text(separator=' ')
        try:
            lang = detect(text)
            return lang == 'en'
        except LangDetectException:
            logging.debug("Language detection failed, assuming English")
            return True

    def crawl(self, entry, lock: threading.Lock):
        try: self._crawl(entry)
        finally: lock.release()

    def _crawl(self, entry):
        _, url, parent_url, recrawl = entry
        keywords_found = False
        english = False
        logging.debug(f"Crawling {url}")
        if not parent_url: parent_url = "Seed"

        try:
            headers = self.download_site(entry, headers=True).headers
            try: english = Language.get(headers['content-language']).language == 'en'
            except: pass
            html = self.download_site(entry).text
            if not html: raise Exception("Empty HTML content.")
            if len(html.encode()) > 1e8:
                raise Exception(f"Page to large, Bytes: {len(html.encode())}, {url}")

            soup = BeautifulSoup(html, PARSER)
            english |= self.is_english(soup)
            if not english: raise BrokenPipeError("Url not detected as English.")

            keywords_found = any(re.search(reg, html.lower()) for reg in self.keywords)
            if not keywords_found: raise BrokenPipeError("No keywords found.")

            doc = self.doc_collection.get_document(url)
            if recrawl and doc: doc._update_html(html)
            else: doc = Document(url=url, html=html, parent_url=parent_url, relevant_keywords=self.keywords)

            doc.update_metrics()

            self.add_document_to_cache(doc)

            urls = list(set(self.get_linked_urls(url, soup)))
            random.shuffle(urls)
            urls = urls[:100]
            added = self.add_urls_to_frontier(urls, parent_url=url)
            with self.frontier_lock:
                logging.info(f"Frontier size: {len(self.frontier)}, "
                             f"Added {added}/{len(urls)} URLs from {url}")
        except BrokenPipeError as e: logging.debug(f"Error crawling {url}: {e}")
        except Exception as e:
            logging.error(f"Error crawling {url}: {e}")
            # exc_type, exc_value, exc_traceback = sys.exc_info()
            # tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            # logging.error(f"Full traceback:\n{''.join(tb_lines)}")
        finally:
            doc = self.doc_collection.get_document(url)
            relevance = doc.relevance_score if doc else "no relevance score"
            with self.visited_lock:
                self.visited_pages[url] = {
                    "keywords_found": keywords_found,
                    "parent_url": parent_url,
                }
                logging.debug(
                    f"Visited {len(self.visited_pages)} pages (english={english}, relevance={relevance}, URL: {url})")

    def run(self, amount: Optional[int] = None):
        if self.auto_resume and self._can_resume():
            logging.info(f"Auto-resuming from previous state: {self.state_path}")
            try:
                self.load_state()
            except Exception as e:
                logging.warning(f"Failed to load previous state: {e}. Starting fresh crawl.")

            try:
                self.doc_collection.load_from_file(self.path)
                self.add_stale_docs_to_frontier()
            except Exception as e:
                logging.warning(f"Failed to load document collection: {e}. Starting fresh crawl.")
                self.doc_collection = DocumentCollection()

            print(self.get_crawling_stats())
            if isinstance(self.visited_pages, dict) and isinstance(amount, int):
                amount += len(self.visited_pages)

        with TrackingThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = set()
            def update_futures():
                visiting = list(map(lambda x: x[2], futures))
                logging.info(colored(f'Active threads: {executor.active_count}, Scheduled: {len(futures)}, '
                                    f'{self.get_crawling_stats()}, Sites: {visiting[:20]} ...', 'green'))
                filtered = set(filter(lambda f: f[0].done(), futures))
                futures.difference_update(filtered)

            try:
                while True:
                    prior_time = time.time()
                    with self.visited_lock: amt_visited = len(self.visited_pages)
                    if not amount or amt_visited <= amount:
                        futures.update(self.schedule_urls(executor))
                    if not futures: break

                    post_time = time.time()
                    diff = post_time - prior_time
                    if diff < 2: time.sleep(2 - diff)
                    update_futures()
            finally:
                while futures:
                    time.sleep(2)
                    update_futures()
                self.cleanup_and_shutdown()

    def schedule_urls(self, executor: TrackingThreadPoolExecutor):
        futures = set()
        to_remove = set()
        with self.frontier_lock: frontier = self.frontier.copy()
        for entry in sorted(frontier, key=lambda x: x[0]):
            domain = self.get_domain(urlparse(entry[1]))

            with self.domain_lock:
                now = time.time()
                if self.domain_dict[domain]['next_access'] > now: continue
                lock = self.domain_locks[domain]
                if lock.locked(): continue
                else: lock.acquire()

            with self.delay_lock:
                delay = self.domain_delay.get(domain, DEFAULT_DELAY)
                delay += DEFAULT_DELAY + random.uniform(0, (delay / 8))

            with self.domain_lock:
                self.domain_dict[domain]['next_access'] = now + delay

            to_remove.add(entry)
            futures.add((executor.submit(self.crawl, entry, lock), time.time(), entry[1]))
        with self.frontier_lock:
            self.frontier = [entry for entry in self.frontier if entry not in to_remove]
        return futures

    def _can_resume(self) -> bool:
        """
        Check if previous crawler state and document collection exists and
        is recent enough to resume.
        Returns True if can resume, False otherwise.
        """
        if not os.path.exists(self.state_path):
            logging.info(f"State file {self.state_path} not found. Cannot resume.")
            return False
        if not os.path.exists(self.doc_collection_path):
            logging.info(f"Document collection file {self.doc_collection_path} not found. Cannot resume.")
            return False

        stat = os.stat(self.state_path)
        age_hours = (time.time() - stat.st_mtime) / 3600
        logging.info(f"Found recent crawler state ({age_hours:.1f} hours old)")
        return True

    def get_crawling_stats(self):
        """Get statistics about the crawling session."""
        with self.doc_lock:      stats = {'total_documents': len(self.doc_collection.documents)}
        with self.visited_lock:  stats['visited_pages'] = len(self.visited_pages)
        with self.frontier_lock: stats['frontier'] = len(self.frontier)
        return stats

    def save_current_state(self):
        """Save the current frontier and visited pages to a file."""
        with self.visited_lock, self.frontier_lock, self.delay_lock:
            state = {
                "visited_pages": self.visited_pages.copy(),
                "frontier": self.frontier.copy(),
                "domain_delay": self.domain_delay.copy(),
            }
        with self.write_lock:
            with open(self.state_path, 'w') as f: json.dump(state, f)
        logging.debug(f"Saved current state to {self.state_path}")

    def load_state(self):
        """Load the frontier and visited pages from a file."""
        with self.write_lock:
            with open(self.state_path, 'r') as f:
                state = json.load(f)

                frontier_data = state.get('frontier', [])
                self.frontier = [tuple(item) for item in frontier_data if isinstance(item, list) and len(item) == 4]

                self.visited_pages = state.get('visited_pages', {})
                self.domain_delay = state.get('domain_delay', {})
            logging.info(f"Loaded crawler state from {self.state_path}")
            logging.info(f"Stats: {self.get_crawling_stats()}")

    def add_stale_docs_to_frontier(self):
        """
        Add stale documents to the frontier for recrawling
        and remove from visited pages.
        """
        recrawl_interval = self.recrawl_interval
        with self.doc_lock:
            now = time.time()
            stale_docs = []
            for url, doc in self.doc_collection.documents.items():
                if now - doc.last_crawl_timestamp > recrawl_interval:
                    stale_docs.append((doc.relevance_score, url, doc.parent_url))

            # add stale documents to the frontier with their relevance score as priority
            # remove url from visited pages
            for score, url, parent_url in stale_docs:
                self.add_urls_to_frontier(url, priority=-score, parent_url=parent_url, recrawl=True)
                with self.visited_lock: del self.visited_pages[url]
            logging.info(f"Added {len(stale_docs)} stale documents to the frontier for recrawling.")

    def cleanup_and_shutdown(self):
        """Perform cleanup tasks on shutdown."""
        logging.info("Crawler is shutting down, performing cleanup...")
        self.finalize_crawl(backup=True)
        if self.auto_resume:
            self.save_current_state()

        stats = self.get_crawling_stats()
        logging.info(f"Final stats: {stats}")
        logging.info("Cleanup complete. Crawler has shut down gracefully.")
