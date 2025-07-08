import logging
import random
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from collections import deque
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
        # Initialize as a list of (priority, url) tuples for heap operations
        # Lower priority values = higher priority (processed first)
        initial_urls = [(0, url) for url in (urls or [])]  # Default priority 0
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
        # Initialize document collection
        # This will hold the documents crawled by this instance
        self.doc_collection = DocumentCollection()
        self.write_frequency = 10  # Write to file every 10 documents

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
        try:
            html = self.download_sitehtml(url)
            english = self.is_english(html)
            if not english:
                logging.info(f"Page {url} skipped: not detected as English.")
                return
            
            text_lower = html.lower()
            keywords_found = any(kw in text_lower for kw in self.keywords)
            self.doc_collection.add_document(Document(url=url))
            if keywords_found:
                for linked_url in set(self.get_linked_urls(url, html)):
                    self.add_url_to_visit(linked_url, priority=1)
            else:
                logging.info(f"Page {url} skipped: no keywords found.")
        finally:
            with self.lock:
                self.visited_pages[url] = {
                    'keywords_found': keywords_found,
                    'is_english': english
                }
                logging.info(f"Visited {len(self.visited_pages)} pages (Visited {url}, english={english}, keywords_found={keywords_found})")

    def run(self, amount=None):
        """Run the crawler, processing URLs in priority order."""
        # start_url_amt = len(self.visited_pages)
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
        logging.info("Crawling complete.")

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
