import logging
import random
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
from collections import defaultdict
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import multiprocessing
from langdetect import detect, LangDetectException
from .proxy_manager import ProxyManager
from langcodes import Language
import urllib.robotparser as urobot
import validators
import re
import json
import os
import socket

from .utils import predict_language_from_url, uniquify

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.DEBUG
)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
socket.setdefaulttimeout(2)

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
                 urls: list,
                 max_workers: int = multiprocessing.cpu_count() // 2,
                 keywords: list = [r't\S+bingen'],
                 user_agents: list = default_user_agents,
                 use_proxies: bool = True,
                 verbose: bool = False,
                 out_path: str = "data/crawled.jsonl"):
        self.verbose = verbose
        self.use_proxies = use_proxies
        self.keywords = keywords
        self.user_agents = user_agents
        self.max_workers = max_workers
        self.out_path = uniquify(out_path)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        self.visited_pages = {}
        self.urls_to_visit = set(urls)
        self.proxy_manager = ProxyManager(max_workers=max_workers, verbose=False) if use_proxies else None
        self.allowed_lang_prefixes = ['en']
        self.domain_lock = threading.Lock()
        self.visit_lock = threading.Lock()
        self.domain_dict = defaultdict(lambda: defaultdict(str))

    def get_random_headers(self):
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

    def is_captcha_page(self, html):
        markers = ['captcha', 'g-recaptcha', 'hcaptcha', 'confirm this search was made by a human']
        text = BeautifulSoup(html, 'html.parser').get_text(separator=' ').lower()
        return any(m in text for m in markers)

    def download_url(self, url, headers_only=False):
        last_exc = None
        attempt = 0
        domain = urlparse(url).hostname
        request_fn = requests.head if headers_only else requests.get

        with self.domain_lock:
            last_time = self.domain_dict[domain]['last_access']
        if last_time:
            now = time.time()
            delay = random.uniform(5, 7)
            wait_time = last_time + delay - now
            if wait_time > 0: time.sleep(wait_time)
        with self.domain_lock:
            self.domain_dict[domain]['last_access'] = time.time()

        if not self.use_proxies:
            try:
                resp = request_fn(url, headers=self.get_random_headers(), timeout=10)
                resp.raise_for_status()
                return resp
            except Exception as e:
                logging.error(f"Direct request failed for {url}: {e}")
                raise

        while self.proxy_manager.proxies:
            attempt += 1
            entry = self.proxy_manager.get_random_proxy()
            proxy_url = entry['proxy']
            proxies = {'http': proxy_url, 'https': proxy_url}
            headers_only = self.get_random_headers()
            start = time.time()
            try:
                resp = request_fn(url, proxies=proxies, headers=headers_only, timeout=10)
                elapsed = time.time() - start
                if 400 <= resp.status_code < 600:
                    logging.info(f"Received {resp.status_code} for {url} via {proxy_url}, stopping retries.")
                    return resp.text
                if self.is_captcha_page(resp.text):
                    logging.warning(f"Captcha detected for {url}, stopping retries.")
                    return resp.text
                resp.raise_for_status()
                self.proxy_manager.update_proxy_quality(proxy_url, success=True, elapsed=elapsed)
                return resp
            except Exception as e:
                last_exc = e
                self.proxy_manager.update_proxy_quality(proxy_url, success=False)
                logging.warning(f"Proxy {proxy_url} failed on attempt {attempt}: {e}")
        logging.error(f"All attempts failed for {url}")
        raise last_exc

    def get_linked_urls(self, url: str, html: str):
        soup = BeautifulSoup(html, 'html.parser')
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('/'): href = yield urljoin(url, href)
            elif href.startswith('http'): yield href

    def is_useragent_allowed(self, url: str):
        parse = urlparse(url)
        domain = parse.hostname
        path = parse.path
        base_url = f"{parse.scheme}://{parse.netloc}"

        with self.domain_lock:
            entry = self.domain_dict[domain]
            if 'robot' in entry: return entry['robot']('*', path)

        robot = urobot.RobotFileParser()
        robot.set_url(base_url + "/robots.txt")
        robot.read()

        with self.domain_lock:
            # entry['robot'] = lambda *args, **kwargs: True
            self.domain_dict[domain]['robot'] = robot.can_fetch
            return self.domain_dict[domain]['robot']('*', path)

    def add_urls_to_visit(self, urls: list[str]):
        added = 0
        language_denied = []
        for url in set(urls):
            with self.visit_lock:
                if url in self.visited_pages: continue
            if not validators.url(url): continue
            if predict_language_from_url(url) not in ["und", "en"]:
                language_denied.append(url)
                continue
            if not self.is_useragent_allowed(url): continue
            with self.visit_lock:
                if url in self.urls_to_visit: continue
                self.urls_to_visit.add(url)
            added += 1
        if language_denied: logging.info(f"Assumed non english: {language_denied}")
        return added

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
        if self.verbose: logging.info(f"Crawling {url}")
        try:
            headers = self.download_url(url, headers_only=True).headers
            try: english = Language.get(headers['content-language']).language == 'en'
            except: pass
            html = self.download_url(url).text
            english |= self.is_english(html)
            if not english: raise BaseException
            keywords_found = any(re.search(regex, html.lower()) for regex in self.keywords)
            if keywords_found:
                to_add = set(self.get_linked_urls(url, html))
                added = self.add_urls_to_visit(to_add)
                with self.visit_lock:
                    logging.info(f"Frontier size: {len(self.urls_to_visit)}, "
                                 f"Added {added}/{len(to_add)} URLs from {url} to the frontier")
                    with open(self.out_path, "a") as f:
                        f.write(json.dumps({"url": url, "html": html}))
        except Exception as e:
            logging.warning(f"Crawler error: {e}")
        with self.visit_lock:
            self.visited_pages[url] = {
                'keywords_found': keywords_found,
                'is_english': english,
            }
            logging.info(
                f"Visited {len(self.visited_pages)} pages (english={english}, kws_found={keywords_found}, URL: {url})")

        with self.domain_lock:
            self.domain_dict[urlparse(url).hostname]['in_use'] = False

    def run(self, amount: int = None):
        start_url_amt = len(self.visited_pages)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = set()
            while True:
                with self.visit_lock:
                    if amount is not None and len(self.visited_pages) - start_url_amt >= amount: break
                    urls = list(self.urls_to_visit.copy())
                random.shuffle(urls)
                for next_url in urls:
                    if not len(futures) < self.max_workers: break
                    with self.visit_lock:
                        if next_url in self.visited_pages:
                            self.urls_to_visit.remove(next_url)
                            continue
                    domain = urlparse(next_url).hostname
                    with self.domain_lock:
                        if self.domain_dict[domain].get('in_use'): continue
                        self.domain_dict[domain]['in_use'] = True
                    with self.visit_lock: self.urls_to_visit.remove(next_url)
                    futures.add((executor.submit(self.crawl, next_url), time.time()))
                if not futures: break
                # wait x seconds to start more workers
                time.sleep(2)
                now = time.time()
                futures.difference_update(set(filter(lambda f: f[0].done() or now - f[1] > 60, futures)))
                logging.debug(f"Active threads {len(executor._threads)}")
        logging.info("Crawling complete.")
