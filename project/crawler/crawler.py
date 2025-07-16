from urllib.parse import urljoin, urlparse, ParseResult
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from collections import defaultdict
from langdetect import detect, LangDetectException
from .proxy_manager import ProxyManager
from langcodes import Language
from termcolor import colored
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

from .utils import predict_language_from_url, uniquify
from .utils import TrackingThreadPoolExecutor, TimeoutRobotFileParser

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


class Crawler:
    def __init__(self,
                 urls: list,
                 max_workers: int = multiprocessing.cpu_count() // 2,
                 keywords: list = [r't\S{1,4}bingen'],
                 user_agents: list = default_user_agents,
                 use_proxies: bool = True,
                 verbose: bool = False,
                 out_path: str = "data/crawled.jsonl"):
        self.verbose = verbose
        self.use_proxies = use_proxies
        self.keywords = [re.compile(kw) for kw in keywords]
        self.user_agents = user_agents
        self.max_workers = max_workers
        self.out_path = uniquify(out_path)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

        self.proxy_manager = ProxyManager(max_workers=max_workers, verbose=False) if use_proxies else None
        self.allowed_lang_prefixes = ['en']
        self.domain_lock = threading.Lock()
        self.visit_lock = threading.Lock()
        self.json_lock = threading.Lock()
        self.domain_dict = defaultdict(lambda: defaultdict(int))

        self.visited_pages = {}
        self.urls_to_visit = set(urls)

    def get_domain(self, parse: ParseResult): return parse.hostname.split(".")[-2:]

    def get_random_headers(self):
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }

    def is_captcha_page(self, html):
        markers = ['captcha', 'g-recaptcha', 'hcaptcha', 'confirm this search was made by a human']
        text = BeautifulSoup(html, PARSER).get_text(separator=' ').lower()
        return any(m in text for m in markers)

    def download_url(self, url: str, headers_only: bool = False):
        last_exc = None
        attempt = 0
        domain = self.get_domain(urlparse(url))
        request_fn = requests.head if headers_only else requests.get

        with self.domain_lock:
            last_time = self.domain_dict[domain]['last_access']
            delay = self.domain_dict[domain]['delay']
        if last_time:
            now = time.time()
            delay += DEFAULT_DELAY + random.uniform(0, 2)
            wait_time = last_time + delay - now
            if wait_time > 0: time.sleep(wait_time)
        with self.domain_lock:
            self.domain_dict[domain]['last_access'] = time.time()

        if not self.use_proxies:
            try:
                resp = request_fn(url, headers=self.get_random_headers(), timeout=3)
                if resp.status_code == 429:
                    with self.domain_lock:
                        self.domain_dict[domain]["delay"] += 1
                        delay = self.domain_dict[domain]["delay"]
                    with self.visit_lock: self.urls_to_visit.add(url)
                    logging.warning(colored(f'429 Received: Delay increased to {delay}s for: {domain}', 'red'))
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
                resp = request_fn(url, proxies=proxies, headers=headers_only, timeout=3)
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
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('/'): href = yield urljoin(url, href)
            elif href.startswith('http'): yield href

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

    def add_urls_to_visit(self, urls: list[str]):
        added = 0
        language_denied = []
        for url in set(urls):
            # may remove interesting queries
            # parse = urlparse(url)
            # url = urljoin(url, parse.path)

            if not validators.url(url): continue
            t = mimetypes.guess_type(url)[0]
            if str(t).split("/")[0] not in ["None", "text"]: continue
            with self.visit_lock:
                if url in self.urls_to_visit: continue
                if url in self.visited_pages: continue
            if predict_language_from_url(url) not in ["und", "en"]:
                language_denied.append(url)
                continue
            if not self.is_useragent_allowed(url): continue
            with self.visit_lock: self.urls_to_visit.add(url)
            added += 1
        if language_denied: logging.info(f"Assumed non english: {language_denied}")
        return added

    def is_english(self, soup: BeautifulSoup):
        text = soup.get_text(separator=' ')
        try:
            lang = detect(text)
            return lang == 'en'
        except LangDetectException:
            logging.info("Language detection failed, assuming English")
            return True

    def crawl(self, url: str):
        try:
            keywords_found = False
            english = False
            if self.verbose: logging.info(f"Crawling {url}")
            headers = self.download_url(url, headers_only=True).headers
            try: english = Language.get(headers['content-language']).language == 'en'
            except: pass
            html = self.download_url(url).text
            if len(html.encode()) > 1e8: raise Exception(f"Page to big, Bytes: {len(html.encode())}, {url}")
            soup = BeautifulSoup(html, PARSER)
            english |= self.is_english(soup)
            if not english: raise BaseException
            keywords_found = any(regex.search(html.lower()) for regex in self.keywords)
            if keywords_found:
                to_add = set(self.get_linked_urls(url, soup))
                added = self.add_urls_to_visit(to_add)
                with self.visit_lock:
                    logging.info(f"Frontier size: {len(self.urls_to_visit)}, "
                                 f"Added {added}/{len(to_add)} URLs from {url}")
                with self.json_lock:
                    with open(self.out_path, "a") as f:
                        f.write(json.dumps({"url": url, "html": html}))
        except Exception as e:
            logging.warning(f"Crawler error: {e}")
        finally:
            with self.visit_lock:
                self.visited_pages[url] = {
                    'keywords_found': keywords_found,
                    'is_english': english,
                }
                logging.debug(
                    f"Visited {len(self.visited_pages)} pages (english={english}, kws_found={keywords_found}, URL: {url})")

        domain = self.get_domain(urlparse(url))
        with self.domain_lock:
            self.domain_dict[domain]['in_use'] = False

    def run(self, amount: int = None):
        start_url_amt = len(self.visited_pages)
        with TrackingThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = set()
            while True:
                prior_time = time.time()
                if not amount or len(self.visited_pages) - start_url_amt <= amount:
                    with self.visit_lock:
                        urls = list(self.urls_to_visit.copy())
                    futures.update(self.schedule_urls(urls, executor))
                if not futures: break
                # wait x seconds to start more workers
                post_time = time.time()
                diff = post_time - prior_time
                if diff < 2: time.sleep(2 - diff)
                futures.difference_update(set(filter(lambda f: f[0].done(), futures)))
                visiting = list(map(lambda x: x[2], futures))
                with self.visit_lock:
                    logging.info(colored(f'Active threads: {executor.active_count}, '
                                         f'Visited: {len(self.visited_pages)}, Sites: {visiting}', 'green'))
        logging.info("Crawling complete.")

    def schedule_urls(self, urls: list[str], executor: TrackingThreadPoolExecutor):
        futures = set()
        random.shuffle(urls)
        for next_url in urls:
            domain = self.get_domain(urlparse(next_url))
            with self.domain_lock:
                if self.domain_dict[domain].get('in_use'): continue
                self.domain_dict[domain]['in_use'] = True
            with self.visit_lock: self.urls_to_visit.remove(next_url)
            futures.add((executor.submit(self.crawl, next_url), time.time(), next_url))
        return futures