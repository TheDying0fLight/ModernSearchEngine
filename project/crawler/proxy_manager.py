import logging
import random
import requests
from bs4 import BeautifulSoup
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

txt_urls = [
    "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text",
    "https://raw.githubusercontent.com/mmpx12/proxy-list/refs/heads/master/proxies.txt",
]

class ProxyManager:
    def __init__(
        self,
        test_url='https://httpbin.org/ip',
        timeout=2,
        max_workers=20,
        cooldown=1.0,
        entry_ttl=86400
    ):
        self.test_url = test_url
        self.timeout = timeout
        self.max_workers = max_workers
        self.cooldown = cooldown
        self.entry_ttl = entry_ttl
        self.all_proxies = {}
        self.proxies = []
        self._lock = threading.RLock()
        self._refresh_proxies(init=True)

    def _get_proxy_list(self, init=False):
        proxy_list = set()
        url = "https://free-proxy-list.net/"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        textarea = soup.find('textarea', class_='form-control')
        if textarea:
            lines = textarea.text.strip().splitlines()
            proxy_list.update([f"http://{line.strip()}" for line in lines])

        if init:
            for url in txt_urls:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.text, 'html.parser')
                for s in str(soup).split('\n'):
                    proxy_list.update(str(s).split('\r'))

        proxy_list = [p for p in proxy_list if re.search('://', p) and not re.search('[a-zA-Z]', p[6:])]
        return proxy_list

    def _cleanup_proxies(self):
        now = time.time()
        with self._lock:
            to_remove = [url for url, meta in self.all_proxies.items() if now - meta['added_at'] > self.entry_ttl]
            for url in to_remove:
                logging.info(f"Removing stale proxy {url} (older than {self.entry_ttl} seconds)")
                self.all_proxies.pop(url, None)
            self.proxies = [p for p in self.proxies if p['proxy'] in self.all_proxies]

    def _refresh_proxies(self, init=False):
        with self._lock:
            logging.info("Cleaning up old proxies...")
            self._cleanup_proxies()
            logging.info("Fetching proxies list...")

            raw_list = self._get_proxy_list(init)
            logging.info(f"Found {len(raw_list)} candidate proxies. Testing...")

            valid = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_map = {executor.submit(self._test_proxy, proxy): proxy for proxy in raw_list}
                for future in as_completed(future_map):
                    proxy_url, elapsed = future.result()
                    if proxy_url and not self.all_proxies.get(proxy_url):
                        now = time.time()
                        quality_score = 1.0 / elapsed if elapsed and elapsed > 0 else 1.0
                        entry = {
                            'proxy': proxy_url,
                            'quality': quality_score,
                            'fail_count': 0,
                            'last_used': 0.0,
                            'added_at': now
                        }
                        self.all_proxies[proxy_url] = entry
                        valid.append(entry)
            self.proxies.extend(valid)
            logging.info(f"{len(self.proxies)} proxies valid after testing.")

    def _test_proxy(self, proxy_url):
        proxies = {'http': proxy_url, 'https': proxy_url}
        start = time.time()
        try:
            resp = requests.get(self.test_url, proxies=proxies, timeout=self.timeout)
            resp.raise_for_status()
            return proxy_url, time.time() - start
        except Exception:
            return None, None

    def update_proxy_quality(self, proxy_url, success, elapsed=None):
        with self._lock:
            entry = self.all_proxies.get(proxy_url)
            if not entry:
                return
            if success and elapsed is not None:
                score = 1.0 / elapsed if elapsed > 0 else entry['quality']
                entry['quality'] = 0.7 * entry['quality'] + 0.3 * score
            else:
                entry['fail_count'] += 1
                logging.info(f"Proxy {proxy_url} failed {entry['fail_count']} time(s)")
            if entry['fail_count'] >= 2:
                logging.info(f"Dropping proxy {proxy_url} after {entry['fail_count']} failures")
                self.all_proxies.pop(proxy_url, None)
                self.proxies = [p for p in self.proxies if p['proxy'] != proxy_url]

    def get_random_proxy(self):
        now = time.time()
        with self._lock:
            available = [p for p in self.proxies if now - p['last_used'] >= self.cooldown]
            if not available:
                logging.info("No proxies available due to cooldown, refreshing proxy list...")
                self._refresh_proxies()
                now = time.time()
                available = [p for p in self.proxies if now - p['last_used'] >= self.cooldown]
                if not available:
                    raise RuntimeError("No proxies available to use after refresh.")
            total_quality = sum(p['quality'] for p in available)
            weights = [p['quality'] / total_quality for p in available] if total_quality > 0 else None
            chosen = random.choices(available, weights=weights)[0]
            chosen['last_used'] = now
            return chosen