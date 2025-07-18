

import asyncio
import json
import os
import re
import time
from typing import List, Dict, Set, Tuple, Optional, TYPE_CHECKING
from collections import defaultdict
from urllib.parse import urljoin
import requests
from difflib import SequenceMatcher
import threading
import concurrent.futures

if TYPE_CHECKING:
    from typing import Self


class TrieNode:
    """Trie node for efficient prefix matching"""
    
    def __init__(self):
        self.children = {}
        self.is_end_word = False
        self.priority = 0
        self.word = ""
    
    def insert(self, word: str, priority: int = 1):
        """Insert a word into the trie"""
        node = self
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_word = True
        node.priority = max(node.priority, priority)  # Keep highest priority
        node.word = word
    
    def search_prefix(self, prefix: str, max_results: int = 10) -> List[Tuple[str, int]]:
        """Search for words with given prefix"""
        node = self
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        # Collect all words from this node
        results = []
        self._collect_words(node, results, max_results)
        
        # Sort by priority and return
        results.sort(key=lambda x: (-x[1], len(x[0])))
        return results[:max_results]
    
    def _collect_words(self, node: "TrieNode", results: List[Tuple[str, int]], max_results: int):
        """Collect all words from a node"""
        if len(results) >= max_results:
            return
        
        if node.is_end_word:
            results.append((node.word, node.priority))
        
        for child in node.children.values():
            if len(results) >= max_results:
                break
            self._collect_words(child, results, max_results)


class AutocompleteEngine:
    """
    Advanced autocomplete engine that mimics Google's search suggestions.
    Features:
    - Multiple dictionary sources (English, German, Tübingen-specific)
    - Intelligent ranking based on popularity, relevance, and context
    - Fuzzy matching and typo correction
    - Recent searches integration
    - Caching for performance
    - Asynchronous loading
    """

    def __init__(self, cache_dir: str = "autocomplete_cache"):
        self.cache_dir = cache_dir
        self.dictionaries = {}
        self.word_frequencies = defaultdict(int)
        self.recent_searches = []
        self.popular_queries = set()
        self.trie = TrieNode()
        self.is_loaded = False
        self.loading_lock = threading.Lock()
        
        # Configuration for history limits
        self.max_recent_searches = 4    # Maximum stored in memory
        self.max_history_display = 4     # Maximum shown in suggestions
        self.max_total_suggestions = 8   # Maximum total suggestions shown
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Dictionary sources
        self.dictionary_sources = {
            'english_words': 'https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt',
            'tübingen_terms': self._get_tübingen_terms()
        }
        
        self._start_background_loading()

    def _get_tübingen_terms(self) -> List[str]:
        """Get Tübingen-specific terms and locations VIA CHATGPT!"""
        return [
            # Locations and attractions
            'tübingen', 'tuebingen', 'tubingen',
            'hohentübingen castle', 'hohentuebingen castle',
            'neckar river', 'neckar',
            'old town', 'altstadt',
            'market square', 'marktplatz',
            'stiftskirche', 'collegiate church',
            'botanical garden', 'botanischer garten',
            'university of tübingen', 'eberhard karls university',
            'schloss hohentübingen',
            'hölderlinturm', 'hoelderlinturm',
            'stocherkahn', 'punting',
            'platanenallee',
            'wilhelmstraße', 'wilhelmstrasse',
            'ammergasse',
            'judengasse',
            'kornhausstraße', 'kornhausstrasse',
            
            # Food and dining
            'swabian cuisine', 'schwäbische küche',
            'spätzle', 'spaetzle',
            'maultaschen',
            'brezn', 'pretzel',
            'weisswurst', 'weißwurst',
            'beer garden', 'biergarten',
            'gasthof', 'gasthaus',
            'bäckerei', 'baeckerei',
            'konditorei',
            'restaurant',
            'cafe', 'café',
            'kneipe', 'pub',
            'wurstbude',
            
            # Culture and events
            'festival', 'fest',
            'museum', 'museen',
            'theater',
            'concert', 'konzert',
            'art gallery', 'galerie',
            'cultural event', 'kulturelle veranstaltung',
            'stadtfest',
            'weihnachtsmarkt', 'christmas market',
            'kunsthalle',
            'city library', 'stadtbibliothek',
            
            # Academic and university
            'university', 'universität',
            'students', 'studenten',
            'campus',
            'faculty', 'fakultät',
            'library', 'bibliothek',
            'research', 'forschung',
            'lecture', 'vorlesung',
            'seminar',
            'dissertation',
            'professor',
            'institute', 'institut',
            
            # Transportation and practical
            'bus', 'bahn', 'train',
            'parking', 'parkplatz',
            'hotel', 'accommodation',
            'tourist information', 'touristeninformation',
            'city hall', 'rathaus',
            'post office', 'post',
            'bank',
            'pharmacy', 'apotheke',
            'hospital', 'krankenhaus',
            'shopping', 'einkaufen',
            'supermarket',
            
            # Common search patterns
            'things to do', 'sehenswürdigkeiten',
            'attractions', 'attraktionen',
            'sightseeing', 'besichtigung',
            'tourist guide', 'reiseführer',
            'opening hours', 'öffnungszeiten',
            'admission', 'eintritt',
            'price', 'preis',
            'map', 'karte',
            'directions', 'wegbeschreibung',
            'contact', 'kontakt',
            'website', 'webseite',
            'phone', 'telefon',
            'address', 'adresse',
            'reviews', 'bewertungen',
            'photos', 'fotos',
            'history', 'geschichte',
            'architecture', 'architektur',
            'medieval', 'mittelalterlich',
            'historic', 'historisch',
            'traditional', 'traditionell',
            'modern',
            'beautiful', 'schön',
            'famous', 'berühmt',
            'popular', 'beliebt',
            'best', 'beste',
            'top',
            'recommended', 'empfohlen'
        ]

    def _start_background_loading(self):
        """Start loading dictionaries in background thread"""
        def load_async():
            try:
                self._load_dictionaries()
                self._build_trie()
                with self.loading_lock:
                    self.is_loaded = True
            except Exception as e:
                print(f"Error loading dictionaries: {e}")
        
        thread = threading.Thread(target=load_async, daemon=True)
        thread.start()

    def _load_dictionaries(self):
        """Load dictionaries from various sources"""
        print("Loading autocomplete dictionaries...")
        
        # Load Tübingen-specific terms first (highest priority)
        self.dictionaries['tübingen_terms'] = self._get_tübingen_terms()
        
        # Load popular search patterns
        self.dictionaries['popular_patterns'] = self._get_popular_patterns()
        
        # Try to load online dictionaries
        for name, url in self.dictionary_sources.items():
            if name == 'tübingen_terms':
                continue  # Already loaded
                
            cache_file = os.path.join(self.cache_dir, f"{name}.txt")
            
            # Check if cached version exists and is recent (1 day)
            if os.path.exists(cache_file) and time.time() - os.path.getmtime(cache_file) < 86400:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        words = [line.strip().lower() for line in f if line.strip()]
                        self.dictionaries[name] = words[:5000]  # Limit to 5000 words
                        print(f"Loaded {len(self.dictionaries[name])} words from {name} (cached)")
                except Exception as e:
                    print(f"Error loading cached {name}: {e}")
            else:
                # Download fresh dictionary
                try:
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        words = [line.strip().lower() for line in response.text.split('\n') if line.strip()]
                        self.dictionaries[name] = words[:5000]  # Limit to 5000 words
                        
                        # Cache the dictionary
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            f.write('\n'.join(words))
                        
                        print(f"Downloaded and cached {len(self.dictionaries[name])} words from {name}")
                except Exception as e:
                    print(f"Error downloading {name}: {e}")

    def _get_popular_patterns(self) -> List[str]:
        """Get popular search patterns and combinations"""
        base_terms = ['tübingen', 'tuebingen']
        categories = [
            'attractions', 'sights', 'places', 'things to do',
            'food', 'restaurants', 'dining', 'cafes',
            'hotels', 'accommodation', 'stay',
            'events', 'festivals', 'culture',
            'university', 'campus', 'students',
            'history', 'historic', 'medieval',
            'castle', 'church', 'museum',
            'river', 'neckar', 'boat',
            'shopping', 'market', 'square',
            'nightlife', 'bars', 'pubs',
            'transportation', 'parking', 'bus',
            'tours', 'guide', 'tourist',
            'photos', 'pictures', 'images',
            'map', 'directions', 'location',
            'weather', 'climate',
            'best', 'top', 'recommended'
        ]
        
        patterns = []
        for base in base_terms:
            for category in categories:
                patterns.append(f"{base} {category}")
                patterns.append(f"{category} {base}")
                patterns.append(f"{category} in {base}")
        
        # Add common question patterns
        question_patterns = [
            'what to do in tübingen',
            'where to eat in tübingen',
            'how to get to tübingen',
            'when to visit tübingen',
            'why visit tübingen',
            'best time to visit tübingen',
            'tübingen opening hours',
            'tübingen admission prices',
            'tübingen contact information'
        ]
        
        patterns.extend(question_patterns)
        return patterns

    def _build_trie(self):
        """Build trie structure for fast prefix matching"""
        self.trie = TrieNode()
        
        # Add all dictionary words to trie with priorities
        priority_order = ['tübingen_terms', 'popular_patterns', 'english_words']
        
        for dict_name in priority_order:
            if dict_name in self.dictionaries:
                priority = len(priority_order) - priority_order.index(dict_name)
                for word in self.dictionaries[dict_name]:
                    if word and len(word) > 2:  # Only add words longer than 2 characters
                        self.trie.insert(word.lower(), priority)

    def add_recent_search(self, query: str):
        """Add a recent search to the autocomplete history"""
        if query and query.strip():
            query = query.strip().lower()
            # Remove if already exists
            if query in self.recent_searches:
                self.recent_searches.remove(query)
            # Add to beginning
            self.recent_searches.insert(0, query)
            # Keep only configured number of recent searches
            self.recent_searches = self.recent_searches[:self.max_recent_searches]
            
            # Also add to trie with high priority
            if hasattr(self, 'trie'):
                self.trie.insert(query, 10)  # High priority for recent searches

    def get_suggestions(self, query: str, max_results: int = 8) -> List[Dict]:
        """
        Get intelligent autocomplete suggestions for a query.
        Returns a list of suggestion dictionaries with metadata.
        """
        if not query or len(query) < 1:
            return self._get_default_suggestions(max_results)
        
        query = query.lower().strip()
        suggestions = []
        
        # Recent searches (highest priority)
        recent_matches = [s for s in self.recent_searches if query in s]
        for match in recent_matches[:3]:  # Max 3 recent matches
            suggestions.append({
                'text': match,
                'type': 'recent',
                'priority': 10,
                'icon': '🕒'
            })
        
        # Exact prefix matches from trie
        if hasattr(self, 'trie') and self.trie:
            trie_suggestions = self.trie.search_prefix(query, max_results - len(suggestions))
            for word, priority in trie_suggestions:
                if word not in [s['text'] for s in suggestions]:
                    suggestions.append({
                        'text': word,
                        'type': 'dictionary',
                        'priority': priority,
                        'icon': '🔍'
                    })
        
        # Fuzzy matching for typo correction
        if len(suggestions) < max_results:
            fuzzy_matches = self._get_fuzzy_matches(query, max_results - len(suggestions))
            for match in fuzzy_matches:
                if match['text'] not in [s['text'] for s in suggestions]:
                    suggestions.append(match)
        
        # Smart completion patterns
        if len(suggestions) < max_results:
            pattern_matches = self._get_pattern_matches(query, max_results - len(suggestions))
            for match in pattern_matches:
                if match['text'] not in [s['text'] for s in suggestions]:
                    suggestions.append(match)
        
        # Sort by priority and relevance
        suggestions.sort(key=lambda x: (-x['priority'], len(x['text'])))
        
        return suggestions[:max_results]

    def _get_default_suggestions(self, max_results: int) -> List[Dict]:
        """Get default suggestions when no query is provided"""
        defaults = []
        
        # Recent searches (limited by max_history_display)
        for search in self.recent_searches[:self.max_history_display]:
            defaults.append({
                'text': search,
                'type': 'recent',
                'priority': 10,
                'icon': '🕒'
            })
        
        # Popular Tübingen searches
        popular = [
            'tübingen attractions',
            'things to do in tübingen',
            'tübingen restaurants',
            'hohentübingen castle',
            'tübingen university',
            'tübingen old town',
            'neckar river tübingen',
            'tübingen hotels'
        ]
        
        for term in popular:
            if len(defaults) >= max_results:
                break
            if term not in [s['text'] for s in defaults]:
                defaults.append({
                    'text': term,
                    'type': 'popular',
                    'priority': 8,
                    'icon': '🔥'
                })
        
        return defaults[:max_results]

    def _get_fuzzy_matches(self, query: str, max_results: int) -> List[Dict]:
        """Get fuzzy matches for typo correction"""
        if not hasattr(self, 'trie') or not self.trie or len(query) < 3:
            return []
        
        fuzzy_matches = []
        
        # Check Tübingen-specific terms for fuzzy matches
        if 'tübingen_terms' in self.dictionaries:
            for term in self.dictionaries['tübingen_terms']:
                if term and len(term) > 2:
                    similarity = SequenceMatcher(None, query, term).ratio()
                    if similarity > 0.6:  # 60% similarity threshold
                        fuzzy_matches.append({
                            'text': term,
                            'type': 'fuzzy',
                            'priority': int(similarity * 7),  # Priority based on similarity
                            'icon': '✨'
                        })
        
        # Sort by similarity and return top matches
        fuzzy_matches.sort(key=lambda x: -x['priority'])
        return fuzzy_matches[:max_results]

    def _get_pattern_matches(self, query: str, max_results: int) -> List[Dict]:
        """Get smart pattern-based completions"""
        patterns = []
        
        # Common completion patterns
        if 'tübingen' in query or 'tuebingen' in query:
            completions = [
                'tübingen attractions',
                'tübingen restaurants',
                'tübingen hotels',
                'tübingen university',
                'tübingen castle',
                'tübingen old town',
                'tübingen things to do',
                'tübingen map'
            ]
            for completion in completions:
                if query in completion and len(patterns) < max_results:
                    patterns.append({
                        'text': completion,
                        'type': 'pattern',
                        'priority': 6,
                        'icon': '💡'
                    })
        
        # Question patterns
        question_starters = ['what', 'where', 'how', 'when', 'why']
        if any(query.startswith(starter) for starter in question_starters):
            question_completions = [
                'what to do in tübingen',
                'where to eat in tübingen',
                'how to get to tübingen',
                'when to visit tübingen',
                'why visit tübingen'
            ]
            for completion in question_completions:
                if query in completion and len(patterns) < max_results:
                    patterns.append({
                        'text': completion,
                        'type': 'question',
                        'priority': 5,
                        'icon': '❓'
                    })
        
        return patterns

    def is_ready(self) -> bool:
        """Check if the autocomplete engine is ready to use"""
        with self.loading_lock:
            return self.is_loaded


# Global autocomplete engine instance
_autocomplete_engine = None
_engine_lock = threading.Lock()


def get_autocomplete_engine() -> AutocompleteEngine:
    """Get the global autocomplete engine instance (singleton)"""
    global _autocomplete_engine
    if _autocomplete_engine is None:
        with _engine_lock:
            if _autocomplete_engine is None:
                _autocomplete_engine = AutocompleteEngine()
    return _autocomplete_engine
