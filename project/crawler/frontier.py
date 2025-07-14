from dataclasses import dataclass, field
import time
from typing import List, Dict, Set, Optional, Any
from urllib.parse import urljoin, urlparse

@dataclass
class Frontier:
    frontier: List[tuple] = field(default_factory=list)
    visited_pages: Set[str] = field(default_factory=set)





@dataclass
class URL:
    url: str
    parent_url: Optional[str] = None
    priority: int = 0
    visited: bool = False
    timestamp: float = field(default_factory=time.time)
    path_depth: int = 0

    def __post_init__(self):
        parsed = urlparse(self.url)
        self.path_depth = len([p for p in parsed.path.split('/') if p])
        
