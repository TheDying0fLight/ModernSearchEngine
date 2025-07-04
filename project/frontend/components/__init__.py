"""
Frontend components for the Tübingen Search Engine.
This module contains all reusable UI components organized by functionality.
"""

from .help_tab import HelpTab
from .favorites_tab import FavoritesTab
from .search_tab import SearchTab
from .history_tab import HistoryTab
from .history_container import HistoryContainer, HistoryItem
from .results_view import ResultsView, ResultCard
from .loading_indicator import LoadingIndicator
from .suggestions_view import SuggestionsView

__all__ = [
    'HelpTab',
    'FavoritesTab', 
    'SearchTab',
    'HistoryTab',
    'HistoryContainer',
    'HistoryItem',
    'ResultsView',
    'ResultCard',
    'LoadingIndicator',
    'SuggestionsView'
]
