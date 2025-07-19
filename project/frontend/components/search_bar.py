import flet as ft
from .suggestions_view import SuggestionsView
from .autocomplete_engine import get_autocomplete_engine


class SearchBar(ft.SearchBar):
    def __init__(self, search_func):
        self.search_func = search_func
        self.autocomplete_engine = get_autocomplete_engine()
        self.suggestions_view = SuggestionsView(on_suggestion_click=self.handle_suggestion_submit)
        
        # Enhanced styling to match Google's design
        super().__init__(
            width=600,
            height=48,
            view_elevation=8,
            divider_color=ft.Colors.BLUE_300,
            bar_hint_text="Search for anything about Tübingen...",
            view_hint_text="Search suggestions",
            bar_leading=ft.Icon(ft.Icons.SEARCH, color=ft.Colors.BLUE_600, size=20),
            view_leading=ft.Icon(ft.Icons.SEARCH, color=ft.Colors.BLUE_600, size=20),
            bar_trailing=[],
            on_change=self.handle_change,
            on_submit=self.handle_submit,
            on_tap=self.handle_tap,
            controls=[self.suggestions_view.get_container()],
            # Enhanced styling
            bar_bgcolor=ft.Colors.WHITE,
            bar_overlay_color=ft.Colors.GREY_50,
            view_bgcolor=ft.Colors.WHITE,
            view_surface_tint_color=ft.Colors.BLUE_50,
        )

    def handle_change(self, e: ft.ControlEvent):
        """Handle search input changes with intelligent autocomplete"""
        query = e.data.strip() if e.data else ""
        
        # Update suggestions with new query
        self.suggestions_view.update_suggestions(query)
        
        # Auto-open suggestions for non-empty queries
        if query and not self.view_open:
            self.open_view()
        
        self.update()

    def handle_submit(self, e: ft.ControlEvent):
        """Handle search submission"""
        if e.data and e.data.strip():
            self.submit(e.data.strip())

    def handle_suggestion_submit(self, suggestion):
        """Handle suggestion selection"""
        self.submit(suggestion)


    def submit(self, data):
        """Submit search query with enhanced feedback"""
        self.close_view()
        self.value = data
        self.update()

        # Add to recent searches in autocomplete engine
        if self.autocomplete_engine.is_ready():
            self.autocomplete_engine.add_recent_search(data)
        
        # Also update suggestions view
        self.suggestions_view.add_recent_search(data)

        # Call search function
        if self.search_func:
            self.search_func(data)

    def handle_tap(self, e):
        """Handle search bar tap to open suggestions"""
        current_query = self.value.strip() if self.value else ""
        
        # Update suggestions for current query
        self.suggestions_view.update_suggestions(current_query)
        
        # Open suggestions view
        self.open_view()

    def get_search_suggestions(self, query: str = ""):
        """Get search suggestions for external use"""
        if self.autocomplete_engine.is_ready():
            return self.autocomplete_engine.get_suggestions(query, max_results=10)
        return []

    def clear_search(self):
        """Clear the search bar"""
        self.value = ""
        self.close_view()
        self.update()

    def set_search_query(self, query: str):
        """Set search query programmatically"""
        self.value = query
        self.update()

    @property
    def view_open(self):
        """Check if suggestions view is open"""
        # This is a property that should be available in ft.SearchBar
        # If not, we'll implement a workaround
        return getattr(self, '_view_open', False)

    def open_view(self):
        """Open the suggestions view"""
        self._view_open = True
        super().open_view()

    def close_view(self):
        """Close the suggestions view"""
        self._view_open = False
        super().close_view()