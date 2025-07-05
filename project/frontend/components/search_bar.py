import flet as ft
from .suggestions_view import SuggestionsView


class SearchBar(ft.SearchBar):
    """Enhanced search bar with component-based suggestions"""

    def __init__(self, search_func):
        self.search_func = search_func
        self.suggestions_view = SuggestionsView(on_suggestion_click=self.handle_suggestion_submit)

        super().__init__(
            width=500,
            view_elevation=8,
            divider_color=ft.Colors.BLUE_300,
            bar_hint_text="Search documents about Tübingen...",
            view_hint_text="Search suggestions",
            bar_leading=ft.Icon(ft.Icons.SEARCH, color=ft.Colors.BLUE_600),
            view_leading=ft.Icon(ft.Icons.SEARCH, color=ft.Colors.BLUE_600),
            on_change=self.handle_change,
            on_submit=self.handle_submit,
            on_tap=self.handle_tap,
            controls=[self.suggestions_view.get_container()],
        )

    def handle_change(self, e: ft.ControlEvent):
        """Handle search input changes with improved filtering"""
        query = e.data.lower().strip() if e.data else ""
        # Update suggestions based on query
        self.suggestions_view.update_suggestions(query)
        self.update()

    def handle_submit(self, e: ft.ControlEvent):
        """Handle search submission"""
        if e.data and e.data.strip():
            self.submit(e.data.strip())

    def handle_suggestion_submit(self, suggestion):
        """Handle suggestion selection"""
        self.submit(suggestion)

    def submit(self, data):
        """Submit search query"""
        self.close_view()
        self.value = data
        self.update()

        # Add to recent searches
        self.suggestions_view.add_recent_search(data)

        # Call search function
        if self.search_func:
            self.search_func(data)

    def handle_tap(self, e):
        """Handle search bar tap to open suggestions"""
        current_query = self.value.lower().strip() if self.value else ""
        self.suggestions_view.update_suggestions(current_query)
        self.open_view()
