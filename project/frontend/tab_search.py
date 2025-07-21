import threading
import time
import flet as ft
import re
from .components import LoadingIndicator, EmptyState, SearchBar, get_autocomplete_engine
from .results_view import ResultsView, Result
from.tab import Tab

class SearchTab(Tab):
    """Main search tab component containing search interface and results"""

    def __init__(self, page: ft.Page, clustering_options: list[str], on_favorite_toggle=None):
        self.page = page
        self.on_favorite_toggle = on_favorite_toggle
        self.autocomplete_engine = get_autocomplete_engine()

        # Autocomplete status indicator
        self.autocomplete_status = ft.Row([
            ft.Icon(ft.Icons.SMART_TOY, color=ft.Colors.GREEN_500, size=16),
            ft.Text("Smart suggestions ready", size=12, color=ft.Colors.GREEN_600)
        ], alignment=ft.MainAxisAlignment.CENTER, spacing=5)

        # Initialize components
        self.search_bar = SearchBar(search_func=lambda query: self.page.go(f'/search?q={query}&c={self.header.get_cluster_option()}'))
        self.loading_indicator = LoadingIndicator("Searching documents...")
        self.results_view = ResultsView(self.handle_favorite_toggle, self.handle_result_click)
        self.header = SearchHeader(self.page, self.search_bar, self.autocomplete_status, clustering_options)

        # Check autocomplete status
        self._check_autocomplete_status()

        # Create the tab container
        super().__init__(
            text="Search",
            icon=ft.Icons.SEARCH,
            controls=[
                self.header,
                self.loading_indicator,
                self.results_view
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _check_autocomplete_status(self):
        """Check autocomplete engine status and update indicator"""
        if self.autocomplete_engine.is_ready():
            self.autocomplete_status.controls[0].name = ft.Icons.SMART_TOY
            self.autocomplete_status.controls[0].color = ft.Colors.GREEN_500
            self.autocomplete_status.controls[1].value = "Smart suggestions ready"
            self.autocomplete_status.controls[1].color = ft.Colors.GREEN_600
        else:
            self.autocomplete_status.controls[0].name = ft.Icons.HOURGLASS_EMPTY
            self.autocomplete_status.controls[0].color = ft.Colors.ORANGE_500
            self.autocomplete_status.controls[1].value = "Loading smart suggestions..."
            self.autocomplete_status.controls[1].color = ft.Colors.ORANGE_600
            # Check again in a bit
            self._schedule_status_check()

    def _schedule_status_check(self):
        def check_later():
            time.sleep(2)
            if hasattr(self, 'autocomplete_status'):
                self._check_autocomplete_status()
                if hasattr(self, 'autocomplete_status'):
                    self.autocomplete_status.update()
        thread = threading.Thread(target=check_later, daemon=True)
        thread.start()

    def handle_result_click(self, result_data: Result):
        """Handle result click"""
        self.page.launch_url(result_data.url)

    def handle_favorite_toggle(self, result_data: Result, is_favorited):
        """Handle favorite toggle"""
        if self.on_favorite_toggle:
            success = self.on_favorite_toggle(result_data, is_favorited)
            if success:
                action = "Added to" if is_favorited else "Removed from"
                print(f"{action} favorites: {result_data.title}")
        else:
            print(f"Favorite toggle for: {result_data.title}")

    def start_loading(self, query):
        self.loading_indicator.show(f"Searching for '{query}'...")

    def display_results(self, query, results):
        """Display search results"""
        self.loading_indicator.hide()
        self.results_view.show_results(query, results)

    def clear_results(self):
        """Clear current search results"""
        self.results_view.hide_results()

class SearchHeader(ft.Row):
    def __init__(self, page: ft.Page, search_bar: SearchBar, autocomplete_status, cluster_options = list[str]):
        self.side_width = 250
        self.page = page
        self.dropdown = ft.Dropdown(
            value=cluster_options[0],
            options=[ft.DropdownOption(s) for s in cluster_options], width=self.side_width,
            border_color=ft.Colors.GREY_400,
            text_style=ft.TextStyle(size=14, color=ft.Colors.GREY_600),
            on_change=lambda e: self.on_cluster_option_change()
        )

        super().__init__([
            ft.Column([], width=self.side_width),
            ft.Column([
                ft.Text(
                    "🔍 Tübingen Search",
                    size=32,
                    weight=ft.FontWeight.BOLD,
                    color=ft.Colors.BLUE_800
                ),
                ft.Text(
                    "Discover Tübingen's history, culture, and attractions",
                    size=16,
                    color=ft.Colors.GREY_600,
                    text_align=ft.TextAlign.CENTER
                ),
                search_bar,
                autocomplete_status
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, expand=True),
            ft.Column(
                [ft.Text('Choose Clustering Algorithm', size=14, color=ft.Colors.GREY_600), self.dropdown],
                width=self.side_width,
                alignment=ft.MainAxisAlignment.START,
                horizontal_alignment=ft.CrossAxisAlignment.START),
            ], vertical_alignment=ft.CrossAxisAlignment.START, spacing=0)

    def get_cluster_option(self):
        return self.dropdown.value

    def on_cluster_option_change(self):
        pattern = r'(search\?q=.*&c=)(.*)'
        new_route = re.sub(pattern, lambda m: m.group(1) + self.get_cluster_option(), self.page.route)
        self.page.go(new_route)